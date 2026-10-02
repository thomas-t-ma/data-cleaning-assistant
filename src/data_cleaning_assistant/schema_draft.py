from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from .actions import ActionType
from .engine import profile_dataset
from .inference import looks_like_identifier_name
from .io import load_dataset
from .planner import file_sha256, propose_actions
from .study_schema import StudySchema


class SchemaSuggestionKind(str, Enum):
    GRAIN_KEY = "grain_key"
    CANONICAL_MAPPING = "canonical_mapping"
    MISSING_TOKEN = "missing_token"
    FOREIGN_KEY = "foreign_key"


class SchemaSuggestionDecision(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"


class SchemaSuggestion(BaseModel):
    id: str
    kind: SchemaSuggestionKind
    table: str
    column: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    proposed: dict[str, Any] = Field(default_factory=dict)
    rationale: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    decision: SchemaSuggestionDecision = SchemaSuggestionDecision.PROPOSED


class SchemaDraft(BaseModel):
    version: int = 1
    source_path: str
    source_sha256: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    suggestions: list[SchemaSuggestion] = Field(default_factory=list)


def _stable_suggestion_id(
    kind: SchemaSuggestionKind,
    table: str,
    column: str | None,
    proposed: dict[str, Any],
) -> str:
    payload = json.dumps(
        {
            "kind": kind.value,
            "table": table,
            "column": column,
            "proposed": proposed,
        },
        sort_keys=True,
        default=str,
    )
    suffix = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"schema_{suffix}"


def _suggestion(
    *,
    kind: SchemaSuggestionKind,
    table: str,
    column: str | None,
    confidence: float,
    proposed: dict[str, Any],
    rationale: str,
    evidence: dict[str, Any] | None = None,
) -> SchemaSuggestion:
    return SchemaSuggestion(
        id=_stable_suggestion_id(kind, table, column, proposed),
        kind=kind,
        table=table,
        column=column,
        confidence=confidence,
        proposed=proposed,
        rationale=rationale,
        evidence=evidence or {},
    )


def build_schema_draft(path: str | Path) -> SchemaDraft:
    """Build a non-authoritative schema draft from existing deterministic evidence."""

    resolved = Path(path).expanduser().resolve()
    report = profile_dataset(resolved)
    plan = propose_actions(resolved)
    suggestions: list[SchemaSuggestion] = []

    for table in report.tables:
        if table.candidate_keys:
            candidate = table.candidate_keys[0]
            if candidate.confidence >= 0.95:
                proposed = {"columns": list(candidate.columns)}
                suggestions.append(
                    _suggestion(
                        kind=SchemaSuggestionKind.GRAIN_KEY,
                        table=table.name,
                        column=None,
                        confidence=candidate.confidence,
                        proposed=proposed,
                        rationale=(
                            "The strongest candidate key is highly complete and "
                            "nearly unique. Confirm that it represents the intended "
                            "row grain before adopting it."
                        ),
                        evidence={
                            "uniqueness": candidate.uniqueness,
                            "completeness": candidate.completeness,
                        },
                    )
                )

    for action in plan.actions:
        if action.column is None:
            continue
        if looks_like_identifier_name(action.column):
            continue

        if action.type == ActionType.NORMALIZE_CATEGORIES:
            mapping = action.parameters.get("mapping")
            if not isinstance(mapping, dict):
                continue
            for source, target in mapping.items():
                proposed = {
                    "source": str(source),
                    "target": str(target),
                }
                suggestions.append(
                    _suggestion(
                        kind=SchemaSuggestionKind.CANONICAL_MAPPING,
                        table=action.table,
                        column=action.column,
                        confidence=action.confidence,
                        proposed=proposed,
                        rationale=(
                            "Deterministic profiling found an equivalent-looking "
                            "category variant. Confirm that the source and target "
                            "mean exactly the same thing before making this part of "
                            "the study dictionary."
                        ),
                        evidence={"issue_ids": list(action.issue_ids)},
                    )
                )

        elif action.type == ActionType.NORMALIZE_MISSING:
            values = action.parameters.get("values")
            if not isinstance(values, list):
                continue
            for token in values:
                proposed = {"token": str(token)}
                suggestions.append(
                    _suggestion(
                        kind=SchemaSuggestionKind.MISSING_TOKEN,
                        table=action.table,
                        column=action.column,
                        confidence=action.confidence,
                        proposed=proposed,
                        rationale=(
                            "Deterministic profiling identified this value as a "
                            "structural null sentinel. Confirm that it means missing "
                            "for this specific field."
                        ),
                        evidence={"issue_ids": list(action.issue_ids)},
                    )
                )

    for relation in report.relationships:
        if relation.confidence < 0.90:
            continue

        if relation.relationship == "many-to-one":
            child_table = relation.left_table
            child_column = relation.left_column
            parent_table = relation.right_table
            parent_column = relation.right_column
        elif relation.relationship == "one-to-many":
            child_table = relation.right_table
            child_column = relation.right_column
            parent_table = relation.left_table
            parent_column = relation.left_column
        else:
            continue

        proposed = {
            "child_table": child_table,
            "child_column": child_column,
            "parent_table": parent_table,
            "parent_column": parent_column,
        }
        suggestions.append(
            _suggestion(
                kind=SchemaSuggestionKind.FOREIGN_KEY,
                table=child_table,
                column=child_column,
                confidence=relation.confidence,
                proposed=proposed,
                rationale=(
                    "The same identifier-like field links these tables with a "
                    "high-confidence many-to-one relationship. Confirm the parent "
                    "table before declaring a foreign key."
                ),
                evidence={
                    "relationship": relation.relationship,
                    "shared_value_count": relation.shared_value_count,
                    "left_coverage": relation.left_coverage,
                    "right_coverage": relation.right_coverage,
                },
            )
        )

    deduped = {suggestion.id: suggestion for suggestion in suggestions}
    ordered = sorted(
        deduped.values(),
        key=lambda item: (
            item.kind.value,
            item.table,
            item.column or "",
            item.id,
        ),
    )
    return SchemaDraft(
        source_path=str(resolved),
        source_sha256=file_sha256(resolved),
        suggestions=ordered,
    )


def materialize_study_schema(
    draft: SchemaDraft,
    *,
    dataset_path: str | Path | None = None,
) -> StudySchema:
    """Compile explicitly approved draft suggestions into a real StudySchema."""

    tables: dict[str, dict[str, Any]] = {}
    foreign_keys: list[dict[str, str]] = []

    def table_payload(name: str) -> dict[str, Any]:
        return tables.setdefault(name, {"columns": {}})

    def column_payload(table: str, column: str) -> dict[str, Any]:
        payload = table_payload(table)
        columns = payload.setdefault("columns", {})
        return columns.setdefault(column, {})

    for suggestion in draft.suggestions:
        if suggestion.decision != SchemaSuggestionDecision.APPROVED:
            continue

        if suggestion.kind == SchemaSuggestionKind.GRAIN_KEY:
            payload = table_payload(suggestion.table)
            columns = suggestion.proposed.get("columns")
            if not isinstance(columns, list) or not columns:
                raise ValueError(
                    f"Invalid approved grain-key suggestion {suggestion.id!r}."
                )
            if payload.get("grain_key") and payload["grain_key"] != columns:
                raise ValueError(
                    f"Multiple approved grain keys conflict for {suggestion.table!r}."
                )
            payload["grain_key"] = list(columns)

        elif suggestion.kind == SchemaSuggestionKind.CANONICAL_MAPPING:
            if suggestion.column is None:
                raise ValueError(
                    f"Canonical mapping {suggestion.id!r} has no column."
                )
            source = suggestion.proposed.get("source")
            target = suggestion.proposed.get("target")
            if not isinstance(source, str) or not isinstance(target, str):
                raise ValueError(
                    f"Invalid canonical mapping suggestion {suggestion.id!r}."
                )
            payload = column_payload(suggestion.table, suggestion.column)
            mapping = payload.setdefault("canonical_map", {})
            previous = mapping.get(source)
            if previous is not None and previous != target:
                raise ValueError(
                    f"Approved schema suggestions map {source!r} to both "
                    f"{previous!r} and {target!r}."
                )
            mapping[source] = target

        elif suggestion.kind == SchemaSuggestionKind.MISSING_TOKEN:
            if suggestion.column is None:
                raise ValueError(
                    f"Missing-token suggestion {suggestion.id!r} has no column."
                )
            token = suggestion.proposed.get("token")
            if not isinstance(token, str):
                raise ValueError(
                    f"Invalid missing-token suggestion {suggestion.id!r}."
                )
            payload = column_payload(suggestion.table, suggestion.column)
            values = payload.setdefault("missing_tokens", [])
            if token not in values:
                values.append(token)

        elif suggestion.kind == SchemaSuggestionKind.FOREIGN_KEY:
            required = (
                "child_table",
                "child_column",
                "parent_table",
                "parent_column",
            )
            proposed = suggestion.proposed
            if not all(isinstance(proposed.get(key), str) for key in required):
                raise ValueError(
                    f"Invalid foreign-key suggestion {suggestion.id!r}."
                )
            item = {key: str(proposed[key]) for key in required}
            if item not in foreign_keys:
                foreign_keys.append(item)

    payload = {
        "version": 1,
        "tables": tables,
        "foreign_keys": foreign_keys,
    }
    schema = StudySchema.model_validate(payload)

    if dataset_path is not None:
        loaded = load_dataset(dataset_path)
        schema.validate_dataset(loaded.tables)

    return schema


def load_schema_draft(path: str | Path) -> SchemaDraft:
    resolved = Path(path).expanduser().resolve()
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("Schema draft must be a YAML/JSON mapping.")
    return SchemaDraft.model_validate(raw)


def write_schema_draft(
    draft: SchemaDraft,
    output_path: str | Path,
) -> Path:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = draft.model_dump(mode="json")
    output.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return output


def write_study_schema(
    schema: StudySchema,
    output_path: str | Path,
) -> Path:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = schema.model_dump(mode="json", exclude_none=True)
    output.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return output
