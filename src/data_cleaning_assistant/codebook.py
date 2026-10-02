from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from .inference import (
    infer_semantic_type,
    is_missing_like,
    looks_like_free_text_response,
    looks_like_identifier_name,
    normalize_text,
)
from .study_schema import ColumnSpec, StudySchema, TableSpec


_COUNT_NAME_PATTERN = re.compile(r"\b(how many|count|number)\b", re.IGNORECASE)


class CodebookSuggestion(BaseModel):
    table: str
    column: str
    meaningful_count: int = Field(ge=0)
    observed_unique_count: int = Field(ge=0)
    core_coverage: float = Field(ge=0.0, le=1.0)
    core_values: list[str] = Field(default_factory=list)
    core_counts: dict[str, int] = Field(default_factory=dict)
    canonical_map: dict[str, str] = Field(default_factory=dict)
    unresolved_variant_groups: list[list[str]] = Field(default_factory=list)
    rare_values: list[str] = Field(default_factory=list)
    review_required: bool = True


class CodebookSuggestionReport(BaseModel):
    version: int = 1
    source_path: str
    source_sha256: str
    interpretation_note: str = (
        "Observed codebook suggestions are review aids, not study-schema authority. "
        "Rare values may be valid write-ins, and suggested canonical mappings cover "
        "only exact case/whitespace variants."
    )
    suggestions: list[CodebookSuggestion] = Field(default_factory=list)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _core_values(
    counts: Counter[str],
    *,
    coverage_target: float,
    max_core_values: int,
) -> tuple[list[str], float]:
    total = sum(counts.values())
    chosen: list[str] = []
    covered = 0
    for value, count in sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        if len(chosen) >= max_core_values:
            break
        chosen.append(value)
        covered += count
        if total and covered / total >= coverage_target:
            break
    return chosen, (covered / total if total else 0.0)


def suggest_codebooks(
    path: str | Path,
    tables: dict[str, pd.DataFrame],
    *,
    coverage_target: float = 0.95,
    minimum_coverage: float = 0.90,
    max_core_values: int = 12,
    max_observed_values: int = 100,
    rare_preview_limit: int = 20,
) -> CodebookSuggestionReport:
    """Suggest dominant observed codebooks without making them authoritative."""
    resolved = Path(path).expanduser().resolve()
    suggestions: list[CodebookSuggestion] = []

    for table, frame in tables.items():
        for raw_column in frame.columns:
            column = str(raw_column)
            series = frame[raw_column]
            if looks_like_identifier_name(column):
                continue
            if looks_like_free_text_response(column, series):
                continue

            inferred_type, _confidence, _ratios = infer_semantic_type(column, series)
            if inferred_type in {"identifier", "numeric", "date"}:
                continue
            if _COUNT_NAME_PATTERN.search(column):
                continue

            values = [
                str(value).strip()
                for value in series.tolist()
                if not is_missing_like(value) and str(value).strip()
            ]
            if len(values) < 20:
                continue

            delimited_count = sum(";" in value for value in values)
            if delimited_count >= max(5, math.ceil(len(values) * 0.01)):
                continue

            counts: Counter[str] = Counter(values)
            if not 2 <= len(counts) <= max_observed_values:
                continue

            core, coverage = _core_values(
                counts,
                coverage_target=coverage_target,
                max_core_values=max_core_values,
            )
            if coverage < minimum_coverage:
                continue

            normalized_groups: dict[str, list[str]] = defaultdict(list)
            for value in counts:
                normalized_groups[normalize_text(value)].append(value)

            canonical_map: dict[str, str] = {}
            unresolved: list[list[str]] = []
            for variants in normalized_groups.values():
                if len(variants) < 2:
                    continue
                highest = max(counts[value] for value in variants)
                winners = [
                    value for value in variants
                    if counts[value] == highest
                ]
                if len(winners) != 1:
                    unresolved.append(sorted(variants))
                    continue
                canonical = winners[0]
                for variant in variants:
                    if variant != canonical:
                        canonical_map[variant] = canonical

            canonical_core = []
            for value in core:
                canonical = canonical_map.get(value, value)
                if canonical not in canonical_core:
                    canonical_core.append(canonical)
            if len(canonical_core) < 2:
                continue

            core_set = set(core)
            rare = [
                value
                for value, _count in sorted(
                    counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
                if value not in core_set and value not in canonical_map
            ][:rare_preview_limit]

            canonical_core_counts: Counter[str] = Counter()
            for value, count in counts.items():
                canonical = canonical_map.get(value, value)
                if canonical in canonical_core:
                    canonical_core_counts[canonical] += count

            suggestions.append(
                CodebookSuggestion(
                    table=table,
                    column=column,
                    meaningful_count=len(values),
                    observed_unique_count=len(counts),
                    core_coverage=coverage,
                    core_values=canonical_core,
                    core_counts=dict(canonical_core_counts),
                    canonical_map=canonical_map,
                    unresolved_variant_groups=unresolved,
                    rare_values=rare,
                )
            )

    suggestions.sort(
        key=lambda item: (
            item.table,
            -item.core_coverage,
            len(item.core_values),
            item.column,
        )
    )
    return CodebookSuggestionReport(
        source_path=str(resolved),
        source_sha256=_sha256(resolved),
        suggestions=suggestions,
    )



def load_codebook_suggestion_report(path: str | Path) -> CodebookSuggestionReport:
    """Load a previously generated read-only codebook suggestion report."""
    resolved = Path(path).expanduser().resolve()
    import yaml

    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Codebook suggestion report must contain a YAML mapping.")
    return CodebookSuggestionReport.model_validate(payload)


def promote_codebook_mappings(
    report: CodebookSuggestionReport,
    tables: dict[str, pd.DataFrame],
    approved_targets: set[tuple[str, str]],
) -> StudySchema:
    """Promote explicitly approved canonical maps into a StudySchema.

    Promotion deliberately does not infer allowed_values from frequency-derived
    core values. Rare/write-in values may be scientifically valid. Only exact
    case/whitespace canonical mappings already present in reviewed suggestions
    become schema authority.
    """
    suggestions = {
        (suggestion.table, suggestion.column): suggestion
        for suggestion in report.suggestions
    }
    unknown = approved_targets - set(suggestions)
    if unknown:
        formatted = ", ".join(
            f"{table}::{column}" for table, column in sorted(unknown)
        )
        raise ValueError(f"Unknown codebook suggestion target(s): {formatted}")

    if not approved_targets:
        raise ValueError(
            "No codebook targets were approved. "
            "Select at least one reviewed TABLE::COLUMN target."
        )

    table_specs: dict[str, TableSpec] = {}
    for table, column in sorted(approved_targets):
        suggestion = suggestions[(table, column)]
        if not suggestion.canonical_map:
            raise ValueError(
                f"{table}::{column} has no unambiguous canonical mapping to promote."
            )
        table_spec = table_specs.setdefault(table, TableSpec())
        table_spec.columns[column] = ColumnSpec(
            canonical_map=dict(suggestion.canonical_map)
        )

    schema = StudySchema(version=1, tables=table_specs)
    schema.validate_dataset(tables)
    return schema
