from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .actions import ActionType, CleaningAction, CleaningPlan, PlanningNote
from .engine import profile_dataset
from .inference import looks_like_identifier_name
from .io import load_dataset
from .planning_policy import structural_only_column
from .study_schema import StudySchema


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_action_id(
    action_type: ActionType,
    table: str,
    column: str | None,
    parameters: dict[str, Any],
) -> str:
    payload = json.dumps(
        {
            "type": action_type.value,
            "table": table,
            "column": column,
            "parameters": parameters,
        },
        sort_keys=True,
        default=str,
    )
    suffix = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"action_{suffix}"


def _canonical_category(frame: pd.DataFrame, column: str, variants: list[str]) -> str:
    counts = Counter(str(value) for value in frame[column].tolist())
    return max(
        variants,
        key=lambda value: (
            counts[value],
            int(value == value.strip()),
            int(value[:1].isupper()),
            -len(value),
            value,
        ),
    ).strip()


def _canonical_multiselect_component(
    frame: pd.DataFrame,
    column: str,
    variants: list[str],
    delimiter: str,
) -> str | None:
    counts: Counter[str] = Counter()
    for value in frame[column].tolist():
        for component in str(value).split(delimiter):
            component = component.strip()
            if component:
                counts[component] += 1

    highest_count = max(counts[value] for value in variants)
    winners = [value for value in variants if counts[value] == highest_count]
    if len(winners) != 1:
        return None
    return winners[0].strip()


def _multiselect_action(issue, frame: pd.DataFrame) -> CleaningAction | None:
    if issue.column is None:
        return None
    delimiter = str(issue.evidence.get("delimiter", ";"))
    groups = issue.evidence.get("variant_groups", [])
    mapping: dict[str, str] = {}
    for raw_group in groups:
        variants = [str(value) for value in raw_group]
        if len(variants) < 2:
            continue
        canonical = _canonical_multiselect_component(
            frame,
            issue.column,
            variants,
            delimiter,
        )
        if canonical is None:
            continue
        for variant in variants:
            if variant != canonical:
                mapping[variant] = canonical
    if not mapping:
        return None

    parameters = {"delimiter": delimiter, "mapping": mapping}
    return CleaningAction(
        id=_stable_action_id(
            ActionType.NORMALIZE_MULTISELECT,
            issue.table,
            issue.column,
            parameters,
        ),
        type=ActionType.NORMALIZE_MULTISELECT,
        table=issue.table,
        column=issue.column,
        title="Normalize multi-select component labels",
        description=(
            "Normalize capitalization variants inside structured "
            "multi-select responses while preserving every selected component. "
            "Review the component mapping before approval."
        ),
        parameters=parameters,
        confidence=issue.confidence,
        issue_ids=[issue.id],
    )


def _category_action(issue, frame: pd.DataFrame) -> CleaningAction | None:
    if issue.column is None:
        return None
    groups = issue.evidence.get("variant_groups", [])
    mapping: dict[str, str] = {}
    for raw_group in groups:
        variants = [str(value) for value in raw_group]
        if len(variants) < 2:
            continue
        canonical = _canonical_category(frame, issue.column, variants)
        for variant in variants:
            if variant != canonical:
                mapping[variant] = canonical
    if not mapping:
        return None

    parameters = {"mapping": mapping}
    return CleaningAction(
        id=_stable_action_id(
            ActionType.NORMALIZE_CATEGORIES,
            issue.table,
            issue.column,
            parameters,
        ),
        type=ActionType.NORMALIZE_CATEGORIES,
        table=issue.table,
        column=issue.column,
        title="Normalize equivalent-looking category labels",
        description=(
            "Replace capitalization/whitespace variants with a single proposed label. "
            "Review the mapping before approval."
        ),
        parameters=parameters,
        confidence=issue.confidence,
        issue_ids=[issue.id],
    )


def _missing_action(issue) -> CleaningAction | None:
    if issue.column is None:
        return None
    source = issue.evidence.get("structural_tokens")
    if source is None:
        source = issue.evidence.get("tokens", {})
    tokens = sorted(str(value) for value in source)
    if not tokens:
        return None
    parameters = {"values": tokens}
    return CleaningAction(
        id=_stable_action_id(
            ActionType.NORMALIZE_MISSING,
            issue.table,
            issue.column,
            parameters,
        ),
        type=ActionType.NORMALIZE_MISSING,
        table=issue.table,
        column=issue.column,
        title="Normalize textual missing-value tokens",
        description=(
            "Replace structural null sentinels with true null values. Contextual labels such "
            "as Unknown or Not applicable are not included without an explicit study schema."
        ),
        parameters=parameters,
        confidence=issue.confidence,
        issue_ids=[issue.id],
    )


def _date_action(issue) -> CleaningAction | None:
    if issue.column is None:
        return None
    parameters = {"format": "%Y-%m-%d", "unparseable": "keep"}
    return CleaningAction(
        id=_stable_action_id(
            ActionType.STANDARDIZE_DATES,
            issue.table,
            issue.column,
            parameters,
        ),
        type=ActionType.STANDARDIZE_DATES,
        table=issue.table,
        column=issue.column,
        title="Standardize parseable date values",
        description=(
            "Convert parseable dates to ISO YYYY-MM-DD while leaving unparseable values unchanged."
        ),
        parameters=parameters,
        confidence=issue.confidence,
        issue_ids=[issue.id],
    )


def _numeric_action(issue) -> CleaningAction | None:
    if issue.column is None or looks_like_identifier_name(issue.column):
        return None
    parameters = {"unparseable": "keep"}
    return CleaningAction(
        id=_stable_action_id(
            ActionType.COERCE_NUMERIC,
            issue.table,
            issue.column,
            parameters,
        ),
        type=ActionType.COERCE_NUMERIC,
        table=issue.table,
        column=issue.column,
        title="Convert parseable values to numbers",
        description=(
            "Convert values that can be parsed numerically while leaving unparseable values unchanged."
        ),
        parameters=parameters,
        confidence=issue.confidence,
        issue_ids=[issue.id],
    )


def _trim_action(issue) -> CleaningAction | None:
    if issue.column is None:
        return None
    parameters: dict[str, Any] = {}
    return CleaningAction(
        id=_stable_action_id(
            ActionType.TRIM_WHITESPACE,
            issue.table,
            issue.column,
            parameters,
        ),
        type=ActionType.TRIM_WHITESPACE,
        table=issue.table,
        column=issue.column,
        title="Trim surrounding whitespace",
        description=(
            "Remove leading and trailing whitespace while leaving internal spacing unchanged."
        ),
        parameters=parameters,
        confidence=issue.confidence,
        issue_ids=[issue.id],
    )


def _duplicate_action(issue) -> CleaningAction:
    parameters = {"keep": "first"}
    return CleaningAction(
        id=_stable_action_id(
            ActionType.REMOVE_EXACT_DUPLICATES,
            issue.table,
            None,
            parameters,
        ),
        type=ActionType.REMOVE_EXACT_DUPLICATES,
        table=issue.table,
        column=None,
        title="Remove exact duplicate rows",
        description=(
            "Keep the first copy of each fully identical row and remove later copies. "
            "This is never auto-approved because repeated identical rows can be legitimate observations."
        ),
        parameters=parameters,
        confidence=issue.confidence,
        issue_ids=[issue.id],
    )


def _schema_actions(
    tables: dict[str, pd.DataFrame],
    schema: StudySchema,
) -> list[CleaningAction]:
    """Only explicit schema transformations, each still requiring human approval."""
    schema.validate_dataset(tables)
    actions: list[CleaningAction] = []
    for table, table_spec in schema.tables.items():
        for column, column_spec in table_spec.columns.items():
            observed = {str(value) for value in tables[table][column].tolist()}
            mapping = {
                old: new
                for old, new in column_spec.canonical_map.items()
                if old in observed and old != new
            }
            if mapping:
                parameters = {"mapping": mapping}
                actions.append(
                    CleaningAction(
                        id=_stable_action_id(
                            ActionType.NORMALIZE_CATEGORIES, table, column, parameters
                        ),
                        type=ActionType.NORMALIZE_CATEGORIES,
                        table=table,
                        column=column,
                        title="Study-dictionary canonicalization",
                        description=(
                            "Exact source-to-canonical mapping explicitly declared in the "
                            "researcher's study schema. Review before approval."
                        ),
                        parameters=parameters,
                        confidence=1.0,
                        issue_ids=[],
                    )
                )

            tokens = sorted(set(column_spec.missing_tokens) & observed)
            if tokens:
                parameters = {"values": tokens}
                actions.append(
                    CleaningAction(
                        id=_stable_action_id(
                            ActionType.NORMALIZE_MISSING, table, column, parameters
                        ),
                        type=ActionType.NORMALIZE_MISSING,
                        table=table,
                        column=column,
                        title="Study-defined missing-token normalization",
                        description=(
                            "Only the tokens explicitly defined as null by this study schema; "
                            "review before approval."
                        ),
                        parameters=parameters,
                        confidence=1.0,
                        issue_ids=[],
                    )
                )
    return actions


def propose_actions(
    path: str | Path,
    *,
    schema: StudySchema | None = None,
) -> CleaningPlan:
    resolved = Path(path).expanduser().resolve()
    loaded = load_dataset(resolved)
    report = profile_dataset(resolved, schema=schema)
    actions: list[CleaningAction] = []
    notes: list[PlanningNote] = []

    for issue in report.issues:
        if issue.table not in loaded.tables:
            continue
        frame = loaded.tables[issue.table]
        action: CleaningAction | None = None

        if issue.type == "category_inconsistency":
            action = _category_action(issue, frame)
        elif issue.type == "multiselect_component_inconsistency":
            action = _multiselect_action(issue, frame)
        elif issue.type == "hidden_missing_values":
            if structural_only_column(issue, frame):
                notes.append(
                    PlanningNote(
                        table=issue.table,
                        column=issue.column,
                        issue_ids=[issue.id],
                        reason="structural_only_column",
                    )
                )
                continue
            action = _missing_action(issue)
        elif issue.type == "malformed_or_mixed_dates":
            action = _date_action(issue)
        elif issue.type == "mixed_numeric_values":
            action = _numeric_action(issue)
        elif issue.type == "surrounding_whitespace":
            action = _trim_action(issue)
        elif issue.type == "exact_duplicate_rows":
            action = _duplicate_action(issue)

        if action is not None:
            actions.append(action)

    if schema is not None:
        # An explicit study declaration wins over heuristic proposals for the
        # same target and action type; never mix incompatible category mappings.
        declared = _schema_actions(loaded.tables, schema)
        controlled = {(a.type, a.table, a.column) for a in declared}
        category_controlled = {
            (table, column)
            for table, table_spec in schema.tables.items()
            for column, column_spec in table_spec.columns.items()
            if column_spec.allowed_values is not None or column_spec.canonical_map
        }
        actions = [
            action
            for action in actions
            if (action.type, action.table, action.column) not in controlled
            and not (
                action.type
                in {
                    ActionType.NORMALIZE_CATEGORIES,
                    ActionType.NORMALIZE_MULTISELECT,
                }
                and (action.table, action.column) in category_controlled
            )
        ]
        actions.extend(declared)
        # A study-dictionary declaration explicitly authorizes an otherwise
        # suppressed all-structural column; never mislabel it as omitted.
        declared_targets = {
            (action.table, action.column)
            for action in declared
            if action.type == ActionType.NORMALIZE_MISSING
        }
        notes = [
            note for note in notes
            if (note.table, note.column) not in declared_targets
        ]

    deduped = {action.id: action for action in actions}
    return CleaningPlan(
        source_path=str(resolved),
        source_sha256=file_sha256(resolved),
        actions=list(deduped.values()),
        planning_notes=notes,
        study_schema_sha256=schema.source_sha256 if schema else None,
    )
