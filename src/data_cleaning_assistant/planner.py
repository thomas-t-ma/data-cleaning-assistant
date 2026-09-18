from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .actions import ActionType, CleaningAction, CleaningPlan
from .engine import profile_dataset
from .inference import looks_like_identifier_name
from .io import load_dataset


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_action_id(
    action_type: ActionType,
    table: str,
    column: str,
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


def _category_action(issue, frame: pd.DataFrame) -> CleaningAction | None:
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
    tokens = sorted(str(value) for value in issue.evidence.get("tokens", {}))
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
            "Replace the listed textual tokens with true null values. "
            "Approval is required because some study protocols use special missing codes."
        ),
        parameters=parameters,
        confidence=issue.confidence,
        issue_ids=[issue.id],
    )


def _date_action(issue) -> CleaningAction:
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
    if looks_like_identifier_name(issue.column):
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


def propose_actions(path: str | Path) -> CleaningPlan:
    resolved = Path(path).expanduser().resolve()
    loaded = load_dataset(resolved)
    report = profile_dataset(resolved)
    actions: list[CleaningAction] = []

    for issue in report.issues:
        if not issue.column or issue.table not in loaded.tables:
            continue
        frame = loaded.tables[issue.table]
        action: CleaningAction | None = None
        if issue.type == "category_inconsistency":
            action = _category_action(issue, frame)
        elif issue.type == "hidden_missing_values":
            action = _missing_action(issue)
        elif issue.type == "malformed_or_mixed_dates":
            action = _date_action(issue)
        elif issue.type == "mixed_numeric_values":
            action = _numeric_action(issue)

        if action is not None:
            actions.append(action)

    deduped = {action.id: action for action in actions}
    return CleaningPlan(
        source_path=str(resolved),
        source_sha256=file_sha256(resolved),
        actions=list(deduped.values()),
    )
