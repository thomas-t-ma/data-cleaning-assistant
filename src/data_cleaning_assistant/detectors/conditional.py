from __future__ import annotations

import re
import uuid

import pandas as pd

from ..inference import is_missing_like, normalize_text
from ..models import Issue, Severity


_TRIGGER_PATTERN = re.compile(
    r"^\s*(?:\d+\.\s*)?if\s+(yes|no)\b",
    re.IGNORECASE,
)
_YES = {"yes", "y", "true", "1"}
_NO = {"no", "n", "false", "0"}
_UNCERTAIN = {
    "unclear",
    "unsure",
    "unknown",
    "not sure",
    "not applicable",
    "n/a",
}


def _id() -> str:
    return f"issue_{uuid.uuid4().hex[:12]}"


def _response_state(value) -> str | None:
    normalized = normalize_text(value)
    if normalized in _YES:
        return "yes"
    if normalized in _NO:
        return "no"
    return None


def _binary_parent_score(series: pd.Series) -> float:
    values = [
        normalize_text(value)
        for value in series.tolist()
        if not is_missing_like(value) and str(value).strip()
    ]
    if len(values) < 3:
        return 0.0
    recognized = sum(
        value in _YES or value in _NO or value in _UNCERTAIN
        for value in values
    )
    if not any(value in _YES or value in _NO for value in values):
        return 0.0
    return recognized / len(values)


def _has_required_detail(value) -> bool:
    if is_missing_like(value) or not str(value).strip():
        return False
    normalized = normalize_text(value)
    return normalized not in _YES and normalized not in _NO


def detect_conditional_consistency(
    table: str,
    frame: pd.DataFrame,
    *,
    search_previous_columns: int = 4,
) -> list[Issue]:
    """Flag contradictions in explicit If yes/no dependent fields.

    Relationships are inferred only from an explicit dependent-field phrase and
    a nearby yes/no-like parent column. Findings are QA-only and never create an
    executable cleaning action.
    """
    columns = [str(column) for column in frame.columns]
    issues: list[Issue] = []

    for dependent_index, dependent_column in enumerate(columns):
        match = _TRIGGER_PATTERN.match(dependent_column)
        if match is None:
            continue
        trigger_state = match.group(1).casefold()

        parent_column: str | None = None
        parent_score = 0.0
        distance = 0
        lower_bound = max(-1, dependent_index - search_previous_columns - 1)
        for parent_index in range(dependent_index - 1, lower_bound, -1):
            candidate = columns[parent_index]
            score = _binary_parent_score(frame[candidate])
            if score >= 0.80:
                parent_column = candidate
                parent_score = score
                distance = dependent_index - parent_index
                break
        if parent_column is None:
            continue

        contradiction_rows: list[int] = []
        missing_detail_rows: list[int] = []
        for row_position, (parent_value, dependent_value) in enumerate(
            zip(
                frame[parent_column].tolist(),
                frame[dependent_column].tolist(),
                strict=True,
            )
        ):
            parent_state = _response_state(parent_value)
            if parent_state is None:
                continue

            if parent_state == trigger_state:
                if not _has_required_detail(dependent_value):
                    missing_detail_rows.append(row_position)
                continue

            dependent_normalized = normalize_text(dependent_value)
            opposite_state_answer = (
                dependent_normalized in _YES
                if trigger_state == "yes"
                else dependent_normalized in _NO
            )
            if _has_required_detail(dependent_value) or opposite_state_answer:
                contradiction_rows.append(row_position)

        affected_rows = sorted(set(contradiction_rows + missing_detail_rows))
        if not affected_rows:
            continue

        issues.append(
            Issue(
                id=_id(),
                type="conditional_dependency_inconsistency",
                severity=(
                    Severity.MEDIUM if contradiction_rows else Severity.LOW
                ),
                table=table,
                column=dependent_column,
                title="Conditional field responses are inconsistent",
                description=(
                    f"{dependent_column!r} explicitly applies when the nearby "
                    f"parent field is {trigger_state!r}. Some rows either contain "
                    "a dependent value when the parent says otherwise or omit "
                    "the required detail when the trigger condition is met. "
                    "This is a review flag only; no values are changed."
                ),
                confidence=min(0.99, 0.90 + 0.09 * parent_score),
                affected_count=len(affected_rows),
                row_indices_sample=affected_rows[:10],
                evidence={
                    "parent_column": parent_column,
                    "dependent_column": dependent_column,
                    "trigger_state": trigger_state,
                    "parent_binary_score": parent_score,
                    "column_distance": distance,
                    "contradiction_count": len(contradiction_rows),
                    "missing_detail_count": len(missing_detail_rows),
                    "contradiction_rows_sample": contradiction_rows[:10],
                    "missing_detail_rows_sample": missing_detail_rows[:10],
                },
                suggested_action=(
                    "Review the parent/dependent responses together. Do not "
                    "auto-fill, clear, or recode either field."
                ),
                auto_fix_safe=False,
            )
        )

    return issues
