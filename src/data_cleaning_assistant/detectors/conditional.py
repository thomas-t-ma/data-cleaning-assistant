from __future__ import annotations

import re
import uuid

import pandas as pd

from ..inference import is_missing_like, normalize_text, parse_number
from ..models import Issue, Severity


_TRIGGER_PATTERN = re.compile(
    r"^\s*(?:\d+\.\s*)?if\s+(yes|no)\b",
    re.IGNORECASE,
)
_SKIP_PATTERN = re.compile(
    r"\bskip\b.*\bif\s+(yes|no)\b",
    re.IGNORECASE,
)
_TEXT_YES = {"yes", "y", "true"}
_TEXT_NO = {"no", "n", "false"}
_YES = _TEXT_YES | {"1"}
_NO = _TEXT_NO | {"0"}
_COUNT_PATTERN = re.compile(r"\b(how many|count|number)\b", re.IGNORECASE)
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
    return normalized not in _TEXT_YES and normalized not in _TEXT_NO


def _trigger_from_column(column: str) -> tuple[str, str] | None:
    direct = _TRIGGER_PATTERN.match(column)
    if direct is not None:
        return direct.group(1).casefold(), "explicit_if_prefix"

    skip = _SKIP_PATTERN.search(column)
    if skip is None:
        return None
    skipped_state = skip.group(1).casefold()
    trigger_state = "no" if skipped_state == "yes" else "yes"
    return trigger_state, "explicit_skip_instruction"


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
        trigger = _trigger_from_column(dependent_column)
        if trigger is None:
            continue
        trigger_state, relationship_basis = trigger

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

        count_like = _COUNT_PATTERN.search(dependent_column) is not None
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

            numeric_detail = (
                parse_number(dependent_value) if count_like else None
            )

            if parent_state == trigger_state:
                if (
                    count_like
                    and trigger_state == "yes"
                    and numeric_detail is not None
                    and numeric_detail <= 0
                ):
                    contradiction_rows.append(row_position)
                elif not _has_required_detail(dependent_value):
                    missing_detail_rows.append(row_position)
                continue

            if (
                count_like
                and trigger_state == "yes"
                and parent_state == "no"
                and numeric_detail == 0
            ):
                continue

            dependent_normalized = normalize_text(dependent_value)
            opposite_state_answer = (
                dependent_normalized in _TEXT_YES
                if trigger_state == "yes"
                else dependent_normalized in _TEXT_NO
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
                    "relationship_basis": relationship_basis,
                    "parent_binary_score": parent_score,
                    "column_distance": distance,
                    "count_like_dependent": count_like,
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
