from __future__ import annotations

import re
import uuid

import pandas as pd

from ..inference import is_missing_like, looks_like_free_text_response, normalize_text
from ..models import Issue, Severity


_DEPENDENT_PATTERN = re.compile(
    r"\bif\s+other\b|"
    r"\bother\b.*\b(specify|explain|describe|detail|details)\b|"
    r"\b(specify|explain|describe)\b.*\bother\b",
    re.IGNORECASE,
)
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = {
    "if", "other", "specify", "explain", "describe", "detail", "details",
    "please", "what", "which", "the", "a", "an", "of", "to", "for", "in",
    "on", "and", "or", "this", "that", "your", "you",
}


def _id() -> str:
    return f"issue_{uuid.uuid4().hex[:12]}"


def _semantic_tokens(column: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_PATTERN.findall(column)
        if token.casefold() not in _STOPWORDS and not token.isdigit()
    }


def _is_other_label(value: str) -> bool:
    normalized = normalize_text(value)
    return (
        normalized == "other"
        or normalized.startswith("other ")
        or normalized.startswith("other(")
        or normalized.startswith("other:")
        or normalized.startswith("other-")
    )


def _selects_other(value) -> bool:
    if is_missing_like(value) or not str(value).strip():
        return False
    return any(
        _is_other_label(component.strip())
        for component in str(value).split(";")
        if component.strip()
    )


def _has_detail(value) -> bool:
    return not is_missing_like(value) and bool(str(value).strip())


def _other_parent_score(name: str, series: pd.Series) -> float:
    if looks_like_free_text_response(name, series):
        return 0.0

    values = [
        str(value).strip()
        for value in series.tolist()
        if not is_missing_like(value) and str(value).strip()
    ]
    if len(values) < 3:
        return 0.0

    unique_values = set(values)
    unique_ratio = len(unique_values) / len(values)
    if len(unique_values) > 50 and unique_ratio > 0.50:
        return 0.0

    other_count = sum(_selects_other(value) for value in values)
    if other_count == 0:
        return 0.0

    return min(0.99, 0.90 + 0.09 * (other_count / len(values)))


def detect_other_specify_consistency(
    table: str,
    frame: pd.DataFrame,
    *,
    search_previous_columns: int = 4,
) -> list[Issue]:
    """Flag inconsistencies between an Other category and its write-in field.

    The dependent column must explicitly refer to Other and the parent must
    actually contain an Other option. Findings are QA-only and never executable.
    """
    columns = [str(column) for column in frame.columns]
    issues: list[Issue] = []

    for dependent_index, dependent_column in enumerate(columns):
        if _DEPENDENT_PATTERN.search(dependent_column) is None:
            continue

        dependent_tokens = _semantic_tokens(dependent_column)
        candidates: list[
            tuple[tuple[int, int, int, float, int], str, float, int, str]
        ] = []
        lower_bound = max(-1, dependent_index - search_previous_columns - 1)

        for parent_index in range(dependent_index - 1, lower_bound, -1):
            candidate = columns[parent_index]
            score = _other_parent_score(candidate, frame[candidate])
            if score <= 0.0:
                continue

            distance = dependent_index - parent_index
            lexical_overlap = len(dependent_tokens & _semantic_tokens(candidate))
            adjacent = distance == 1
            if not adjacent and lexical_overlap == 0:
                continue

            match_basis = (
                "lexical_overlap"
                if lexical_overlap > 0
                else "adjacent_other_category"
            )
            rank = (
                int(adjacent and lexical_overlap > 0),
                int(lexical_overlap > 0),
                int(adjacent),
                score,
                -distance,
            )
            candidates.append((rank, candidate, score, distance, match_basis))

        if not candidates:
            continue

        _rank, parent_column, parent_score, distance, match_basis = max(
            candidates,
            key=lambda item: item[0],
        )

        unexpected_detail_rows: list[int] = []
        missing_detail_rows: list[int] = []
        for row_position, (parent_value, dependent_value) in enumerate(
            zip(
                frame[parent_column].tolist(),
                frame[dependent_column].tolist(),
                strict=True,
            )
        ):
            if is_missing_like(parent_value) or not str(parent_value).strip():
                continue

            selected_other = _selects_other(parent_value)
            detail_present = _has_detail(dependent_value)
            if selected_other and not detail_present:
                missing_detail_rows.append(row_position)
            elif not selected_other and detail_present:
                unexpected_detail_rows.append(row_position)

        affected_rows = sorted(set(unexpected_detail_rows + missing_detail_rows))
        if not affected_rows:
            continue

        issues.append(
            Issue(
                id=_id(),
                type="other_specify_inconsistency",
                severity=(
                    Severity.MEDIUM if unexpected_detail_rows else Severity.LOW
                ),
                table=table,
                column=dependent_column,
                title="Other-category write-in responses are inconsistent",
                description=(
                    f"{dependent_column!r} appears to be a write-in field for "
                    f"{parent_column!r}. Some rows either select an Other option "
                    "without supplying detail or populate the write-in field when "
                    "Other is not selected. This is a review flag only; no values "
                    "are changed."
                ),
                confidence=parent_score,
                affected_count=len(affected_rows),
                row_indices_sample=affected_rows[:10],
                evidence={
                    "parent_column": parent_column,
                    "dependent_column": dependent_column,
                    "column_distance": distance,
                    "parent_match_basis": match_basis,
                    "parent_other_score": parent_score,
                    "unexpected_detail_count": len(unexpected_detail_rows),
                    "missing_detail_count": len(missing_detail_rows),
                    "unexpected_detail_rows_sample": unexpected_detail_rows[:10],
                    "missing_detail_rows_sample": missing_detail_rows[:10],
                },
                suggested_action=(
                    "Review the category and write-in response together. Do not "
                    "auto-fill, clear, or recode either field."
                ),
                auto_fix_safe=False,
            )
        )

    return issues
