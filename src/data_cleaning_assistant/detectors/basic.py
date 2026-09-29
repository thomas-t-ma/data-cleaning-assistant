from __future__ import annotations

import re
import statistics
import uuid
from collections import defaultdict

import pandas as pd

from ..inference import (
    CONTEXTUAL_MISSING_TOKENS,
    DATE_NAME_PATTERN,
    MISSING_TOKENS,
    STRUCTURAL_MISSING_TOKENS,
    infer_semantic_type,
    is_contextual_missing_label,
    is_empty_cell,
    is_missing_like,
    is_null,
    looks_like_free_text_response,
    looks_like_identifier_name,
    normalize_text,
    parse_date,
    parse_number,
)
from ..models import Issue, Severity


def _id() -> str:
    return f"issue_{uuid.uuid4().hex[:12]}"


def _sample_rows(mask: list[bool], limit: int = 10) -> list[int]:
    return [index for index, matched in enumerate(mask) if matched][:limit]


def _hidden_missing(table: str, column: str, series: pd.Series) -> list[Issue]:
    mask: list[bool] = []
    counts: dict[str, int] = defaultdict(int)
    for value in series.tolist():
        if is_empty_cell(value):
            mask.append(False)
            continue
        normalized = normalize_text(value)
        hit = normalized in MISSING_TOKENS
        mask.append(hit)
        if hit:
            counts[str(value)] += 1

    affected = sum(mask)
    if not affected:
        return []

    structural = {
        token: count
        for token, count in counts.items()
        if normalize_text(token) in STRUCTURAL_MISSING_TOKENS
    }
    contextual = {
        token: count
        for token, count in counts.items()
        if normalize_text(token) in CONTEXTUAL_MISSING_TOKENS
    }

    description = (
        "One or more textual tokens look like missing values but are not encoded as null."
    )
    if contextual:
        description += (
            " Contextual labels such as Unknown or Not applicable may be meaningful study states "
            "and must not be collapsed to null without explicit schema information."
        )

    return [
        Issue(
            id=_id(),
            type="hidden_missing_values",
            severity=Severity.MEDIUM,
            table=table,
            column=column,
            title="Text values may represent missing data",
            description=description,
            confidence=0.97 if structural else 0.75,
            affected_count=affected,
            row_indices_sample=_sample_rows(mask),
            evidence={
                "tokens": dict(counts),
                "structural_tokens": structural,
                "contextual_tokens": contextual,
            },
            suggested_action=(
                "Normalize only structural null sentinels by default; preserve contextual "
                "missingness labels unless a study schema explicitly defines them as null."
            ),
            auto_fix_safe=False,
        )
    ]


def _constant_column(table: str, column: str, series: pd.Series) -> list[Issue]:
    values = [value for value in series.tolist() if not is_empty_cell(value)]
    if not values or len({repr(value) for value in values}) != 1 or len(series) < 2:
        return []
    return [
        Issue(
            id=_id(),
            type="constant_column",
            severity=Severity.LOW,
            table=table,
            column=column,
            title="Column is constant",
            description="Every non-null observation has the same value.",
            confidence=1.0,
            affected_count=len(series),
            evidence={"value": values[0]},
            suggested_action="Consider removing the column if it carries no analytic information.",
        )
    ]


def _surrounding_whitespace(table: str, column: str, series: pd.Series) -> list[Issue]:
    mask = [
        isinstance(value, str)
        and not is_missing_like(value)
        and value != value.strip()
        for value in series.tolist()
    ]
    affected = sum(mask)
    if not affected:
        return []

    examples = []
    for value, matched in zip(series.tolist(), mask, strict=False):
        if matched and value not in examples:
            examples.append(value)
        if len(examples) >= 10:
            break

    return [
        Issue(
            id=_id(),
            type="surrounding_whitespace",
            severity=Severity.LOW,
            table=table,
            column=column,
            title="Values contain leading or trailing whitespace",
            description=(
                "Some text values contain surrounding whitespace. Trimming can improve exact matching "
                "without changing internal spaces."
            ),
            confidence=0.99,
            affected_count=affected,
            row_indices_sample=_sample_rows(mask),
            evidence={"examples": examples},
            suggested_action="Review and trim surrounding whitespace.",
            auto_fix_safe=False,
        )
    ]


def _category_variants(table: str, column: str, series: pd.Series) -> list[Issue]:
    if looks_like_free_text_response(column, series):
        return []

    non_null = [value for value in series.tolist() if not is_missing_like(value)]
    if not non_null:
        return []

    # Structured multi-select columns are normalized at component level. Whole-
    # value case normalization can be correct only accidentally and cannot reuse
    # a component mapping across different option combinations.
    if sum(";" in str(value) for value in non_null) >= 2:
        return []

    raw_unique = {str(value) for value in non_null}
    if len(raw_unique) > 200:
        return []

    # Sparse free-text fields can look categorical when most rows are missing.
    # Suppress cosmetic case/whitespace normalization when almost every
    # meaningful response is unique; those values are prose, not a codebook.
    if (
        len(raw_unique) > 10
        and len(raw_unique) / len(non_null) > 0.50
    ):
        return []

    groups: dict[str, set[str]] = defaultdict(set)
    for value in raw_unique:
        groups[normalize_text(value)].add(value)
    variants = [sorted(values) for values in groups.values() if len(values) > 1]
    if not variants:
        return []

    variant_values = {value for group in variants for value in group}
    mask = [not is_null(value) and str(value) in variant_values for value in series.tolist()]
    return [
        Issue(
            id=_id(),
            type="category_inconsistency",
            severity=Severity.MEDIUM,
            table=table,
            column=column,
            title="Equivalent-looking category variants detected",
            description=(
                "Some values differ only by capitalization or surrounding whitespace and may represent "
                "the same category."
            ),
            confidence=0.98,
            affected_count=sum(mask),
            row_indices_sample=_sample_rows(mask),
            evidence={"variant_groups": variants[:20]},
            suggested_action="Review and merge equivalent category labels.",
            auto_fix_safe=False,
        )
    ]


def _mixed_numeric(table: str, column: str, series: pd.Series) -> list[Issue]:
    values = [
        value
        for value in series.tolist()
        if not is_missing_like(value) and not is_contextual_missing_label(value)
    ]
    if len(values) < 3:
        return []

    numeric_count = sum(parse_number(value) is not None for value in values)
    ratio = numeric_count / len(values)
    if ratio < 0.60 or ratio >= 1.0:
        return []

    bad_values = [str(value) for value in values if parse_number(value) is None]
    unique_bad = sorted(set(bad_values))[:20]
    full_mask = [
        not is_missing_like(value)
        and not is_contextual_missing_label(value)
        and parse_number(value) is None
        for value in series.tolist()
    ]
    return [
        Issue(
            id=_id(),
            type="mixed_numeric_values",
            severity=Severity.MEDIUM,
            table=table,
            column=column,
            title="Mostly numeric column contains non-numeric values",
            description=f"{ratio:.1%} of non-null values can be parsed as numbers.",
            confidence=min(0.99, ratio),
            affected_count=len(values) - numeric_count,
            row_indices_sample=_sample_rows(full_mask),
            evidence={"numeric_ratio": ratio, "non_numeric_examples": unique_bad},
            suggested_action="Review non-numeric values before coercing the column to numeric.",
            auto_fix_safe=False,
        )
    ]


def _date_consistency(table: str, column: str, series: pd.Series) -> list[Issue]:
    values = [
        value
        for value in series.tolist()
        if not is_missing_like(value) and not is_contextual_missing_label(value)
    ]
    if len(values) < 3:
        return []
    parsed = [parse_date(value) for value in values]
    success = sum(value is not None for value in parsed)
    ratio = success / len(values)
    likely_date = DATE_NAME_PATTERN.search(column) is not None or ratio >= 0.60
    if not likely_date or success == len(values) or success == 0:
        return []

    bad_values = sorted(
        {
            str(value)
            for value, parsed_value in zip(values, parsed)
            if parsed_value is None
        }
    )
    full_mask = [
        not is_missing_like(value)
        and not is_contextual_missing_label(value)
        and parse_date(value) is None
        for value in series.tolist()
    ]
    return [
        Issue(
            id=_id(),
            type="malformed_or_mixed_dates",
            severity=Severity.MEDIUM,
            table=table,
            column=column,
            title="Date-like column contains unparseable values",
            description=f"{ratio:.1%} of non-null values could be interpreted as dates.",
            confidence=max(0.70, ratio),
            affected_count=len(values) - success,
            row_indices_sample=_sample_rows(full_mask),
            evidence={"date_parse_ratio": ratio, "unparseable_examples": bad_values[:20]},
            suggested_action="Review malformed values before standardizing the date format.",
            auto_fix_safe=False,
        )
    ]


def _identifier_hint(table: str, column: str, series: pd.Series) -> list[Issue]:
    if not looks_like_identifier_name(column):
        return []
    values = [value for value in series.tolist() if not is_null(value)]
    if not values:
        return []
    unique_ratio = len({repr(value) for value in values}) / len(values)
    return [
        Issue(
            id=_id(),
            type="potential_identifier",
            severity=Severity.INFO,
            table=table,
            column=column,
            title="Potential identifier column",
            description=(
                "The column name suggests an identifier. Identifier fields should generally not be "
                "imputed or treated as ordinary numeric predictors."
            ),
            confidence=0.90,
            affected_count=len(values),
            evidence={"unique_ratio": unique_ratio},
            suggested_action="Mark this column as an identifier during later cleaning/analysis steps.",
        )
    ]


def _iqr_outliers(table: str, column: str, series: pd.Series) -> list[Issue]:
    inferred, _, ratios = infer_semantic_type(column, series)
    if inferred == "identifier" or ratios["numeric_ratio"] < 0.90:
        return []
    numbers_with_rows = [
        (idx, parse_number(value))
        for idx, value in enumerate(series.tolist())
        if not is_null(value)
    ]
    numbers_with_rows = [
        (idx, value) for idx, value in numbers_with_rows if value is not None
    ]
    if len(numbers_with_rows) < 8:
        return []

    numbers = sorted(value for _, value in numbers_with_rows)
    try:
        quartiles = statistics.quantiles(numbers, n=4, method="inclusive")
    except statistics.StatisticsError:
        return []
    q1, q3 = quartiles[0], quartiles[2]
    iqr = q3 - q1
    if iqr <= 0:
        return []
    low, high = q1 - 3 * iqr, q3 + 3 * iqr
    outliers = [
        (idx, value)
        for idx, value in numbers_with_rows
        if value < low or value > high
    ]
    if not outliers:
        return []

    return [
        Issue(
            id=_id(),
            type="statistical_outlier",
            severity=Severity.LOW,
            table=table,
            column=column,
            title="Extreme numeric values detected",
            description=(
                "Values fall beyond 3×IQR from the first or third quartile. This is a review flag, "
                "not evidence that the values are erroneous."
            ),
            confidence=0.80,
            affected_count=len(outliers),
            row_indices_sample=[idx for idx, _ in outliers[:10]],
            evidence={
                "lower_fence": low,
                "upper_fence": high,
                "examples": [value for _, value in outliers[:10]],
            },
            suggested_action="Review extreme values; do not remove them automatically.",
            auto_fix_safe=False,
        )
    ]


def _unnamed_columns(table: str, frame: pd.DataFrame) -> list[Issue]:
    issues: list[Issue] = []
    for raw_column in frame.columns:
        column = str(raw_column)
        if re.match(r"^(column_\d+|unnamed:?\s*\d*)$", column.strip(), re.IGNORECASE):
            issues.append(
                Issue(
                    id=_id(),
                    type="unnamed_column",
                    severity=Severity.LOW,
                    table=table,
                    column=column,
                    title="Column has no meaningful header",
                    description="The column header appears blank or auto-generated.",
                    confidence=0.95,
                    affected_count=len(frame),
                    suggested_action="Review whether the column should be renamed or removed.",
                )
            )
    return issues


def _exact_duplicates(table: str, frame: pd.DataFrame) -> list[Issue]:
    if len(frame) < 2 or len(frame.columns) == 0:
        return []
    duplicated = frame.duplicated(keep=False).tolist()
    affected = sum(bool(value) for value in duplicated)
    if not affected:
        return []
    return [
        Issue(
            id=_id(),
            type="exact_duplicate_rows",
            severity=Severity.MEDIUM,
            table=table,
            title="Exact duplicate rows detected",
            description=(
                "Multiple rows are exactly identical across all columns. They may be true duplicate "
                "records or legitimate repeated observations."
            ),
            confidence=1.0,
            affected_count=affected,
            row_indices_sample=_sample_rows([bool(value) for value in duplicated]),
            evidence={},
            suggested_action="Review duplicated records before removing anything.",
            auto_fix_safe=False,
        )
    ]


def detect_basic_issues(table: str, frame: pd.DataFrame) -> list[Issue]:
    issues: list[Issue] = []
    issues.extend(_unnamed_columns(table, frame))
    issues.extend(_exact_duplicates(table, frame))

    for raw_column in frame.columns:
        column = str(raw_column)
        series = frame[raw_column]
        issues.extend(_hidden_missing(table, column, series))
        issues.extend(_constant_column(table, column, series))
        issues.extend(_category_variants(table, column, series))
        issues.extend(_surrounding_whitespace(table, column, series))
        issues.extend(_mixed_numeric(table, column, series))
        issues.extend(_date_consistency(table, column, series))
        issues.extend(_identifier_hint(table, column, series))
        issues.extend(_iqr_outliers(table, column, series))

    return issues
