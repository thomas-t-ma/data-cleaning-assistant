from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any

import pandas as pd
from dateutil import parser as date_parser


STRUCTURAL_MISSING_TOKENS = {
    "",
    "na",
    "n/a",
    "n.a.",
    "nan",
    "none",
    "null",
    "nil",
    "missing",
    ".",
    "-",
    "--",
}

# These values often describe a scientifically meaningful state rather than an
# absent observation. Detect them for QA, but do not treat them as null by
# default without an explicit study schema or user rule.
CONTEXTUAL_MISSING_TOKENS = {
    "unknown",
    "unk",
    "not available",
    "not applicable",
}

# Backward-compatible detector vocabulary: hidden-missing QA should surface
# both structural null sentinels and contextual missingness labels.
MISSING_TOKENS = STRUCTURAL_MISSING_TOKENS | CONTEXTUAL_MISSING_TOKENS

DATE_NAME_PATTERN = re.compile(
    r"(^|_)(date|dob|birth|time|timestamp|dt)(_|$)|date|dob|birth|timestamp",
    re.IGNORECASE,
)
ID_NAME_PATTERN = re.compile(
    r"(?:^|_)(?:id|mrn|subject|patient|participant|accession|specimen|sample)(?:_|$)"
    r"|(?:patient|subject|participant|accession|specimen|sample).*id$"
    r"|(?:^|_)(?:mrn|identifier)$",
    re.IGNORECASE,
)
DATE_SHAPE_PATTERN = re.compile(
    r"^\s*(?:\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})"
)


def is_null(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
        return bool(result) if not hasattr(result, "__len__") else False
    except (TypeError, ValueError):
        return False



def is_empty_cell(value: Any) -> bool:
    """A native null or exact empty text, not a nonempty sentinel.

    Pandas reads ordinary blank Excel cells as '' when na_filter=False.
    Treat them as already empty for profiling, without mutating the source.
    Whitespace-only strings and tokens such as NA remain distinct.
    """
    return is_null(value) or (isinstance(value, str) and value == "")


def normalize_text(value: Any) -> str:
    return str(value).strip().casefold()


def is_missing_like(value: Any) -> bool:
    """Return True only for nulls and structural missing-value sentinels."""
    return is_null(value) or normalize_text(value) in STRUCTURAL_MISSING_TOKENS


def is_contextual_missing_label(value: Any) -> bool:
    return not is_null(value) and normalize_text(value) in CONTEXTUAL_MISSING_TOKENS


def non_null_values(series: pd.Series, limit: int | None = None) -> list[Any]:
    values = [value for value in series.tolist() if not is_null(value)]
    return values if limit is None else values[:limit]


def meaningful_values(series: pd.Series, limit: int | None = None) -> list[Any]:
    values = [value for value in series.tolist() if not is_missing_like(value)]
    return values if limit is None else values[:limit]


def parse_number(value: Any) -> float | None:
    if is_null(value) or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = str(value).strip().replace(",", "")
    if not text or normalize_text(text) in STRUCTURAL_MISSING_TOKENS:
        return None
    try:
        number = float(text)
        return number if math.isfinite(number) else None
    except ValueError:
        return None


def parse_date(value: Any) -> datetime | None:
    if is_null(value):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    text = str(value).strip()
    if not text or not DATE_SHAPE_PATTERN.search(text):
        return None

    numeric = re.fullmatch(
        r"(\d{1,4})[-/](\d{1,2})[-/](\d{1,4})",
        text,
    )
    if numeric:
        first_text, second_text, third_text = numeric.groups()
        first = int(first_text)
        second = int(second_text)

        # Year-first numeric dates are unambiguous.
        if len(first_text) == 4:
            try:
                return date_parser.parse(text, fuzzy=False, yearfirst=True)
            except (ValueError, OverflowError, TypeError):
                return None

        # For day/month/year or month/day/year forms, refuse to guess when
        # both leading fields could be months. This prevents silent swaps such
        # as 06/08/2025 -> 2025-06-08 on data entered as DD/MM/YYYY.
        if len(third_text) in {2, 4}:
            if first <= 12 and second <= 12:
                return None
            try:
                return date_parser.parse(
                    text,
                    fuzzy=False,
                    dayfirst=first > 12,
                )
            except (ValueError, OverflowError, TypeError):
                return None

    try:
        return date_parser.parse(text, fuzzy=False)
    except (ValueError, OverflowError, TypeError):
        return None


def normalize_column_name(name: str) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name))
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").casefold()
    return text


def looks_like_identifier_name(name: str) -> bool:
    return ID_NAME_PATTERN.search(normalize_column_name(name)) is not None


def infer_semantic_type(name: str, series: pd.Series) -> tuple[str, float, dict[str, float]]:
    values = [
        value
        for value in meaningful_values(series, limit=500)
        if not is_contextual_missing_label(value)
    ]
    if not values:
        return "empty", 1.0, {"numeric_ratio": 0.0, "date_ratio": 0.0}

    numeric_count = sum(parse_number(value) is not None for value in values)
    date_count = sum(parse_date(value) is not None for value in values)
    numeric_ratio = numeric_count / len(values)
    date_ratio = date_count / len(values)

    if looks_like_identifier_name(name):
        return "identifier", 0.90, {"numeric_ratio": numeric_ratio, "date_ratio": date_ratio}
    if DATE_NAME_PATTERN.search(name) and date_ratio >= 0.50:
        return "date", max(0.75, date_ratio), {
            "numeric_ratio": numeric_ratio,
            "date_ratio": date_ratio,
        }
    if date_ratio >= 0.85:
        return "date", date_ratio, {"numeric_ratio": numeric_ratio, "date_ratio": date_ratio}
    if numeric_ratio >= 0.90:
        return "numeric", numeric_ratio, {
            "numeric_ratio": numeric_ratio,
            "date_ratio": date_ratio,
        }

    unique = int(series.nunique(dropna=True))
    row_count = max(len(series), 1)
    if unique <= 50 or unique / row_count <= 0.10:
        return "categorical", 0.75, {"numeric_ratio": numeric_ratio, "date_ratio": date_ratio}
    return "text", 0.70, {"numeric_ratio": numeric_ratio, "date_ratio": date_ratio}
