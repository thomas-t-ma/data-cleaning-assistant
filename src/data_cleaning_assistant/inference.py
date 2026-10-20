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
    r"(^|_)(date|dob|birth|time|timestamp|dt|created|updated|submitted|"
    r"closed|opened|completed)(_|$)|date|dob|birth|timestamp",
    re.IGNORECASE,
)
ID_NAME_PATTERN = re.compile(
    r"(?:^|_)(?:id|mrn|subject|patient|participant|accession|specimen|sample|"
    r"sku|code|key|zip|zipcode|postal|postcode)(?:_|$)"
    r"|(?:patient|subject|participant|accession|specimen|sample).*id$"
    r"|(?:^|_)(?:mrn|identifier|postalcode|zipcode)$",
    re.IGNORECASE,
)
DATE_SHAPE_PATTERN = re.compile(
    r"^\s*(?:\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})"
)
NUMERIC_DATE_PREFIX_PATTERN = re.compile(
    r"^\s*(\d{1,4})[-/](\d{1,2})[-/](\d{1,4})(?=$|[T\s])"
)
CURRENCY_SYMBOLS = "$£€¥₹"
PHONE_NAME_PATTERN = re.compile(
    r"(?:^|_)(?:phone|telephone|tel|mobile|cell)(?:_|$)|phone|telephone|mobile",
    re.IGNORECASE,
)
EMAIL_NAME_PATTERN = re.compile(
    r"(?:^|_)(?:email|e_mail)(?:_|$)|email",
    re.IGNORECASE,
)
BOOLEAN_TRUE_TOKENS = {"yes", "y", "true", "t", "1", "on", "enabled"}
BOOLEAN_FALSE_TOKENS = {"no", "n", "false", "f", "0", "off", "disabled"}
DECLARED_NUMERIC_UNITS = {
    "kg": ("kg", "kilogram", "kilograms"),
    "mg": ("mg", "milligram", "milligrams"),
    "lb": ("lb", "lbs", "pound", "pounds"),
    "cm": ("cm", "centimeter", "centimeters"),
    "mm": ("mm", "millimeter", "millimeters"),
    "km": ("km", "kilometer", "kilometers"),
    "ml": ("ml", "milliliter", "milliliters"),
    "percent": ("%", "percent", "pct"),
}


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


def looks_like_free_text_response(name: str, series: pd.Series) -> bool:
    """Conservatively identify survey/comment fields that should not be category-normalized.

    Repeated prose tokens such as "No"/"no" inside an otherwise free-response
    column are not evidence of a categorical codebook. Use both response
    diversity and prose shape so ordinary low-cardinality categorical columns
    remain eligible for category normalization.
    """
    values = [
        str(value).strip()
        for value in series.tolist()
        if not is_missing_like(value) and str(value).strip()
    ]
    if len(values) < 3:
        return False

    unique_count = len(set(values))
    unique_ratio = unique_count / len(values)
    word_counts = [len(value.split()) for value in values]
    name_hint = re.search(
        r"\b(comment|comments|note|notes|description|details|explain|"
        r"specify|please\s+paste|free[- ]?text)\b",
        name,
        re.IGNORECASE,
    ) is not None

    # Explicit free-response field names are strong structural evidence even
    # in small exports where repeated answers make cardinality look categorical.
    # Require prose-shaped values as a second guard so fields such as
    # "comment_code" with short Yes/No codes are not automatically excluded.
    prose_like_ratio = sum(
        len(value) >= 12 or words >= 2
        for value, words in zip(values, word_counts, strict=False)
    ) / len(values)
    if name_hint and unique_ratio >= 0.40 and prose_like_ratio >= 0.60:
        return True

    if unique_count <= 10 or unique_ratio < 0.60:
        return False

    long_response_ratio = sum(
        len(value) >= 40 or words >= 8
        for value, words in zip(values, word_counts, strict=False)
    ) / len(values)

    # High-diversity prose should not be treated as a category codebook even
    # when a few repeated short answers happen to differ only by case.
    if unique_ratio >= 0.75 and long_response_ratio >= 0.20:
        return True
    return name_hint and unique_ratio >= 0.60


def boolean_class(value: Any) -> bool | None:
    if is_null(value):
        return None
    token = normalize_text(value)
    if token in BOOLEAN_TRUE_TOKENS:
        return True
    if token in BOOLEAN_FALSE_TOKENS:
        return False
    return None


def declared_numeric_units(name: str) -> tuple[str, ...]:
    normalized = normalize_column_name(name)
    tokens = normalized.split("_")
    if not tokens:
        return ()
    last = tokens[-1]
    if last in {"pct", "percentage"}:
        last = "percent"
    return DECLARED_NUMERIC_UNITS.get(last, ())


def has_declared_unit_format(
    value: Any,
    allowed_units: tuple[str, ...],
) -> bool:
    if is_null(value) or not isinstance(value, str) or not allowed_units:
        return False
    text = value.strip()
    for unit in sorted(allowed_units, key=len, reverse=True):
        if unit == "%":
            if re.fullmatch(r"[+-]?\d[\d,]*(?:\.\d+)?\s*%", text):
                return True
        elif re.fullmatch(
            rf"[+-]?\d[\d,]*(?:\.\d+)?\s*{re.escape(unit)}",
            text,
            flags=re.IGNORECASE,
        ):
            return True
    return False


def has_explicit_numeric_format(value: Any) -> bool:
    """Return True for unambiguous numeric presentation artifacts.

    This is intentionally narrower than "looks numeric". Plain digit strings
    are excluded so identifiers and zero-padded codes do not become cleanup
    targets merely because they can be parsed as numbers.
    """
    if is_null(value) or not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False

    if text[0] in CURRENCY_SYMBOLS:
        return True
    after_sign = text[1:].lstrip()[:1] if len(text) >= 2 else ""
    if text[0] in "+-" and after_sign and after_sign in CURRENCY_SYMBOLS:
        return True
    if (
        text.startswith("(")
        and text.endswith(")")
        and re.fullmatch(
            rf"\(\s*[{re.escape(CURRENCY_SYMBOLS)}]?\s*[\d,]+(?:\.\d+)?\s*\)",
            text,
        )
    ):
        return True
    return re.fullmatch(
        r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?",
        text,
    ) is not None


def parse_number(value: Any) -> float | None:
    if is_null(value) or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None

    text = str(value).strip()
    if not text or normalize_text(text) in STRUCTURAL_MISSING_TOKENS:
        return None

    accounting_negative = text.startswith("(") and text.endswith(")")
    if accounting_negative:
        text = text[1:-1].strip()

    # Support common currency prefixes without teaching the parser to strip
    # arbitrary letters or codes.
    sign = ""
    if text[:1] in {"+", "-"}:
        sign, text = text[0], text[1:].lstrip()
    if text[:1] in CURRENCY_SYMBOLS:
        text = text[1:].lstrip()
    elif not sign and len(text) >= 2 and text[0] in CURRENCY_SYMBOLS and text[1] in {"+", "-"}:
        sign, text = text[1], text[2:].lstrip()

    text = text.replace(",", "")
    if sign:
        text = sign + text

    try:
        number = float(text)
    except ValueError:
        return None
    if accounting_negative:
        number = -abs(number)
    return number if math.isfinite(number) else None


def infer_numeric_date_order(
    series: pd.Series,
    *,
    min_directional_evidence: int = 3,
) -> tuple[str | None, dict[str, int | float]]:
    """Infer month-first/day-first order only from values that prove direction.

    Ambiguous values such as 04/05/2025 never vote. A column is inferred only
    when enough values support one direction and zero support the opposite.
    """

    month_first = 0
    day_first = 0
    ambiguous = 0
    year_first = 0
    invalid = 0

    for value in meaningful_values(series):
        text = str(value).strip()
        match = NUMERIC_DATE_PREFIX_PATTERN.match(text)
        if match is None:
            continue
        first_text, second_text, third_text = match.groups()
        first = int(first_text)
        second = int(second_text)

        if len(first_text) == 4:
            year_first += 1
            continue
        if len(third_text) not in {2, 4}:
            invalid += 1
            continue

        if first <= 12 and second > 12:
            month_first += 1
        elif first > 12 and second <= 12:
            day_first += 1
        elif first <= 12 and second <= 12:
            ambiguous += 1
        else:
            invalid += 1

    directional = month_first + day_first
    evidence: dict[str, int | float] = {
        "month_first_evidence": month_first,
        "day_first_evidence": day_first,
        "ambiguous_values": ambiguous,
        "year_first_values": year_first,
        "invalid_values": invalid,
        "directional_evidence": directional,
    }

    if directional < min_directional_evidence:
        evidence["confidence"] = 0.0
        return None, evidence
    if month_first and day_first:
        evidence["confidence"] = max(month_first, day_first) / directional
        return None, evidence

    order = "month_first" if month_first else "day_first"
    confidence = min(0.999, 0.95 + min(directional, 1000) / 20000)
    evidence["confidence"] = confidence
    return order, evidence


def parse_date(
    value: Any,
    *,
    numeric_order: str | None = None,
) -> datetime | None:
    if is_null(value):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    text = str(value).strip()
    if not text or not DATE_SHAPE_PATTERN.search(text):
        return None

    numeric = NUMERIC_DATE_PREFIX_PATTERN.match(text)
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
                if numeric_order == "month_first":
                    day_first = False
                elif numeric_order == "day_first":
                    day_first = True
                else:
                    return None
            else:
                day_first = first > 12
            try:
                return date_parser.parse(
                    text,
                    fuzzy=False,
                    dayfirst=day_first,
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
    if PHONE_NAME_PATTERN.search(name):
        return "phone", 0.95, {
            "numeric_ratio": numeric_ratio,
            "date_ratio": date_ratio,
        }
    if EMAIL_NAME_PATTERN.search(name):
        return "email", 0.95, {
            "numeric_ratio": numeric_ratio,
            "date_ratio": date_ratio,
        }
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
