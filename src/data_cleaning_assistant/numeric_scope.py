"""Conservative scope guard for generic measurements across mixed assays/units.

Never pool a generic value/result column when neighboring metadata show that
multiple measurements or units are mixed. This is read-only QA; no imputation,
conversion, or automatic assignment to assay groups is attempted.
"""
from __future__ import annotations

import re

import pandas as pd

from .inference import (
    is_missing_like,
    looks_like_identifier_name,
    normalize_column_name,
    parse_number,
)
from .models import Issue, Severity

_GENERIC_VALUE = re.compile(
    r"^(?:(?:test|lab|laboratory|measurement|measured|numeric|result)_)?"
    r"(?:value|result|reading)$"
)
_ASSAY_METADATA = re.compile(
    r"(?:^|_)(?:assay|analyte|units?|panel)(?:_|$)"
    r"|(?:^|_)(?:lab|laboratory)_test(?:_|$)"
    r"|(?:^|_)(?:test|measurement)_(?:name|type|code|description)(?:_|$)"
)


def heterogeneous_measurement_metadata(
    frame: pd.DataFrame, numeric_column: str
) -> dict[str, int]:
    """Name-based *safety signal*, not authority to identify assay groups."""
    if not _GENERIC_VALUE.fullmatch(normalize_column_name(numeric_column)):
        return {}
    metadata: dict[str, int] = {}
    for raw_column in frame.columns:
        column = str(raw_column)
        if column == numeric_column or looks_like_identifier_name(column):
            continue
        if not _ASSAY_METADATA.search(normalize_column_name(column)):
            continue
        meaningful = [
            str(value).strip()
            for value in frame[column]
            if not is_missing_like(value)
        ]
        unique = len(set(meaningful))
        if unique > 1:
            metadata[column] = unique
    return metadata


def unscoped_measurement_issue(
    table: str,
    column: str,
    frame: pd.DataFrame,
    original_outlier: Issue,
) -> Issue:
    """Replace misleading pooled outlier counts with one scope warning."""
    metadata = heterogeneous_measurement_metadata(frame, column)
    return Issue(
        id=f"qa_unscoped_{table}_{column}"[:120],
        type="unscoped_numeric_measurement",
        severity=Severity.MEDIUM,
        table=table,
        column=column,
        title="Pooled outlier check skipped across different assays or units",
        description=(
            "A generic numeric result column has multiple observed assay/unit "
            "metadata categories. A single pooled IQR fence is not scientifically "
            "interpretable; declare numeric_groups only after validating those "
            "metadata fields. No data were changed or classified as outliers."
        ),
        confidence=1.0,
        affected_count=sum(parse_number(value) is not None for value in frame[column]),
        evidence={
            "grouping_metadata_unique_counts": metadata,
            "suppressed_pooled_outlier_count": original_outlier.affected_count,
            "grouped_outlier_check_performed": False,
        },
        suggested_action=(
            "Review assay/unit metadata and declare numeric_groups in the study "
            "schema; do not merge measurements or perform implicit unit conversions."
        ),
        auto_fix_safe=False,
    )
