from __future__ import annotations

import statistics
from typing import Any

import pandas as pd

from .inference import infer_semantic_type, is_null, non_null_values, parse_number
from .models import ColumnProfile, TableProfile


def _safe_sample(series: pd.Series, n: int = 5) -> list[Any]:
    output: list[Any] = []
    seen: set[str] = set()
    for value in non_null_values(series):
        key = repr(value)
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
        if len(output) >= n:
            break
    return output


def _numeric_summary(series: pd.Series) -> dict[str, float | int | None] | None:
    numbers = [parse_number(value) for value in non_null_values(series)]
    numbers = [value for value in numbers if value is not None]
    if not numbers:
        return None
    return {
        "count": len(numbers),
        "min": min(numbers),
        "max": max(numbers),
        "mean": statistics.fmean(numbers),
        "median": statistics.median(numbers),
    }


def profile_table(name: str, frame: pd.DataFrame) -> TableProfile:
    row_count = len(frame)
    columns: list[ColumnProfile] = []

    for column_name in frame.columns:
        column_name = str(column_name)
        series = frame[column_name]
        null_count = sum(is_null(value) for value in series.tolist())
        unique_count = int(series.nunique(dropna=False))
        inferred_type, confidence, ratios = infer_semantic_type(column_name, series)

        numeric_summary = None
        if inferred_type == "numeric" or ratios["numeric_ratio"] >= 0.50:
            numeric_summary = _numeric_summary(series)

        columns.append(
            ColumnProfile(
                name=column_name,
                dtype=str(series.dtype),
                row_count=row_count,
                null_count=null_count,
                null_percent=(null_count / row_count * 100.0) if row_count else 0.0,
                unique_count=unique_count,
                unique_percent=(unique_count / row_count * 100.0) if row_count else 0.0,
                inferred_type=inferred_type,
                type_confidence=confidence,
                sample_values=_safe_sample(series),
                numeric_summary=numeric_summary,
            )
        )

    return TableProfile(
        name=name,
        row_count=row_count,
        column_count=len(frame.columns),
        columns=columns,
    )
