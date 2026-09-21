"""Shared conservative policy for structural-only missingness proposals.

An entirely empty/structural-sentinel column remains a QA coverage finding.
Routine proposals to rewrite its many sentinel cells are omitted unless the
study schema explicitly authorizes a transformation for that column.
"""
from __future__ import annotations

import pandas as pd

from .inference import is_missing_like
from .models import Issue


def structural_only_column(issue: Issue, frame: pd.DataFrame) -> bool:
    """A detected structural sentinel exists but no meaningful value exists.

    True nulls count as absent. Contextual labels such as 'Unknown' count as
    meaningful, so mixed contextual/structural columns stay reviewable.
    """
    if issue.type != "hidden_missing_values":
        return False
    if issue.column is None or issue.column not in frame.columns:
        return False
    structural = issue.evidence.get("structural_tokens", {})
    if not structural:
        return False
    series = frame[issue.column]
    if len(series) == 0:
        return False
    return not any(not is_missing_like(value) for value in series.tolist())
