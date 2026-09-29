from __future__ import annotations

import re
import uuid
from difflib import SequenceMatcher

import pandas as pd

from ..inference import is_missing_like, normalize_text
from ..models import Issue, Severity


def _id() -> str:
    return f"issue_{uuid.uuid4().hex[:12]}"


def _simplify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_text(value))


def detect_fuzzy_categories(table: str, frame: pd.DataFrame) -> list[Issue]:
    issues: list[Issue] = []
    for raw_column in frame.columns:
        column = str(raw_column)
        values = [str(v).strip() for v in frame[raw_column].tolist() if not is_missing_like(v)]
        unique = sorted(set(values))
        if len(unique) < 2 or len(unique) > 80:
            continue
        if len(unique) > 10 and len(unique) / len(values) > 0.50:
            continue

        pairs: list[dict[str, object]] = []
        for i, left in enumerate(unique):
            left_simple = _simplify(left)
            if len(left_simple) < 3:
                continue
            for right in unique[i + 1 :]:
                right_simple = _simplify(right)
                if len(right_simple) < 3 or left_simple == right_simple:
                    continue
                if left_simple.isdigit() or right_simple.isdigit():
                    continue
                ratio = SequenceMatcher(None, left_simple, right_simple).ratio()
                if ratio < 0.92:
                    continue
                if min(len(left_simple), len(right_simple)) < 5 and ratio < 0.96:
                    continue
                pairs.append({"left": left, "right": right, "similarity": round(ratio, 4)})

        if not pairs:
            continue
        affected_values = {str(item["left"]) for item in pairs} | {str(item["right"]) for item in pairs}
        affected = sum(str(v).strip() in affected_values for v in frame[raw_column].tolist())
        issues.append(
            Issue(
                id=_id(),
                type="fuzzy_category_match",
                severity=Severity.LOW,
                table=table,
                column=column,
                title="Similar category labels may represent the same concept",
                description=(
                    "Some category labels are highly similar after punctuation/spacing normalization. "
                    "They are suggestions only because similar labels can still be scientifically distinct."
                ),
                confidence=max(float(item["similarity"]) for item in pairs),
                affected_count=affected,
                evidence={"candidate_pairs": pairs[:20]},
                suggested_action="Review each proposed category pair before merging labels.",
                auto_fix_safe=False,
            )
        )
    return issues
