from __future__ import annotations

import itertools
import re
from collections import Counter

import pandas as pd

from .inference import (
    is_missing_like,
    looks_like_identifier_name,
    normalize_column_name,
    normalize_text,
)
from .models import (
    CandidateKey,
    CrossTableRelationship,
    GrainInference,
    Issue,
    Severity,
)


SUBJECT_PATTERNS = ("patient", "subject", "participant", "person", "mrn")
GRAIN_PATTERNS: dict[str, tuple[str, ...]] = {
    "specimen": ("specimen", "sample", "aliquot", "accession"),
    "visit": ("visit", "encounter", "appointment"),
    "procedure": ("procedure", "transfusion", "apheresis", "surgery"),
    "medication": ("medication", "drug", "dose", "rx"),
    "event": ("event", "adverse", "outcome"),
}


def _meaningful_series(series: pd.Series) -> pd.Series:
    return series[~series.map(is_missing_like)]


def _completeness(frame: pd.DataFrame, columns: list[str]) -> float:
    if len(frame) == 0:
        return 1.0
    complete = frame[columns].apply(lambda row: not any(is_missing_like(v) for v in row), axis=1)
    return float(complete.mean())


def _uniqueness(frame: pd.DataFrame, columns: list[str]) -> float:
    if len(frame) == 0:
        return 1.0
    complete = frame[columns].apply(lambda row: not any(is_missing_like(v) for v in row), axis=1)
    subset = frame.loc[complete, columns]
    if len(subset) == 0:
        return 0.0
    return float(len(subset.drop_duplicates()) / len(subset))


def infer_candidate_keys(frame: pd.DataFrame, max_keys: int = 5) -> list[CandidateKey]:
    if len(frame) == 0 or len(frame.columns) == 0:
        return []

    names = [str(c) for c in frame.columns]
    scored: list[CandidateKey] = []

    for column in names:
        completeness = _completeness(frame, [column])
        uniqueness = _uniqueness(frame, [column])
        semantic_bonus = 0.08 if looks_like_identifier_name(column) else 0.0
        if completeness >= 0.90 and uniqueness >= 0.97:
            confidence = min(0.99, 0.55 * uniqueness + 0.37 * completeness + semantic_bonus)
            scored.append(
                CandidateKey(
                    columns=[column],
                    uniqueness=uniqueness,
                    completeness=completeness,
                    confidence=confidence,
                )
            )

    if not any(key.confidence >= 0.95 for key in scored):
        prioritized = sorted(
            names,
            key=lambda c: (
                0 if looks_like_identifier_name(c) else 1,
                0 if re.search(r"date|time|visit|specimen|sample|accession", c, re.IGNORECASE) else 1,
                c,
            ),
        )[:12]
        for left, right in itertools.combinations(prioritized, 2):
            completeness = _completeness(frame, [left, right])
            uniqueness = _uniqueness(frame, [left, right])
            if completeness >= 0.85 and uniqueness >= 0.98:
                bonus = (
                    0.05
                    if looks_like_identifier_name(left) or looks_like_identifier_name(right)
                    else 0.0
                )
                confidence = min(0.98, 0.54 * uniqueness + 0.38 * completeness + bonus)
                scored.append(
                    CandidateKey(
                        columns=[left, right],
                        uniqueness=uniqueness,
                        completeness=completeness,
                        confidence=confidence,
                    )
                )

    scored.sort(key=lambda key: (-key.confidence, len(key.columns), key.columns))
    chosen: list[CandidateKey] = []
    for key in scored:
        key_set = set(key.columns)
        if any(
            set(existing.columns).issubset(key_set) and existing.confidence >= key.confidence
            for existing in chosen
        ):
            continue
        chosen.append(key)
        if len(chosen) >= max_keys:
            break
    return chosen


def _find_subject_column(frame: pd.DataFrame) -> str | None:
    names = [str(c) for c in frame.columns]
    candidates: list[tuple[int, float, str]] = []
    for name in names:
        normalized = normalize_column_name(name)
        if not any(pattern in normalized for pattern in SUBJECT_PATTERNS):
            continue
        series = _meaningful_series(frame[name])
        if len(series) == 0:
            continue
        uniqueness = float(series.astype(str).nunique() / len(series))
        priority = 0 if looks_like_identifier_name(name) else 1
        candidates.append((priority, -uniqueness, name))
    return min(candidates)[2] if candidates else None


def infer_grain(table_name: str, frame: pd.DataFrame) -> GrainInference:
    normalized_table = normalize_column_name(table_name)
    normalized_columns = [normalize_column_name(str(c)) for c in frame.columns]
    evidence: list[str] = []

    scores: Counter[str] = Counter()
    for label, patterns in GRAIN_PATTERNS.items():
        for pattern in patterns:
            if pattern in normalized_table:
                scores[label] += 3
                evidence.append(f"table name contains '{pattern}'")
            hits = sum(pattern in column for column in normalized_columns)
            if hits:
                scores[label] += min(3, hits)
                evidence.append(f"{hits} column(s) reference '{pattern}'")

    subject_column = _find_subject_column(frame)
    if subject_column:
        series = _meaningful_series(frame[subject_column])
        if len(series):
            unique_ratio = float(series.astype(str).nunique() / len(series))
            if unique_ratio >= 0.97:
                scores["subject"] += 4
                evidence.append(f"{subject_column} is nearly unique ({unique_ratio:.1%})")
            else:
                scores["observation"] += 2
                evidence.append(f"{subject_column} repeats across rows ({unique_ratio:.1%} unique)")

    if not scores:
        return GrainInference(
            label="unknown",
            confidence=0.35,
            evidence=["No strong structural cues detected"],
        )

    label, score = scores.most_common(1)[0]
    second = scores.most_common(2)[1][1] if len(scores) > 1 else 0
    confidence = min(0.96, 0.55 + 0.06 * score + 0.03 * max(score - second, 0))

    if label == "observation":
        specific = [
            (name, value)
            for name, value in scores.items()
            if name not in {"observation", "subject"}
        ]
        if specific:
            best_specific = max(specific, key=lambda item: item[1])
            if best_specific[1] >= 2:
                label = best_specific[0]
                confidence = min(0.94, confidence + 0.05)

    return GrainInference(label=label, confidence=confidence, evidence=evidence[:8])


def repeated_entity_issues(table: str, frame: pd.DataFrame) -> list[Issue]:
    subject_column = _find_subject_column(frame)
    if not subject_column or len(frame) < 2:
        return []
    series = frame[subject_column]
    meaningful = ~series.map(is_missing_like)
    duplicates = series[meaningful].astype(str).duplicated(keep=False)
    repeated_indices = list(series[meaningful][duplicates].index)
    if not repeated_indices:
        return []
    value_counts = series[meaningful].astype(str).value_counts()
    groups = value_counts[value_counts > 1]
    return [
        Issue(
            id=f"structure_repeated_{table}_{subject_column}".replace(" ", "_")[:120],
            type="repeated_entity_rows",
            severity=Severity.INFO,
            table=table,
            column=subject_column,
            title="Identifier repeats across multiple rows",
            description=(
                "The likely subject/entity identifier appears on multiple rows. This often indicates "
                "longitudinal or one-to-many data rather than duplicate records."
            ),
            confidence=0.92,
            affected_count=len(repeated_indices),
            row_indices_sample=[int(i) for i in repeated_indices[:10]],
            evidence={
                "repeated_entity_count": len(groups),
                "largest_group_size": int(groups.max()),
                "examples": {str(k): int(v) for k, v in groups.head(10).items()},
            },
            suggested_action=(
                "Treat repeated identifiers as entity structure unless other evidence "
                "shows true duplication."
            ),
            auto_fix_safe=False,
        )
    ]


def _normalized_duplicate_cell(value):
    if is_missing_like(value):
        return ("missing",)
    if isinstance(value, str):
        return ("text", normalize_text(value))
    return ("value", type(value).__name__, repr(value))


def normalization_equivalent_duplicate_issues(
    table: str,
    frame: pd.DataFrame,
) -> list[Issue]:
    """Flag non-identical rows that collapse under safe text normalization.

    This is QA-only. It never authorizes row deletion because repeated
    observations may be legitimate even when every field is normalization-
    equivalent.
    """
    if len(frame) < 2 or len(frame.columns) == 0:
        return []

    groups: dict[tuple, list[int]] = {}
    for row_position, row in enumerate(frame.itertuples(index=False, name=None)):
        signature = tuple(_normalized_duplicate_cell(value) for value in row)
        groups.setdefault(signature, []).append(row_position)

    candidate_groups: list[list[int]] = []
    for positions in groups.values():
        if len(positions) < 2:
            continue
        subset = frame.iloc[positions]
        if len(subset.drop_duplicates()) <= 1:
            continue
        candidate_groups.append(positions)

    if not candidate_groups:
        return []

    affected = sorted({position for group in candidate_groups for position in group})
    return [
        Issue(
            id=f"structure_normalized_duplicates_{table}".replace(" ", "_")[:120],
            type="normalization_equivalent_rows",
            severity=Severity.MEDIUM,
            table=table,
            title="Rows become identical after safe text normalization",
            description=(
                "Some rows are not exact duplicates but become identical after "
                "collapsing structural missing sentinels and normalizing text case/"
                "surrounding whitespace. They may be duplicate records or legitimate "
                "repeated observations. No rows are removed automatically."
            ),
            confidence=0.95,
            affected_count=len(affected),
            row_indices_sample=affected[:10],
            evidence={
                "group_count": len(candidate_groups),
                "largest_group_size": max(len(group) for group in candidate_groups),
                "row_groups_sample": [group[:10] for group in candidate_groups[:10]],
            },
            suggested_action=(
                "Review the grouped rows together. Do not delete or merge records "
                "without study-specific evidence."
            ),
            auto_fix_safe=False,
        )
    ]


def infer_relationships(tables: dict[str, pd.DataFrame]) -> list[CrossTableRelationship]:
    relationships: list[CrossTableRelationship] = []
    table_names = list(tables)
    for left_name, right_name in itertools.combinations(table_names, 2):
        left = tables[left_name]
        right = tables[right_name]
        left_by_norm = {normalize_column_name(str(c)): str(c) for c in left.columns}
        right_by_norm = {normalize_column_name(str(c)): str(c) for c in right.columns}
        shared_columns = set(left_by_norm) & set(right_by_norm)

        for normalized in shared_columns:
            left_col = left_by_norm[normalized]
            right_col = right_by_norm[normalized]
            if not (
                looks_like_identifier_name(left_col)
                or looks_like_identifier_name(right_col)
                or any(
                    token in normalized
                    for token in (
                        "accession",
                        "specimen",
                        "sample",
                        "patient",
                        "subject",
                        "participant",
                        "mrn",
                    )
                )
            ):
                continue

            left_values = [
                normalize_text(v) for v in left[left_col].tolist() if not is_missing_like(v)
            ]
            right_values = [
                normalize_text(v) for v in right[right_col].tolist() if not is_missing_like(v)
            ]
            if not left_values or not right_values:
                continue
            left_set, right_set = set(left_values), set(right_values)
            shared = left_set & right_set
            if not shared:
                continue
            left_coverage = len(shared) / len(left_set)
            right_coverage = len(shared) / len(right_set)
            if max(left_coverage, right_coverage) < 0.25:
                continue

            left_unique = len(left_set) == len(left_values)
            right_unique = len(right_set) == len(right_values)
            if left_unique and right_unique:
                relation = "one-to-one"
            elif left_unique and not right_unique:
                relation = "one-to-many"
            elif not left_unique and right_unique:
                relation = "many-to-one"
            else:
                relation = "many-to-many"

            confidence = min(0.99, 0.55 + 0.22 * left_coverage + 0.22 * right_coverage)
            relationships.append(
                CrossTableRelationship(
                    left_table=left_name,
                    left_column=left_col,
                    right_table=right_name,
                    right_column=right_col,
                    relationship=relation,
                    shared_value_count=len(shared),
                    left_coverage=left_coverage,
                    right_coverage=right_coverage,
                    confidence=confidence,
                )
            )

    relationships.sort(
        key=lambda r: (-r.confidence, r.left_table, r.right_table, r.left_column)
    )
    return relationships
