from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .actions import ActionOrigin, ActionType, CleaningAction
from .ai_models import AICandidateContext
from .inference import looks_like_identifier_name
from .models import DatasetReport, Issue


@dataclass(slots=True)
class AICandidate:
    id: str
    action: CleaningAction
    context: AICandidateContext


def _stable_id(
    action_type: ActionType,
    table: str,
    column: str | None,
    parameters: dict[str, Any],
) -> str:
    payload = json.dumps(
        {
            "type": action_type.value,
            "table": table,
            "column": column,
            "parameters": parameters,
        },
        sort_keys=True,
        default=str,
    )
    return f"candidate_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def _make_candidate(
    *,
    action_type: ActionType,
    table: str,
    column: str | None,
    parameters: dict[str, Any],
    issue: Issue,
    evidence: dict[str, Any],
    question: str,
    title: str,
) -> AICandidate:
    candidate_id = _stable_id(action_type, table, column, parameters)
    action = CleaningAction(
        id=f"ai_{candidate_id.removeprefix('candidate_')}",
        type=action_type,
        table=table,
        column=column,
        title=title,
        description=question,
        parameters=parameters,
        confidence=issue.confidence,
        issue_ids=[issue.id],
        origin=ActionOrigin.AI,
    )
    context = AICandidateContext(
        id=candidate_id,
        type=action_type,
        table=table,
        column=column,
        proposed_change=parameters,
        detector_confidence=issue.confidence,
        evidence=evidence,
        question=question,
    )
    return AICandidate(id=candidate_id, action=action, context=context)


def _counts(series: pd.Series) -> Counter[str]:
    return Counter(str(value) for value in series.tolist())


def _canonical_pair(left: str, right: str, counts: Counter[str]) -> tuple[str, str]:
    left_count = counts[left]
    right_count = counts[right]
    if left_count > right_count:
        return right, left
    if right_count > left_count:
        return left, right
    # A deterministic tie-breaker is required only to construct one bounded candidate.
    # The model still decides whether the pair is semantically equivalent at all.
    canonical = min(left, right)
    variant = right if canonical == left else left
    return variant, canonical


def _category_candidates(
    issue: Issue,
    frame: pd.DataFrame,
) -> list[AICandidate]:
    if issue.column is None or issue.column not in frame.columns:
        return []
    counts = _counts(frame[issue.column])
    output: list[AICandidate] = []

    if issue.type == "category_inconsistency":
        for raw_group in issue.evidence.get("variant_groups", []):
            variants = [str(value) for value in raw_group]
            if len(variants) < 2:
                continue
            canonical = max(
                variants,
                key=lambda value: (
                    counts[value],
                    int(value == value.strip()),
                    int(value[:1].isupper()),
                    -len(value),
                    value,
                ),
            ).strip()
            mapping = {
                variant: canonical
                for variant in variants
                if variant != canonical
            }
            if not mapping:
                continue
            output.append(
                _make_candidate(
                    action_type=ActionType.NORMALIZE_CATEGORIES,
                    table=issue.table,
                    column=issue.column,
                    parameters={"mapping": mapping},
                    issue=issue,
                    evidence={
                        "variant_group": variants,
                        "counts": {value: counts[value] for value in variants},
                        "basis": "case_or_whitespace_equivalence",
                    },
                    question=(
                        "Are these case/whitespace variants the same category and safe to map "
                        f"directly to {canonical!r}?"
                    ),
                    title="AI-reviewed category normalization",
                )
            )

    elif issue.type == "fuzzy_category_match":
        for item in issue.evidence.get("candidate_pairs", []):
            if not isinstance(item, dict):
                continue
            left = str(item.get("left", ""))
            right = str(item.get("right", ""))
            if not left or not right:
                continue
            variant, canonical = _canonical_pair(left, right, counts)
            output.append(
                _make_candidate(
                    action_type=ActionType.NORMALIZE_CATEGORIES,
                    table=issue.table,
                    column=issue.column,
                    parameters={"mapping": {variant: canonical}},
                    issue=issue,
                    evidence={
                        "left": left,
                        "right": right,
                        "similarity": item.get("similarity"),
                        "counts": {left: counts[left], right: counts[right]},
                        "basis": "fuzzy_similarity_only",
                    },
                    question=(
                        f"Do {left!r} and {right!r} clearly mean the same category? "
                        f"If yes, map {variant!r} to the more likely canonical label {canonical!r}. "
                        "Defer if they could be scientifically distinct."
                    ),
                    title="AI-reviewed fuzzy category merge",
                )
            )

    return output


def build_ai_candidates(
    report: DatasetReport,
    tables: dict[str, pd.DataFrame],
) -> list[AICandidate]:
    candidates: list[AICandidate] = []

    for issue in report.issues:
        if issue.table not in tables:
            continue
        frame = tables[issue.table]

        if issue.type in {"category_inconsistency", "fuzzy_category_match"}:
            candidates.extend(_category_candidates(issue, frame))
            continue

        if issue.column is None or issue.column not in frame.columns:
            continue

        if issue.type == "hidden_missing_values":
            values = sorted(str(value) for value in issue.evidence.get("tokens", {}))
            if values:
                candidates.append(
                    _make_candidate(
                        action_type=ActionType.NORMALIZE_MISSING,
                        table=issue.table,
                        column=issue.column,
                        parameters={"values": values},
                        issue=issue,
                        evidence={
                            "tokens": issue.evidence.get("tokens", {}),
                            "affected_count": issue.affected_count,
                        },
                        question=(
                            "Are these detector-confirmed missing-like tokens safe to normalize "
                            "to true null values for this field?"
                        ),
                        title="AI-reviewed missing-token normalization",
                    )
                )

        elif issue.type == "malformed_or_mixed_dates":
            candidates.append(
                _make_candidate(
                    action_type=ActionType.STANDARDIZE_DATES,
                    table=issue.table,
                    column=issue.column,
                    parameters={"format": "%Y-%m-%d", "unparseable": "keep"},
                    issue=issue,
                    evidence=issue.evidence,
                    question=(
                        "Should parseable dates be standardized to YYYY-MM-DD while every "
                        "unparseable value is left unchanged?"
                    ),
                    title="AI-reviewed date standardization",
                )
            )

        elif issue.type == "mixed_numeric_values":
            if looks_like_identifier_name(issue.column):
                continue
            candidates.append(
                _make_candidate(
                    action_type=ActionType.COERCE_NUMERIC,
                    table=issue.table,
                    column=issue.column,
                    parameters={"unparseable": "keep"},
                    issue=issue,
                    evidence=issue.evidence,
                    question=(
                        "Should parseable values in this mostly numeric non-identifier field be "
                        "converted to numbers while unparseable values remain unchanged?"
                    ),
                    title="AI-reviewed numeric coercion",
                )
            )

        elif issue.type == "surrounding_whitespace":
            candidates.append(
                _make_candidate(
                    action_type=ActionType.TRIM_WHITESPACE,
                    table=issue.table,
                    column=issue.column,
                    parameters={},
                    issue=issue,
                    evidence=issue.evidence,
                    question=(
                        "Should leading/trailing whitespace be removed while internal whitespace "
                        "and all other characters remain unchanged?"
                    ),
                    title="AI-reviewed whitespace trimming",
                )
            )

        # Deliberately no AI candidate for exact duplicate deletion. Row deletion
        # remains a deterministic finding and explicit human-review action.

    deduped = list({candidate.id: candidate for candidate in candidates}.values())
    for index, candidate in enumerate(deduped, start=1):
        session_id = f"C{index:03d}"
        candidate.id = session_id
        candidate.context.id = session_id
    return deduped
