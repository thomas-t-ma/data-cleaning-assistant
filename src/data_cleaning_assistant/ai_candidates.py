from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .actions import ActionOrigin, ActionType, CleaningAction
from .ai_models import AICandidateContext
from .inference import looks_like_identifier_name
from .models import DatasetReport, Issue
from .planning_policy import structural_only_column
from .study_schema import StudySchema


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


def _zero_padding_equivalent(left: str, right: str) -> bool:
    """Return True when labels differ only by zero-padding of equal numeric tokens."""
    left_parts = re.findall(r"\d+|\D+", left.strip().casefold())
    right_parts = re.findall(r"\d+|\D+", right.strip().casefold())
    if len(left_parts) != len(right_parts):
        return False

    saw_padding_difference = False
    for left_part, right_part in zip(left_parts, right_parts, strict=True):
        left_is_num = left_part.isdigit()
        right_is_num = right_part.isdigit()
        if left_is_num != right_is_num:
            return False
        if left_is_num:
            if int(left_part) != int(right_part):
                return False
            if left_part != right_part:
                saw_padding_difference = True
        elif left_part != right_part:
            return False
    return saw_padding_difference


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            insert_cost = current[right_index - 1] + 1
            delete_cost = previous[right_index] + 1
            replace_cost = previous[right_index - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def _single_edit_typo_evidence(
    left: str,
    right: str,
    counts: Counter[str],
) -> dict[str, Any] | None:
    """Identify a conservative single-token, single-edit typo pattern.

    This intentionally excludes numeric-token changes and very short tokens so
    distinctions like Dose 5 vs Dose 50, Stage II vs Stage III, and Arm A vs Arm B
    do not receive typo evidence.
    """
    left_tokens = re.findall(r"\w+|[^\w\s]", left.casefold())
    right_tokens = re.findall(r"\w+|[^\w\s]", right.casefold())
    if len(left_tokens) != len(right_tokens):
        return None

    differences = [
        (index, left_token, right_token)
        for index, (left_token, right_token) in enumerate(
            zip(left_tokens, right_tokens, strict=True)
        )
        if left_token != right_token
    ]
    if len(differences) != 1:
        return None

    index, left_token, right_token = differences[0]
    if left_token.isdigit() or right_token.isdigit():
        return None
    if min(len(left_token), len(right_token)) < 4:
        return None

    distance = _levenshtein_distance(left_token, right_token)
    if distance != 1:
        return None

    variant, canonical = _canonical_pair(left, right, counts)
    canonical_count = counts[canonical]
    variant_count = counts[variant]
    return {
        "single_edit_typo_pattern": True,
        "differing_token_index": index,
        "left_token": left_token,
        "right_token": right_token,
        "edit_distance": distance,
        "unchanged_token_count": len(left_tokens) - 1,
        "variant_count": variant_count,
        "canonical_count": canonical_count,
        "canonical_more_frequent": canonical_count > variant_count,
    }


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


def _is_composite_multiselect_label(value: str) -> bool:
    """Return True for delimiter-composed multi-select responses.

    Whole-string fuzzy similarity is unsafe for these values: a response that
    contains an additional real option can look nearly identical to its subset.
    Composite labels need component-aware cleaning rather than whole-value merges.
    """
    return ";" in value


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
            if (
                _is_composite_multiselect_label(left)
                or _is_composite_multiselect_label(right)
            ):
                continue
            variant, canonical = _canonical_pair(left, right, counts)
            zero_padding = _zero_padding_equivalent(left, right)
            typo_evidence = _single_edit_typo_evidence(left, right, counts)
            if zero_padding:
                basis = "numeric_zero_padding_equivalence"
            elif typo_evidence is not None:
                basis = "single_edit_typo_pattern"
            else:
                basis = "fuzzy_similarity_only"

            question = (
                f"Do {left!r} and {right!r} clearly mean the same category? "
                f"If yes, map {variant!r} to the more likely canonical label {canonical!r}. "
                "Defer if they could be scientifically distinct."
            )
            if zero_padding:
                question = (
                    f"{left!r} and {right!r} have identical surrounding text and equal numeric "
                    "tokens after removing leading zeros. Is this formatting-only difference safe "
                    f"to normalize by mapping {variant!r} to {canonical!r}?"
                )
            elif typo_evidence is not None:
                question = (
                    f"{left!r} and {right!r} differ in exactly one token by one character edit, "
                    "all other tokens are identical, and the proposed canonical label is "
                    f"{canonical!r}. Is this strong typo-shaped evidence sufficient to normalize "
                    f"{variant!r} to {canonical!r}? Defer if the token difference could encode a "
                    "real scientific distinction."
                )
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
                        "basis": basis,
                        "numeric_zero_padding_equivalent": zero_padding,
                        "single_edit_typo_evidence": typo_evidence,
                    },
                    question=question,
                    title="AI-reviewed fuzzy category merge",
                )
            )

    return output


def build_ai_candidates(
    report: DatasetReport,
    tables: dict[str, pd.DataFrame],
    *,
    schema: StudySchema | None = None,
) -> list[AICandidate]:
    candidates: list[AICandidate] = []

    for issue in report.issues:
        if issue.table not in tables:
            continue
        frame = tables[issue.table]

        column_spec = (
            schema.tables[issue.table].columns.get(issue.column)
            if schema is not None
            and issue.table in schema.tables
            and issue.column is not None
            else None
        )

        if issue.type in {"category_inconsistency", "fuzzy_category_match"}:
            # The study dictionary, when provided, is the only authority for
            # category normalization in a schema-managed column.
            if column_spec is None or not column_spec.has_category_dictionary():
                candidates.extend(_category_candidates(issue, frame))
            continue

        if issue.column is None or issue.column not in frame.columns:
            continue

        if issue.type == "hidden_missing_values":
            if structural_only_column(issue, frame):
                continue
            if column_spec is not None and column_spec.missing_tokens:
                continue
            source = issue.evidence.get("structural_tokens")
            if source is None:
                source = issue.evidence.get("tokens", {})
            values = sorted(str(value) for value in source)
            if values:
                candidates.append(
                    _make_candidate(
                        action_type=ActionType.NORMALIZE_MISSING,
                        table=issue.table,
                        column=issue.column,
                        parameters={"values": values},
                        issue=issue,
                        evidence={
                            "structural_tokens": source,
                            "contextual_tokens_not_in_candidate": issue.evidence.get(
                                "contextual_tokens", {}
                            ),
                            "affected_count": sum(int(value) for value in source.values()),
                            "basis": "structural_null_sentinel",
                        },
                        question=(
                            "Are these structural null sentinels safe to normalize to true null "
                            "values for this field? Contextual labels such as Unknown or Not "
                            "applicable are intentionally excluded from this candidate."
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


def semantic_review_candidates(candidates: list[AICandidate]) -> list[AICandidate]:
    """Keep only candidates whose decision actually requires semantic judgment.

    Mechanical transformations and deterministic case/whitespace category
    equivalences remain local rule/human-review work. The model is reserved for
    fuzzy category candidates where lexical similarity alone cannot establish
    whether two labels have the same scientific meaning.
    """
    semantic_bases = {
        "fuzzy_similarity_only",
        "single_edit_typo_pattern",
        "numeric_zero_padding_equivalence",
    }
    selected = [
        candidate
        for candidate in candidates
        if candidate.action.type == ActionType.NORMALIZE_CATEGORIES
        and candidate.context.evidence.get("basis") in semantic_bases
    ]
    for index, candidate in enumerate(selected, start=1):
        session_id = f"C{index:03d}"
        candidate.id = session_id
        candidate.context.id = session_id
    return selected
