from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from .inference import (
    infer_semantic_type,
    is_missing_like,
    looks_like_free_text_response,
    looks_like_identifier_name,
    normalize_text,
)


_TOKEN_PATTERN = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?")
_NUMERIC_TOKEN = re.compile(r"^\d+(?:\.\d+)?$")
_COUNT_NAME_PATTERN = re.compile(r"\b(how many|count|number)\b", re.IGNORECASE)


class RecodeVariantGroup(BaseModel):
    values: dict[str, int] = Field(default_factory=dict)
    total_count: int = Field(ge=1)
    representative: str | None = None


class RecodeSuggestion(BaseModel):
    table: str
    column: str
    left: RecodeVariantGroup
    right: RecodeVariantGroup
    differing_token_position: int = Field(ge=0)
    edit_distance: int = Field(ge=1)
    suggested_target: str | None = None
    target_basis: str | None = None
    component_mode: bool = False
    review_required: bool = True


class RecodeSuggestionReport(BaseModel):
    version: int = 1
    source_path: str
    source_sha256: str
    interpretation_note: str = (
        "Typo-shaped recoding suggestions are review aids only. They do not "
        "authorize a transformation. Exact case/whitespace variants are grouped "
        "before comparison, and numeric/version tokens must match exactly."
    )
    suggestions: list[RecodeSuggestion] = Field(default_factory=list)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _token_parts(value: str) -> tuple[list[str], list[str]]:
    tokens = _TOKEN_PATTERN.findall(value)
    numeric = [token for token in tokens if _NUMERIC_TOKEN.fullmatch(token)]
    text = [token.casefold() for token in tokens if not _NUMERIC_TOKEN.fullmatch(token)]
    return text, numeric


def _typo_shape(left: str, right: str) -> tuple[int, int] | None:
    left_text, left_numeric = _token_parts(left)
    right_text, right_numeric = _token_parts(right)
    if left_numeric != right_numeric:
        return None
    if len(left_text) != len(right_text) or len(left_text) < 2:
        return None

    differing = [
        index
        for index, (left_token, right_token) in enumerate(
            zip(left_text, right_text, strict=True)
        )
        if left_token != right_token
    ]
    if len(differing) != 1:
        return None

    index = differing[0]
    left_token = left_text[index]
    right_token = right_text[index]
    if min(len(left_token), len(right_token)) < 4:
        return None

    distance = _levenshtein(left_token, right_token)
    allowed = 1 if max(len(left_token), len(right_token)) < 8 else 2
    if distance == 0 or distance > allowed:
        return None
    return index, distance


def _equivalence_groups(
    counts: Counter[str],
) -> list[tuple[RecodeVariantGroup, str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for value in counts:
        grouped[normalize_text(value)].append(value)

    result: list[tuple[RecodeVariantGroup, str]] = []
    for variants in grouped.values():
        highest = max(counts[value] for value in variants)
        winners = [value for value in variants if counts[value] == highest]
        representative = winners[0] if len(winners) == 1 else None
        comparison_value = representative or sorted(variants)[0]
        result.append(
            (
                RecodeVariantGroup(
                    values={value: counts[value] for value in sorted(variants)},
                    total_count=sum(counts[value] for value in variants),
                    representative=representative,
                ),
                comparison_value,
            )
        )
    return result


def suggest_recodes(
    path: str | Path,
    tables: dict[str, pd.DataFrame],
    *,
    max_observed_values: int = 100,
    max_suggestions_per_column: int = 20,
    dominance_ratio: float = 3.0,
) -> RecodeSuggestionReport:
    """Surface typo-shaped categorical recoding candidates without creating actions."""
    resolved = Path(path).expanduser().resolve()
    suggestions: list[RecodeSuggestion] = []

    for table, frame in tables.items():
        for raw_column in frame.columns:
            column = str(raw_column)
            series = frame[raw_column]
            if looks_like_identifier_name(column):
                continue
            if looks_like_free_text_response(column, series):
                continue
            if _COUNT_NAME_PATTERN.search(column):
                continue

            inferred_type, _confidence, _ratios = infer_semantic_type(column, series)
            if inferred_type in {"identifier", "numeric", "date"}:
                continue

            values = [
                str(value).strip()
                for value in series.tolist()
                if not is_missing_like(value) and str(value).strip()
            ]
            if len(values) < 10:
                continue
            delimited_count = sum(";" in value for value in values)
            component_mode = delimited_count >= max(
                5, math.ceil(len(values) * 0.01)
            )
            observed_values = (
                [
                    component.strip()
                    for value in values
                    for component in value.split(";")
                    if component.strip()
                ]
                if component_mode
                else values
            )

            counts: Counter[str] = Counter(observed_values)
            if not 2 <= len(counts) <= max_observed_values:
                continue

            groups = _equivalence_groups(counts)
            candidates: list[RecodeSuggestion] = []
            for (left_group, left_value), (right_group, right_value) in itertools.combinations(
                groups,
                2,
            ):
                evidence = _typo_shape(left_value, right_value)
                if evidence is None:
                    continue
                differing_position, edit_distance = evidence

                suggested_target = None
                target_basis = None
                if (
                    left_group.representative is not None
                    and left_group.total_count
                    >= dominance_ratio * right_group.total_count
                ):
                    suggested_target = left_group.representative
                    target_basis = "observed_frequency_dominance"
                elif (
                    right_group.representative is not None
                    and right_group.total_count
                    >= dominance_ratio * left_group.total_count
                ):
                    suggested_target = right_group.representative
                    target_basis = "observed_frequency_dominance"

                candidates.append(
                    RecodeSuggestion(
                        table=table,
                        column=column,
                        left=left_group,
                        right=right_group,
                        differing_token_position=differing_position,
                        edit_distance=edit_distance,
                        suggested_target=suggested_target,
                        target_basis=target_basis,
                        component_mode=component_mode,
                    )
                )

            candidates.sort(
                key=lambda item: (
                    item.suggested_target is None,
                    item.edit_distance,
                    -max(item.left.total_count, item.right.total_count),
                    sorted(item.left.values)[0],
                    sorted(item.right.values)[0],
                )
            )
            suggestions.extend(candidates[:max_suggestions_per_column])

    suggestions.sort(
        key=lambda item: (
            item.table,
            item.column,
            item.suggested_target is None,
            item.edit_distance,
            -max(item.left.total_count, item.right.total_count),
        )
    )
    return RecodeSuggestionReport(
        source_path=str(resolved),
        source_sha256=_sha256(resolved),
        suggestions=suggestions,
    )
