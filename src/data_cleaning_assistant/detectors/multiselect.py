from __future__ import annotations

import uuid
from collections import Counter, defaultdict

import pandas as pd

from ..inference import is_missing_like, looks_like_free_text_response, normalize_text
from ..models import Issue, Severity


def _id() -> str:
    return f"issue_{uuid.uuid4().hex[:12]}"


def _components(value: str, delimiter: str) -> list[str]:
    return [part.strip() for part in value.split(delimiter) if part.strip()]


def detect_multiselect_component_variants(
    table: str,
    frame: pd.DataFrame,
    *,
    delimiter: str = ";",
) -> list[Issue]:
    """Detect exact case/whitespace variants inside structured multi-select cells.

    This intentionally does not perform fuzzy component matching. v0.9 showed
    that whole-value fuzzy similarity can erase legitimate selected options.
    """
    issues: list[Issue] = []
    for raw_column in frame.columns:
        column = str(raw_column)
        series = frame[raw_column]
        if looks_like_free_text_response(column, series):
            continue

        values = [
            str(value)
            for value in series.tolist()
            if not is_missing_like(value) and str(value).strip()
        ]
        delimited_values = [value for value in values if delimiter in value]
        if len(delimited_values) < 2:
            continue

        component_counts: Counter[str] = Counter()
        groups: dict[str, set[str]] = defaultdict(set)
        for value in values:
            for component in _components(value, delimiter):
                component_counts[component] += 1
                groups[normalize_text(component)].add(component)

        variants = [
            sorted(raw_variants)
            for raw_variants in groups.values()
            if len(raw_variants) > 1
        ]
        if not variants:
            continue

        variant_components = {
            component
            for group in variants
            for component in group
        }
        affected = sum(
            any(
                component in variant_components
                for component in _components(value, delimiter)
            )
            for value in values
        )
        issues.append(
            Issue(
                id=_id(),
                type="multiselect_component_inconsistency",
                severity=Severity.MEDIUM,
                table=table,
                column=column,
                title="Multi-select components contain equivalent-looking variants",
                description=(
                    "Semicolon-delimited responses contain component labels that differ "
                    "only by capitalization or surrounding whitespace. Components are "
                    "compared independently so legitimate additional selections are preserved."
                ),
                confidence=0.98,
                affected_count=affected,
                evidence={
                    "delimiter": delimiter,
                    "variant_groups": variants[:20],
                    "component_counts": {
                        component: component_counts[component]
                        for component in sorted(variant_components)
                    },
                },
                suggested_action=(
                    "Review component-level category normalization without merging "
                    "or dropping other selected options."
                ),
                auto_fix_safe=False,
            )
        )
    return issues
