from __future__ import annotations

from typing import Any

from .ai_models import (
    AIColumnContext,
    AIIssueContext,
    AIPlannerContext,
    AIRelationshipContext,
    AITableContext,
)
from .models import DatasetReport


ACTIONABLE_ISSUE_TYPES = {
    "category_inconsistency",
    "fuzzy_category_match",
    "hidden_missing_values",
    "malformed_or_mixed_dates",
    "mixed_numeric_values",
    "surrounding_whitespace",
    "exact_duplicate_rows",
}


def _compact_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 4:
        return "<truncated>"
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 30:
                output["<truncated>"] = True
                break
            output[str(key)] = _compact_value(item, depth=depth + 1)
        return output
    if isinstance(value, list):
        return [_compact_value(item, depth=depth + 1) for item in value[:30]]
    if isinstance(value, tuple):
        return [_compact_value(item, depth=depth + 1) for item in value[:30]]
    if isinstance(value, str):
        return value[:300]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:300]


def build_ai_context(report: DatasetReport) -> AIPlannerContext:
    tables: list[AITableContext] = []
    for table in report.tables:
        tables.append(
            AITableContext(
                name=table.name,
                row_count=table.row_count,
                column_count=table.column_count,
                grain=table.grain.label if table.grain else None,
                grain_confidence=table.grain.confidence if table.grain else None,
                candidate_keys=[key.columns for key in table.candidate_keys[:5]],
                columns=[
                    AIColumnContext(
                        name=column.name,
                        inferred_type=column.inferred_type,
                        type_confidence=column.type_confidence,
                        null_percent=column.null_percent,
                        unique_count=column.unique_count,
                        unique_percent=column.unique_percent,
                        numeric_summary=column.numeric_summary,
                    )
                    for column in table.columns
                ],
            )
        )

    issues = [
        AIIssueContext(
            id=issue.id,
            type=issue.type,
            severity=issue.severity.value,
            table=issue.table,
            column=issue.column,
            title=issue.title,
            confidence=issue.confidence,
            affected_count=issue.affected_count,
            evidence=_compact_value(issue.evidence),
        )
        for issue in report.issues
        if issue.type in ACTIONABLE_ISSUE_TYPES
    ]

    relationships = [
        AIRelationshipContext(
            left=f"{rel.left_table}.{rel.left_column}",
            right=f"{rel.right_table}.{rel.right_column}",
            relationship=rel.relationship,
            confidence=rel.confidence,
        )
        for rel in report.relationships[:30]
    ]

    return AIPlannerContext(
        tables=tables,
        issues=issues,
        relationships=relationships,
    )
