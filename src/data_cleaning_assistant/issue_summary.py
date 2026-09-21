from __future__ import annotations

from collections import Counter, defaultdict

from .models import DatasetReport, Issue


NOISE_TYPES = {"hidden_missing_values", "constant_column"}


def _coverage_noise(issue: Issue, row_count: int) -> bool:
    """Group only complete-coverage coverage/constant findings, never schema QA."""
    return (
        row_count > 0
        and issue.type in NOISE_TYPES
        and issue.column is not None
        and issue.affected_count == row_count
    )


def summarize_issues(report: DatasetReport) -> dict:
    """A lossless, aggregate index over the full report; original issues remain intact."""
    row_counts = {table.name: table.row_count for table in report.tables}
    summary: dict = {
        "total_issues": len(report.issues),
        "by_severity": dict(Counter(issue.severity.value for issue in report.issues)),
        "by_type": dict(Counter(issue.type for issue in report.issues)),
        "by_table": {},
    }
    for table in report.tables:
        local = [i for i in report.issues if i.table == table.name]
        coverage = [i for i in local if _coverage_noise(i, row_counts[table.name])]
        grouped: dict[str, set[str]] = defaultdict(set)
        for issue in coverage:
            if issue.column is not None:
                grouped[issue.type].add(issue.column)
        summary["by_table"][table.name] = {
            "total_issues": len(local),
            "coverage_noise_issues": len(coverage),
            "coverage_noise_columns": {
                kind: sorted(columns) for kind, columns in sorted(grouped.items())
            },
            "other_issues": len(local) - len(coverage),
        }
    return summary


def visible_issues(report: DatasetReport) -> list[Issue]:
    rows = {table.name: table.row_count for table in report.tables}
    return [
        issue
        for issue in report.issues
        if not _coverage_noise(issue, rows.get(issue.table, 0))
    ]
