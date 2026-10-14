from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel, Field

from .engine import profile_dataset
from .golden_validation import GoldenValidationMetrics, run_golden_validation
from .planner import propose_actions
from .realworld_corpus import realworld_corpus_cases


class CorpusCaseResult(BaseModel):
    case_id: str
    domain: str
    source_inspiration: str
    source_url: str
    passed: bool
    golden_pass: bool
    protected_columns_pass: bool
    review_findings_pass: bool
    golden_metrics: GoldenValidationMetrics
    planned_action_summaries: list[str] = Field(default_factory=list)
    forbidden_columns_exposed: list[str] = Field(default_factory=list)
    missing_review_findings: list[str] = Field(default_factory=list)
    golden_differences: list[dict] = Field(default_factory=list)


class CorpusValidationReport(BaseModel):
    passed: bool
    case_count: int
    passed_cases: int
    cases: list[CorpusCaseResult] = Field(default_factory=list)


def run_realworld_corpus_validation() -> CorpusValidationReport:
    """Run the built-in source-inspired corpus entirely offline."""
    results: list[CorpusCaseResult] = []

    with TemporaryDirectory(prefix="data-cleaner-realworld-corpus-") as temp_dir:
        root = Path(temp_dir)

        for case in realworld_corpus_cases():
            source = root / f"{case.id}.dirty.csv"
            golden = root / f"{case.id}.golden.csv"
            source.write_text(case.dirty_csv, encoding="utf-8")
            golden.write_text(case.golden_csv, encoding="utf-8")

            golden_report = run_golden_validation(source, golden)
            plan = propose_actions(source)
            profile = profile_dataset(source)

            forbidden = set(case.forbidden_action_columns)
            forbidden_exposed = sorted(
                {
                    str(action.column)
                    for action in plan.actions
                    if action.column is not None and action.column in forbidden
                }
            )

            observed_issues = {
                (str(issue.column), issue.type)
                for issue in profile.issues
                if issue.column is not None
            }
            missing_review: list[str] = []
            for column, issue_types in case.expected_review_issues.items():
                for issue_type in issue_types:
                    if (column, issue_type) not in observed_issues:
                        missing_review.append(f"{column}:{issue_type}")

            protected_pass = not forbidden_exposed
            review_pass = not missing_review
            passed = golden_report.passed and protected_pass and review_pass
            results.append(
                CorpusCaseResult(
                    case_id=case.id,
                    domain=case.domain,
                    source_inspiration=case.source_inspiration,
                    source_url=case.source_url,
                    passed=passed,
                    golden_pass=golden_report.passed,
                    protected_columns_pass=protected_pass,
                    review_findings_pass=review_pass,
                    golden_metrics=golden_report.metrics,
                    planned_action_summaries=golden_report.planned_action_summaries,
                    forbidden_columns_exposed=forbidden_exposed,
                    missing_review_findings=missing_review,
                    golden_differences=[
                        item.model_dump()
                        for item in golden_report.differences
                    ],
                )
            )

    passed_cases = sum(item.passed for item in results)
    return CorpusValidationReport(
        passed=passed_cases == len(results),
        case_count=len(results),
        passed_cases=passed_cases,
        cases=results,
    )
