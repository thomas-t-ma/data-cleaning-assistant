from __future__ import annotations

from pathlib import Path

import pytest

from data_cleaning_assistant.engine import profile_dataset
from data_cleaning_assistant.golden_validation import run_golden_validation
from data_cleaning_assistant.planner import propose_actions
from data_cleaning_assistant.realworld_corpus import realworld_corpus_cases


def _write_case(tmp_path: Path, case) -> tuple[Path, Path]:
    source = tmp_path / f"{case.id}.dirty.csv"
    golden = tmp_path / f"{case.id}.golden.csv"
    source.write_text(case.dirty_csv, encoding="utf-8")
    golden.write_text(case.golden_csv, encoding="utf-8")
    return source, golden


def test_realworld_corpus_covers_required_domains() -> None:
    cases = realworld_corpus_cases()

    assert len(cases) == 6
    assert {case.domain for case in cases} == {
        "sales/orders",
        "survey/free text",
        "inventory/wholesale",
        "finance",
        "CRM/contact",
        "public service/free text",
    }


@pytest.mark.parametrize("case", realworld_corpus_cases(), ids=lambda case: case.id)
def test_realworld_golden_pairs_pass(tmp_path: Path, case) -> None:
    source, golden = _write_case(tmp_path, case)

    report = run_golden_validation(source, golden)

    assert report.passed is True, (
        case.id,
        report.metrics.model_dump(),
        [item.model_dump() for item in report.differences],
        report.planned_action_summaries,
    )


@pytest.mark.parametrize("case", realworld_corpus_cases(), ids=lambda case: case.id)
def test_realworld_corpus_preserves_protected_columns(tmp_path: Path, case) -> None:
    source, _golden = _write_case(tmp_path, case)
    plan = propose_actions(source)

    exposed = {
        action.column
        for action in plan.actions
        if action.column in set(case.forbidden_action_columns)
    }
    assert exposed == set(), (case.id, exposed, [action.model_dump() for action in plan.actions])


@pytest.mark.parametrize("case", realworld_corpus_cases(), ids=lambda case: case.id)
def test_realworld_corpus_surfaces_expected_review_only_findings(
    tmp_path: Path,
    case,
) -> None:
    if not case.expected_review_issues:
        pytest.skip("Case has no explicit review-only expectation.")

    source, _golden = _write_case(tmp_path, case)
    report = profile_dataset(source)

    observed = {
        (issue.column, issue.type)
        for issue in report.issues
    }
    missing: list[tuple[str, str]] = []
    for column, issue_types in case.expected_review_issues.items():
        for issue_type in issue_types:
            if (column, issue_type) not in observed:
                missing.append((column, issue_type))

    assert missing == [], (case.id, missing, sorted(observed))
