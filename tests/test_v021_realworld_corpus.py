from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from data_cleaning_assistant.cli import app
from data_cleaning_assistant.engine import profile_dataset
from data_cleaning_assistant.golden_validation import run_golden_validation
from data_cleaning_assistant.inference import (
    has_explicit_numeric_format,
    looks_like_identifier_name,
    parse_date,
    parse_number,
)
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


def test_timestamp_parsing_refuses_ambiguous_numeric_order_without_evidence() -> None:
    assert parse_date("10/11/2026 09:30") is None

    parsed = parse_date("10/11/2026 09:30", numeric_order="month_first")
    assert parsed is not None
    assert parsed.isoformat() == "2026-10-11T09:30:00"


def test_currency_and_accounting_numbers_are_parsed_without_general_text_stripping() -> None:
    assert parse_number("$1,250.00") == 1250.0
    assert parse_number("(85.50)") == -85.5
    assert parse_number("-£42.00") == -42.0
    assert parse_number("REF-0001") is None
    assert has_explicit_numeric_format("$1,250.00") is True
    assert has_explicit_numeric_format("(85.50)") is True
    assert has_explicit_numeric_format("00123") is False


def test_common_code_and_postal_names_are_identifier_like() -> None:
    for name in [
        "SKU",
        "StockCode",
        "Incident_Zip",
        "postal_code",
        "postcode",
        "Unique_Key",
    ]:
        assert looks_like_identifier_name(name), name


def test_golden_validator_accepts_expected_duplicate_row_removal(tmp_path: Path) -> None:
    case = next(item for item in realworld_corpus_cases() if item.id == "retail_orders")
    source, golden = _write_case(tmp_path, case)

    report = run_golden_validation(source, golden)

    assert report.passed is True
    assert report.metrics.source_alignment_available is False
    assert report.metrics.expected_removed_rows == 1
    assert report.metrics.actual_removed_rows == 1


def test_validate_corpus_cli_writes_machine_readable_report(tmp_path: Path) -> None:
    output = tmp_path / "realworld_corpus_report.json"

    result = CliRunner().invoke(
        app,
        ["validate-corpus", "--json-output", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert "REAL-WORLD CORPUS PASS" in result.output
    assert "6/6" in result.output

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["case_count"] == 6
    assert payload["passed_cases"] == 6
