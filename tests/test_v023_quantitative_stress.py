from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from data_cleaning_assistant.cli import app
from data_cleaning_assistant.actions import ActionType
from data_cleaning_assistant.planner import propose_actions
from data_cleaning_assistant.quantitative_stress import quantitative_stress_cases
from data_cleaning_assistant.quantitative_validation import (
    run_quantitative_stress_validation,
)


def test_quantitative_stress_corpus_is_large_and_cross_domain() -> None:
    cases = quantitative_stress_cases()

    assert len(cases) == 5
    assert sum(case.row_count for case in cases) == 2000
    assert {case.domain for case in cases} == {
        "HR/workforce",
        "sales/orders",
        "survey/free text",
        "finance",
        "logistics",
    }
    assert sum(len(case.expected_actions) for case in cases) == 12


def test_quantitative_stress_baseline_has_perfect_action_and_cell_metrics() -> None:
    report = run_quantitative_stress_validation()

    assert report.passed is True
    assert report.passed_cases == report.case_count == 5
    assert report.total_rows == 2000

    assert report.expected_actions == 12
    assert report.action_false_positives == 0
    assert report.action_false_negatives == 0
    assert report.action_precision == 1.0
    assert report.action_recall == 1.0
    assert report.action_f1 == 1.0

    assert report.expected_cell_edits >= 1500
    assert report.cell_false_positives == 0
    assert report.cell_false_negatives == 0
    assert report.cell_precision == 1.0
    assert report.cell_recall == 1.0
    assert report.cell_f1 == 1.0

    assert all(not case.protected_columns_exposed for case in report.cases)


def test_validate_stress_cli_writes_machine_readable_report(
    tmp_path: Path,
) -> None:
    output = tmp_path / "stress_report.json"

    result = CliRunner().invoke(
        app,
        ["validate-stress", "--json-output", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert "QUANTITATIVE STRESS PASS" in result.output
    assert "Action precision" in result.output
    assert "Cell edit recall" in result.output

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["total_rows"] == 2000
    assert payload["action_precision"] == 1.0
    assert payload["action_recall"] == 1.0
    assert payload["cell_precision"] == 1.0
    assert payload["cell_recall"] == 1.0


def test_planner_collapses_pure_whitespace_category_redundancy(
    tmp_path: Path,
) -> None:
    source = tmp_path / "whitespace.csv"
    source.write_text(
        "id,status\n"
        "A001,Active\n"
        "A002, Active \n"
        "A003,Active\n"
        "A004, Active \n",
        encoding="utf-8",
    )

    plan = propose_actions(source)
    keys = {(action.type, action.column) for action in plan.actions}

    assert (ActionType.TRIM_WHITESPACE, "status") in keys
    assert (ActionType.NORMALIZE_CATEGORIES, "status") not in keys


def test_planner_keeps_real_case_normalization_distinct(tmp_path: Path) -> None:
    source = tmp_path / "case.csv"
    source.write_text(
        "id,status\n"
        "A001,Active\n"
        "A002,active\n"
        "A003,Active\n"
        "A004,active\n",
        encoding="utf-8",
    )

    plan = propose_actions(source)
    keys = {(action.type, action.column) for action in plan.actions}

    assert (ActionType.NORMALIZE_CATEGORIES, "status") in keys
