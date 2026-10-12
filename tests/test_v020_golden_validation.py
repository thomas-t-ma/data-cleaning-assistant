from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from data_cleaning_assistant.cli import app
from data_cleaning_assistant.golden_validation import run_golden_validation


SOURCE = (
    "Employee_ID,Join_Date,Salary,Phone\n"
    "EMP001,11/27/2021,55000,-1651623197\n"
    "EMP002,6/30/2023,N/A,-1898471390\n"
    "EMP003,9/21/2022,62000,-5596363211\n"
    "EMP004,4/2/2021,58000,-3476490784\n"
    "EMP005,7/10/2020,60000,-1586734256\n"
)

GOLDEN = (
    "Employee_ID,Join_Date,Salary,Phone\n"
    "EMP001,2021-11-27,55000,-1651623197\n"
    "EMP002,2023-06-30,,-1898471390\n"
    "EMP003,2022-09-21,62000,-5596363211\n"
    "EMP004,2021-04-02,58000,-3476490784\n"
    "EMP005,2020-07-10,60000,-1586734256\n"
)


def _write_pair(tmp_path: Path, expected: str = GOLDEN) -> tuple[Path, Path]:
    source = tmp_path / "employees.csv"
    golden = tmp_path / "employees.golden.csv"
    source.write_text(SOURCE, encoding="utf-8")
    golden.write_text(expected, encoding="utf-8")
    return source, golden


def test_golden_validation_passes_for_human_approved_employee_cleanup(
    tmp_path: Path,
) -> None:
    source, golden = _write_pair(tmp_path)

    report = run_golden_validation(source, golden)

    assert report.passed is True
    assert report.metrics.planned_actions == 2
    assert report.metrics.executed_actions == 2
    assert report.metrics.expected_changed_cells == 6
    assert report.metrics.actual_changed_cells == 6
    assert report.metrics.matched_expected_changes == 6
    assert report.metrics.missed_expected_changes == 0
    assert report.metrics.unexpected_changes == 0
    assert report.metrics.wrong_replacements == 0
    assert report.metrics.remaining_different_cells == 0
    assert set(report.planned_action_summaries) == {
        "standardize_dates:employees.Join_Date",
        "normalize_missing:employees.Salary",
    }


def test_golden_validation_flags_stale_partially_cleaned_date_output(
    tmp_path: Path,
) -> None:
    stale = GOLDEN.replace("EMP004,2021-04-02", "EMP004,4/2/2021")
    source, expected = _write_pair(tmp_path, stale)

    report = run_golden_validation(source, expected)

    assert report.passed is False
    assert report.metrics.unexpected_changes == 1
    assert report.metrics.remaining_different_cells == 1
    assert report.metrics.missed_expected_changes == 0
    assert report.metrics.wrong_replacements == 0

    difference = report.differences[0]
    assert difference.kind == "unexpected_change"
    assert difference.column == "Join_Date"
    assert difference.source == "4/2/2021"
    assert difference.expected == "4/2/2021"
    assert difference.actual == "2021-04-02"


def test_validate_golden_cli_writes_machine_readable_report(tmp_path: Path) -> None:
    source, golden = _write_pair(tmp_path)
    output = tmp_path / "golden_report.json"

    result = CliRunner().invoke(
        app,
        [
            "validate-golden",
            str(source),
            str(golden),
            "--json-output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "GOLDEN VALIDATION PASS" in result.output
    assert "Matched expected changes" in result.output

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["metrics"]["expected_changed_cells"] == 6
    assert payload["metrics"]["remaining_different_cells"] == 0


def test_golden_validation_compares_serialized_numeric_output(tmp_path: Path) -> None:
    source = tmp_path / "inventory.csv"
    golden = tmp_path / "inventory.golden.csv"
    source.write_text(
        "SKU,Quantity\n"
        "00123,10\n"
        "00456,12\n"
        "00789,backorder\n"
        "00999,8\n"
        "00111,15\n",
        encoding="utf-8",
    )
    golden.write_text(
        "SKU,Quantity\n"
        "00123,10\n"
        "00456,12\n"
        "00789,backorder\n"
        "00999,8\n"
        "00111,15\n",
        encoding="utf-8",
    )

    report = run_golden_validation(source, golden)

    assert report.passed is True
    assert report.metrics.planned_actions == 1
    assert report.metrics.executed_actions == 1
    assert report.metrics.expected_changed_cells == 0
    assert report.metrics.actual_changed_cells == 0
    assert report.metrics.remaining_different_cells == 0
