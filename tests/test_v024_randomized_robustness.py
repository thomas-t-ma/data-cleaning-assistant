from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from data_cleaning_assistant.cli import app
from data_cleaning_assistant.randomized_adversarial import randomized_adversarial_cases
from data_cleaning_assistant.randomized_validation import run_randomized_validation


def test_randomized_generator_is_deterministic_and_cross_domain() -> None:
    first = randomized_adversarial_cases()
    second = randomized_adversarial_cases()

    assert len(first) == len(second) == 30
    assert sum(case.row_count for case in first) == 3000
    assert sum(len(case.expected_actions) for case in first) == 75
    assert {case.domain for case in first} == {
        "HR/workforce",
        "sales/orders",
        "survey/free text",
        "finance",
        "logistics",
        "CRM/contact",
    }

    assert [case.id for case in first] == [case.id for case in second]
    assert [case.dirty_csv for case in first] == [case.dirty_csv for case in second]
    assert [case.golden_csv for case in first] == [case.golden_csv for case in second]


def test_randomized_generator_changes_with_seed() -> None:
    baseline = randomized_adversarial_cases(seeds=(101,), rows_per_case=40)
    alternate = randomized_adversarial_cases(seeds=(997,), rows_per_case=40)

    assert [case.id for case in baseline] != [case.id for case in alternate]
    assert [case.dirty_csv for case in baseline] != [
        case.dirty_csv for case in alternate
    ]


def test_randomized_robustness_baseline_preserves_precision_and_recall() -> None:
    report = run_randomized_validation()

    assert report.case_count == 30
    assert report.total_rows == 3000
    assert report.expected_actions == 75

    # This is intentionally a hard gate. If a seed exposes a miss or an
    # unnecessary action, CI should fail and report the exact case rather than
    # relaxing the benchmark.
    assert report.passed is True
    assert report.passed_cases == 30
    assert report.action_false_positives == 0
    assert report.action_false_negatives == 0
    assert report.action_precision == 1.0
    assert report.action_recall == 1.0
    assert report.cell_false_positives == 0
    assert report.cell_false_negatives == 0
    assert report.cell_precision == 1.0
    assert report.cell_recall == 1.0
    assert all(not case.protected_columns_exposed for case in report.cases)


def test_validate_randomized_cli_writes_machine_readable_report(
    tmp_path: Path,
) -> None:
    output = tmp_path / "randomized_report.json"

    result = CliRunner().invoke(
        app,
        ["validate-randomized", "--json-output", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert "RANDOMIZED VALIDATION PASS" in result.output
    assert "Action precision" in result.output
    assert "Cell edit recall" in result.output

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["case_count"] == 30
    assert payload["total_rows"] == 3000
    assert payload["expected_actions"] == 75
