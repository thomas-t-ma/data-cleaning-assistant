from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from data_cleaning_assistant.cli import app
from data_cleaning_assistant.evaluation import run_offline_evaluation


@pytest.fixture(scope="module")
def offline_report():
    return run_offline_evaluation()


def test_offline_evaluation_has_complete_safe_benchmark_baseline(
    offline_report,
) -> None:
    report = offline_report

    assert report.benchmark_case_count == 20
    assert report.benchmark_passed_cases == 20
    assert report.required_total == 10
    assert report.required_hits == 10
    assert report.required_recall == 1.0

    assert report.unsafe_executable_proposals == 0
    assert report.prohibited_candidate_exposures == 0
    assert report.safety_pass is True
    assert report.coverage_pass is True
    assert report.overall_pass is True

    assert report.deterministic_actions > 0
    assert report.semantic_review_candidates > 0


def test_semantic_coverage_is_reported_without_treating_exposure_as_merge(
    offline_report,
) -> None:
    report = offline_report
    semantic = report.semantic_coverage

    assert semantic.total_cases == 40
    assert semantic.safe_cases == 20
    assert semantic.risky_cases == 20
    assert 0 <= semantic.safe_exposed <= semantic.safe_cases
    assert 0 <= semantic.risky_exposed <= semantic.risky_cases
    assert 0.0 <= semantic.safe_coverage <= 1.0
    assert 0.0 <= semantic.risky_exposure_rate <= 1.0
    assert semantic.unexpected_candidate_exposures == 0

    # Risky semantic pairs may be deliberately surfaced for accept/defer review.
    # Their mere exposure must not count as an unsafe executable proposal.
    assert report.unsafe_executable_proposals == 0
    assert report.safety_pass is True


def test_offline_evaluation_cli_writes_machine_readable_scorecard(
    tmp_path: Path,
) -> None:
    output = tmp_path / "offline_evaluation.json"

    result = CliRunner().invoke(
        app,
        [
            "evaluate-offline",
            "--json-output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "OFFLINE EVALUATION PASS" in result.output
    assert "Risky-pair exposure" in result.output
    assert output.exists()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["overall_pass"] is True
    assert payload["benchmark_case_count"] == 20
    assert payload["required_total"] == 10
    assert payload["unsafe_executable_proposals"] == 0
    assert payload["prohibited_candidate_exposures"] == 0
    assert payload["semantic_coverage"]["total_cases"] == 40
