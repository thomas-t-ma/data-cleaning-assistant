from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from data_cleaning_assistant.cli import app
from data_cleaning_assistant.recall_benchmark import (
    run_v010_recall_benchmark,
    v010_recall_benchmark_cases,
)


def test_v010_recall_benchmark_passes_frozen_suite() -> None:
    summary = run_v010_recall_benchmark()

    assert summary.case_count == len(v010_recall_benchmark_cases())
    assert summary.pass_rate == 1.0
    assert summary.required_recall == 1.0
    assert summary.forbidden_hits == 0


def test_v010_recall_benchmark_has_positive_and_negative_controls() -> None:
    cases = v010_recall_benchmark_cases()

    assert any(case.required for case in cases)
    assert any(case.forbidden for case in cases)
    assert any(
        signal.name == "normalize_multiselect"
        for case in cases
        for signal in [*case.required, *case.forbidden]
    )
    assert any(
        signal.name == "conditional_dependency_inconsistency"
        for case in cases
        for signal in [*case.required, *case.forbidden]
    )
    assert any(
        signal.name == "normalization_equivalent_rows"
        for case in cases
        for signal in [*case.required, *case.forbidden]
    )
    assert any(
        signal.kind == "codebook"
        for case in cases
        for signal in [*case.required, *case.forbidden]
    )
    assert any(
        signal.name == "other_specify_inconsistency"
        for case in cases
        for signal in [*case.required, *case.forbidden]
    )


def test_recall_benchmark_cli_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "recall_benchmark.json"

    result = CliRunner().invoke(
        app,
        ["recall-benchmark", "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["pass_rate"] == 1.0
    assert payload["forbidden_hits"] == 0
    assert "No Ollama calls" in result.output
