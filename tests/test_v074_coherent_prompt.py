"""Benchmark-only coherent policy regression checks (no Ollama required)."""

from __future__ import annotations

import json
from pathlib import Path

from data_cleaning_assistant.semantic_stress import semantic_cases


def test_coherent_prompt_replaces_conflicting_baseline(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from semantic_stress_test import _system_prompt, run_suite

    baseline = _system_prompt("baseline")
    appended = _system_prompt("evidence_calibrated")
    coherent = _system_prompt("coherent_policy")

    assert appended.startswith(baseline)
    assert coherent != baseline
    assert not coherent.startswith(baseline)
    assert "if the differing token could encode a real scientific distinction" in baseline
    assert "if the differing token could encode a real scientific distinction" not in coherent
    assert "A merely imaginable alternative meaning is not evidence" in coherent
    assert "Explicit study-schema distinctions" in coherent
    assert "Return exactly one accept/defer selection" in coherent
    assert "expected_decision" not in coherent
    assert "answer_key" not in coherent

    output = tmp_path / "coherent_audit.json"
    result = run_suite(
        None,
        None,
        runs=1,
        audit_only=True,
        output=output,
        prompt_variant="coherent_policy",
        cases=[next(c for c in semantic_cases() if c.id == "s01")],
    )
    assert result["prompt_variant"] == "coherent_policy"
    assert result["system_prompt_sha256"] != ""
    assert json.loads(output.read_text())["prompt_variant"] == "coherent_policy"


def test_production_prompt_remains_unchanged(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from data_cleaning_assistant.ai_planner import SYSTEM_PROMPT
    from semantic_stress_test import _system_prompt

    assert _system_prompt("baseline") == SYSTEM_PROMPT
    assert _system_prompt("coherent_policy") != SYSTEM_PROMPT
