"""Offline regressions for the frozen 100-case validation suite."""

from __future__ import annotations

import json
from pathlib import Path

from data_cleaning_assistant.semantic_holdout import semantic_neighbor_holdout_cases
from data_cleaning_assistant.semantic_stress import semantic_cases
from data_cleaning_assistant.semantic_validation import frozen_validation_cases


def _pair(case) -> frozenset[str]:
    return frozenset((case.canonical, case.variant))


def test_frozen_validation_size_balance_groups_and_freshness() -> None:
    cases = frozen_validation_cases()
    assert len(cases) == 100
    assert len({case.id for case in cases}) == 100
    assert sum(case.expected == "accept" for case in cases) == 50
    assert sum(case.expected == "defer" for case in cases) == 50

    expected_groups = {
        "safe_single_edit_typo": 15,
        "safe_multi_edit_typo": 10,
        "safe_nonmedical_typo": 10,
        "safe_schema_formatting": 5,
        "safe_schema_padding": 10,
        "unsafe_valid_neighbor": 20,
        "unsafe_abbreviation": 10,
        "unsafe_numeric_identifier": 10,
        "unsafe_schema_padding": 5,
        "unsafe_status_semantics": 5,
    }
    actual = {
        group: sum(case.group == group for case in cases)
        for group in expected_groups
    }
    assert actual == expected_groups

    prior_pairs = {
        _pair(case)
        for case in (semantic_cases() + semantic_neighbor_holdout_cases())
    }
    assert not ({_pair(case) for case in cases} & prior_pairs)


def test_validation_safe_cases_reach_candidate_review(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from semantic_stress_test import _prepare_case

    safe_cases = [
        case for case in frozen_validation_cases()
        if case.expected == "accept"
    ]
    prepared = []
    for case in safe_cases:
        folder = tmp_path / case.id
        folder.mkdir()
        prepared.append(_prepare_case(case, folder))

    missing = [
        item["case"].id
        for item in prepared
        if item["candidate"] is None
    ]
    assert missing == []


def test_validation_prompts_hide_answer_key(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from semantic_stress_test import _prepare_case

    for case in frozen_validation_cases()[:5]:
        folder = tmp_path / case.id
        folder.mkdir()
        item = _prepare_case(case, folder)
        assert item["candidate"] is not None
        prompt = item["prompt"]
        assert case.id not in prompt
        assert "expected_decision" not in prompt
        assert "answer_key" not in prompt


def test_frozen_validation_coverage_audit_is_write_once(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from frozen_validation_test import run_validation

    output = tmp_path / "coverage.json"
    result = run_validation(
        None,
        None,
        runs=1,
        modes=("baseline", "evidence_calibrated", "blind_veto"),
        output=output,
        audit_only=True,
    )
    assert result["unique_cases"] == 100
    assert result["safe_cases"] == 50
    assert result["unsafe_cases"] == 50
    assert result["safe_exposed"] == 50
    assert output.exists()
    saved = json.loads(output.read_text())
    assert saved["corpus_sha256"] == result["corpus_sha256"]

    prior = output.read_bytes()
    try:
        run_validation(
            None,
            None,
            runs=1,
            modes=("baseline",),
            output=output,
            audit_only=True,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("Frozen validation report was overwritten")
    assert output.read_bytes() == prior
