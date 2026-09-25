"""Offline checks for the fresh single-edit semantic-neighbor holdout."""

from __future__ import annotations

from pathlib import Path

from data_cleaning_assistant.semantic_holdout import semantic_neighbor_holdout_cases
from data_cleaning_assistant.semantic_stress import semantic_cases


def test_neighbor_holdout_is_balanced_fresh_and_frequency_varied() -> None:
    cases = semantic_neighbor_holdout_cases()
    assert len(cases) == 40
    assert len({case.id for case in cases}) == 40
    assert sum(case.expected == "accept" for case in cases) == 20
    assert sum(case.expected == "defer" for case in cases) == 20
    assert not ({case.id for case in cases} & {case.id for case in semantic_cases()})
    assert len({(case.canonical_count, case.variant_count) for case in cases}) >= 4


def test_neighbor_holdout_all_crosses_single_edit_candidate_surface(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from semantic_stress_test import run_suite

    output = tmp_path / "holdout_coverage.json"
    result = run_suite(
        None,
        None,
        runs=1,
        audit_only=True,
        output=output,
        cases=semantic_neighbor_holdout_cases(),
        prompt_variant="baseline",
    )

    assert result["unique_cases"] == 40
    assert result["safe_cases"] == 20
    assert result["unsafe_cases"] == 20
    assert result["total_case_runs"] == 40
    assert result["exposed_case_runs"] == 40
    assert result["not_exposed_case_runs"] == 0
    assert result["safe_coverage"] == 20
    assert result["unsafe_exposure"] == 20
    assert result["model_calls"] == 0
    assert result["coverage_reasons"] == {}
    assert all(
        row.get("evidence_basis") == "single_edit_typo_pattern"
        for row in result["results"]
    )
