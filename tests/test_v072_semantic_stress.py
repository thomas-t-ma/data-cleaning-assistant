"""Independent semantic stress tests with a fake, offline model provider."""

from __future__ import annotations

import json
from pathlib import Path


from data_cleaning_assistant.ai_models import (
    AICandidateSelection,
    AIPlannerResponse,
    CandidateDecision,
)
from data_cleaning_assistant.semantic_stress import semantic_cases


class FakeModel:
    name = "offline_fake"

    def __init__(self, behavior: str = "right") -> None:
        self.behavior = behavior
        self.prompts: list[str] = []

    def generate_structured(
        self, *, model, system_prompt, user_prompt, response_model
    ) -> AIPlannerResponse:
        assert model == "fixture:model"
        assert response_model is AIPlannerResponse
        assert "accept only" in system_prompt
        self.prompts.append(user_prompt)
        context = json.loads(user_prompt.split("\n\n", 1)[1])
        assert len(context["tables"]) == 1
        assert len(context["candidates"]) == 1
        assert context["relationships"] == []
        assert "answer_key" not in user_prompt
        assert "expected_decision" not in user_prompt
        assert "case_id" not in user_prompt
        candidate = context["candidates"][0]
        mapping = candidate["proposed_change"]["mapping"]
        pair = set(mapping) | set(mapping.values())
        if self.behavior == "all_accept":
            decision = CandidateDecision.ACCEPT
        elif self.behavior == "omit":
            return AIPlannerResponse(summary="omitted", selections=[])
        else:
            decision = (
                CandidateDecision.ACCEPT
                if pair == {"Peripheral Blod", "Peripheral Blood"}
                else CandidateDecision.DEFER
            )
        return AIPlannerResponse(
            summary="Fake offline judgment",
            selections=[
                AICandidateSelection(
                    candidate_id=candidate["id"],
                    decision=decision,
                    confidence=0.8,
                    rationale="Synthetic-only fixture model.",
                )
            ],
        )


def _select(*ids: str):
    cases = {case.id: case for case in semantic_cases()}
    return [cases[id] for id in ids]


def test_corpus_size_balance_and_contrasting_context() -> None:
    cases = semantic_cases()
    assert len(cases) == 40
    assert len({case.id for case in cases}) == 40
    assert sum(case.expected == "accept" for case in cases) == 20
    assert sum(case.expected == "defer" for case in cases) == 20
    safe = next(case for case in cases if case.id == "s16")
    risky = next(case for case in cases if case.id == "n18")
    assert (safe.canonical, safe.variant) == (risky.canonical, risky.variant)
    assert safe.context != risky.context
    assert safe.expected != risky.expected


def test_independent_calls_score_and_source_free(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from semantic_stress_test import run_suite

    provider = FakeModel()
    path = tmp_path / "score.json"
    result = run_suite(
        provider,
        "fixture:model",
        runs=2,
        output=path,
        cases=_select("s01", "n01"),
    )
    assert result["unique_cases"] == 2
    assert result["total_case_runs"] == 4
    assert result["exposed_case_runs"] == 4
    assert result["correct_model_decisions"] == 4
    assert result["unsafe_merges_accepted"] == 0
    assert result["missed_safe_corrections"] == 0
    assert result["strict_suite_pass"] is True
    assert len(provider.prompts) == 4
    assert result["mean_call_latency_seconds"] is not None
    assert path.exists()
    assert result["corpus_sha256"] == json.loads(path.read_text())["corpus_sha256"]


def test_unsafe_merges_report_separately(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from semantic_stress_test import run_suite

    result = run_suite(
        FakeModel("all_accept"),
        "fixture:model",
        runs=1,
        output=tmp_path / "wrong.json",
        cases=_select("s01", "n01"),
    )
    assert result["correct_model_decisions"] == 1
    assert result["unsafe_merges_accepted"] == 1
    assert result["missed_safe_corrections"] == 0
    assert not result["strict_suite_pass"]


def test_missing_selection_not_counted_as_defer(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from semantic_stress_test import run_suite

    result = run_suite(
        FakeModel("omit"),
        "fixture:model",
        runs=1,
        output=tmp_path / "omit.json",
        cases=_select("s01", "n01"),
    )
    assert result["correct_model_decisions"] == 0
    assert result["protocol_failures"] == 2
    assert all(item["actual"] == "missing" for item in result["results"])


def test_coverage_gaps_not_silently_credited(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from semantic_stress_test import run_suite

    result = run_suite(
        None,
        None,
        runs=1,
        audit_only=True,
        output=tmp_path / "coverage.json",
        cases=_select("s01", "n04", "n05"),
    )
    assert result["total_case_runs"] == 3
    assert result["safe_total"] == 1
    assert result["model_calls"] == 0
    assert result["correct_model_decisions"] == 0
    assert result["not_exposed_case_runs"] >= 1
    assert not result["strict_suite_pass"]
    assert all(
        item["actual"] in {"not_evaluated", "not_exposed"}
        for item in result["results"]
    )


def test_context_same_pair_can_have_opposite_expected_decisions(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from semantic_stress_test import _prepare_case

    cases = _select("s16", "n18")
    prepared = []
    for case in cases:
        folder = tmp_path / case.id
        folder.mkdir()
        prepared.append(_prepare_case(case, folder))
    assert all(item["candidate"] is not None for item in prepared)
    assert prepared[0]["prompt"] != prepared[1]["prompt"]
    assert all("Fixture dictionary:" in item["prompt"] for item in prepared)
    assert all("expected_decision" not in item["prompt"] for item in prepared)


def test_duplicate_unknown_ids_fail_protocol(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from semantic_stress_test import _prepare_case, _score_reply

    case = _select("s01")[0]
    prepared = _prepare_case(case, tmp_path)
    candidate = prepared["candidate"]
    good = AICandidateSelection(
        candidate_id=candidate.id,
        decision=CandidateDecision.ACCEPT,
        rationale="fixture",
        confidence=0.8,
    )
    extra = AICandidateSelection(
        candidate_id="invented",
        decision=CandidateDecision.DEFER,
        rationale="fixture",
        confidence=0.1,
    )
    scored = _score_reply(
        AIPlannerResponse(summary="fixture", selections=[good, good, extra]),
        candidate.id,
        "accept",
    )
    assert not scored["protocol_ok"]
    assert not scored["correct"]
    assert scored["duplicate_candidate_id"]
    assert scored["unknown_candidate_ids"] == ["invented"]


def test_never_overwrite_prior_score(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from semantic_stress_test import run_suite

    output = tmp_path / "score.json"
    run_suite(
        None, None, runs=1, output=output,
        audit_only=True, cases=_select("s01"),
    )
    prior = output.read_bytes()
    try:
        run_suite(
            None, None, runs=1, output=output,
            audit_only=True, cases=_select("s01"),
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("Existing experiment was overwritten")
    assert output.read_bytes() == prior
