"""Offline regression for the benchmark-only two-stage semantic veto."""

from __future__ import annotations

import json
from pathlib import Path

from data_cleaning_assistant.ai_models import (
    AICandidateSelection,
    AIPlannerResponse,
    CandidateDecision,
)
from data_cleaning_assistant.semantic_holdout import semantic_neighbor_holdout_cases


class FakeTwoStageProvider:
    name = "fake_two_stage"

    def generate_structured(
        self, *, model, system_prompt, user_prompt, response_model
    ):
        assert model == "fixture:model"
        payload = json.loads(user_prompt.split("\n\n", 1)[1])
        candidate_id = payload.get("candidate_id")
        if response_model is AIPlannerResponse:
            context = payload
            candidate = context["candidates"][0]
            return AIPlannerResponse(
                summary="Primary accepts bounded single-edit candidate.",
                selections=[
                    AICandidateSelection(
                        candidate_id=candidate["id"],
                        decision=CandidateDecision.ACCEPT,
                        rationale="fixture primary",
                        confidence=0.9,
                    )
                ],
            )

        mapping = payload["proposed_mapping"]
        pair = set(mapping) | set(mapping.values())
        verdict = (
            "distinct"
            if pair == {
                "Clinical Term Facial",
                "Clinical Term Fascial",
            }
            else "same_typo"
        )
        return response_model(
            candidate_id=candidate_id,
            verdict=verdict,
            rationale="fixture semantic collision audit",
            confidence=0.9,
        )


def _select(*ids: str):
    cases = {case.id: case for case in semantic_neighbor_holdout_cases()}
    return [cases[case_id] for case_id in ids]


def test_veto_blocks_valid_neighbor_but_preserves_clear_typo(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from semantic_veto_test import run_veto_suite

    result = run_veto_suite(
        FakeTwoStageProvider(),
        "fixture:model",
        runs=1,
        output=tmp_path / "veto.json",
        cases=_select("h01", "r20"),
    )

    assert result["primary_safe_accepts"] == 1
    assert result["final_safe_accepts"] == 1
    assert result["primary_unsafe_accepts"] == 1
    assert result["final_unsafe_accepts"] == 0
    assert result["correct_final_decisions"] == 2
    assert result["audits_called"] == 2
    assert result["protocol_failures"] == 0
    assert result["model_errors"] == 0
    assert result["audit_verdict_counts"] == {
        "same_typo": 1,
        "distinct": 1,
    }
