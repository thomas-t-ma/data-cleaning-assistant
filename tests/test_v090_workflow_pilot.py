from __future__ import annotations

import json
from pathlib import Path

import pytest
from data_cleaning_assistant.actions import ActionDecision
from data_cleaning_assistant.ai_models import CandidateDecision
from data_cleaning_assistant.planner import file_sha256


class PilotFakeProvider:
    name = "offline_fake"

    def __init__(self) -> None:
        self.calls = 0

    def generate_structured(
        self, *, model, system_prompt, user_prompt, response_model
    ):
        assert model == "fake-model"
        assert system_prompt
        self.calls += 1
        context = json.loads(user_prompt.split("\n\n", 1)[1])
        selections = []
        for candidate in context["candidates"]:
            decision = (
                CandidateDecision.ACCEPT
                if candidate["column"] == "specimen_source"
                else CandidateDecision.DEFER
            )
            selections.append(
                {
                    "candidate_id": candidate["id"],
                    "decision": decision.value,
                    "rationale": f"fixture {decision.value}",
                    "confidence": 0.9,
                }
            )
        return response_model.model_validate(
            {"summary": "offline pilot fixture", "selections": selections}
        )


def _source(path: Path) -> None:
    path.write_text(
        "subject_id,specimen_source,stage\n"
        "S001,Peripheral Blood,Stage II\n"
        "S002,Peripheral Blood,Stage III\n"
        "S003,Peripheral Blod,Stage II\n"
        "S004,Peripheral Blood,Stage III\n",
        encoding="utf-8",
    )


def test_workflow_pilot_start_and_score_human_disagreement(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from workflow_pilot import (
        CONTEXT,
        MANIFEST,
        PLAN,
        PROFILE,
        SCORE,
        TRACE,
        score_pilot,
        start_pilot,
    )

    source = tmp_path / "research.csv"
    _source(source)
    before = file_sha256(source)
    pilot = tmp_path / "pilot"

    provider = PilotFakeProvider()
    started = start_pilot(
        source,
        pilot,
        provider=provider,
        model="fake-model",
    )

    assert provider.calls == 2
    assert file_sha256(source) == before
    assert all((pilot / name).exists() for name in (MANIFEST, PROFILE, CONTEXT, PLAN, TRACE))
    assert started["trace"]["candidate_count"] == 2
    assert {row["decision"] for row in started["trace"]["candidates"]} == {
        "accept",
        "defer",
    }

    reviewed = started["plan"].model_copy(deep=True)
    accepted_candidate = next(
        row for row in started["trace"]["candidates"] if row["decision"] == "accept"
    )
    assert accepted_candidate["covered_by_plan_action_ids"]
    accepted_action_id = accepted_candidate["covered_by_plan_action_ids"][0]
    next(a for a in reviewed.actions if a.id == accepted_action_id).decision = (
        ActionDecision.REJECTED
    )

    deferred_candidate = next(
        row for row in started["trace"]["candidates"] if row["decision"] == "defer"
    )
    deferred = next(
        item
        for item in reviewed.deferred_reviews
        if item.candidate_id == deferred_candidate["candidate_id"]
    )
    deferred.status = "approved"
    approved_action = deferred.proposed_action.model_copy(deep=True)
    approved_action.decision = ActionDecision.APPROVED
    reviewed.actions.append(approved_action)

    reviewed_path = pilot / "reviewed_plan.json"
    reviewed_path.write_text(reviewed.model_dump_json(indent=2), encoding="utf-8")

    scored = score_pilot(
        pilot,
        reviewed_path,
        review_minutes=4.5,
        missed_issues=["example reviewer-observed issue"],
    )

    assert scored["resolved_candidate_reviews"] == 2
    assert scored["human_ai_agreement_count"] == 0
    assert scored["human_rejected_ai_accept_count"] == 1
    assert scored["human_approved_ai_defer_count"] == 1
    assert scored["review_minutes"] == 4.5
    assert scored["missed_issues_reported_by_reviewer"] == [
        "example reviewer-observed issue"
    ]
    assert (pilot / SCORE).exists()
    assert file_sha256(source) == before


def test_workflow_pilot_refuses_changed_source(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from workflow_pilot import score_pilot, start_pilot

    source = tmp_path / "research.csv"
    _source(source)
    pilot = tmp_path / "pilot"
    started = start_pilot(
        source,
        pilot,
        provider=PilotFakeProvider(),
        model="fake-model",
    )
    reviewed_path = pilot / "reviewed_plan.json"
    reviewed_path.write_text(
        started["plan"].model_dump_json(indent=2),
        encoding="utf-8",
    )

    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source is missing or changed"):
        score_pilot(pilot, reviewed_path)


def test_nhanes_fetcher_validates_expected_shape(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from fetch_nhanes_sample import validate_csv

    source = tmp_path / "nhanes.csv"
    header = [
        "ID", "SurveyYr", "Gender", "Age", "Race1", "BMI",
        *[f"extra_{index}" for index in range(70)],
    ]
    rows = [
        [str(index), "2009_10", "male", "34", "White", "22.1", *[""] * 70]
        for index in range(200)
    ]
    source.write_text(
        ",".join(header) + "\n"
        + "\n".join(",".join(row) for row in rows)
        + "\n",
        encoding="utf-8",
    )

    assert validate_csv(source) == {"rows": 200, "columns": 76}


def test_nhanes_fetcher_rejects_wrong_shape(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from fetch_nhanes_sample import validate_csv

    source = tmp_path / "wrong.csv"
    source.write_text(
        "ID,SurveyYr,Gender,Age,Race1,BMI\n"
        "1,2009_10,male,34,White,22.1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unexpected NHANES sample shape"):
        validate_csv(source)
