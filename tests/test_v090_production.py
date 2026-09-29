from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from data_cleaning_assistant.actions import ActionDecision, ActionType, CleaningPlan
from data_cleaning_assistant.ai_models import AIPlannerResponse
from data_cleaning_assistant.ai_planner import propose_ai_plan
from data_cleaning_assistant.ai_provider import AIProviderError, OllamaProvider
from data_cleaning_assistant.cli import app
from data_cleaning_assistant.io import load_dataset
from data_cleaning_assistant.inference import parse_date
from data_cleaning_assistant.planner import propose_actions
from data_cleaning_assistant.transforms import apply_plan


class AcceptAllSemanticProvider:
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def generate_structured(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_model,
    ):
        self.calls += 1
        assert model == "fake-model"
        assert system_prompt
        context = json.loads(user_prompt.split("\n\n", 1)[1])
        return response_model.model_validate(
            {
                "summary": "Accept semantic fixture candidates.",
                "selections": [
                    {
                        "candidate_id": candidate["id"],
                        "decision": "accept",
                        "rationale": "Fixture semantic decision.",
                        "confidence": 0.9,
                    }
                    for candidate in context["candidates"]
                ],
            }
        )


class NeverCalledProvider:
    name = "fake"

    def generate_structured(self, **kwargs):
        raise AssertionError("Mechanical-only plans must not call the model.")


def test_numeric_date_parser_fails_closed_on_ambiguous_day_month_order() -> None:
    assert parse_date("06/08/2025") is None
    assert parse_date("02-10-2025") is None
    assert parse_date("10/11/2025") is None

    assert parse_date("29/09/2025").date().isoformat() == "2025-09-29"
    assert parse_date("09/29/2025").date().isoformat() == "2025-09-29"
    assert parse_date("2025/08/06").date().isoformat() == "2025-08-06"


def test_default_ai_routing_skips_mechanical_cleanup(tmp_path: Path) -> None:
    source = tmp_path / "mechanical.csv"
    source.write_text(
        "subject_id,value,label\n"
        "S001,NA, Alpha \n"
        "S002,12, Alpha \n"
        "S003,13, Alpha \n",
        encoding="utf-8",
    )

    result = propose_ai_plan(
        source,
        provider=NeverCalledProvider(),
        model="fake-model",
    )

    assert result.candidates == []
    assert result.review_call_count == 0
    assert result.plan.deferred_reviews == []
    action_types = {action.type for action in result.plan.actions}
    assert ActionType.NORMALIZE_MISSING in action_types
    assert ActionType.TRIM_WHITESPACE in action_types


def test_sparse_free_text_does_not_become_category_normalization(
    tmp_path: Path,
) -> None:
    source = tmp_path / "comments.csv"
    rows = [
        "S001,NA",
        "S002,NA",
        "S003,NA",
        "S004,NA",
        "S005,No",
        "S006,no",
        "S007,first detailed comment",
        "S008,second detailed comment",
        "S009,third detailed comment",
        "S010,fourth detailed comment",
        "S011,fifth detailed comment",
        "S012,sixth detailed comment",
        "S013,seventh detailed comment",
        "S014,eighth detailed comment",
        "S015,ninth detailed comment",
        "S016,tenth detailed comment",
    ]
    source.write_text(
        "subject_id,comments\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )

    result = propose_ai_plan(
        source,
        provider=NeverCalledProvider(),
        model="fake-model",
    )

    assert result.candidates == []
    assert not any(
        action.type == ActionType.NORMALIZE_CATEGORIES
        and action.column == "comments"
        for action in result.plan.actions
    )


def test_default_ai_routing_keeps_semantic_category_review(tmp_path: Path) -> None:
    source = tmp_path / "semantic.csv"
    source.write_text(
        "subject_id,specimen_source\n"
        "S001,Peripheral Blood\n"
        "S002,Peripheral Blood\n"
        "S003,Peripheral Blod\n"
        "S004,Peripheral Blood\n",
        encoding="utf-8",
    )
    provider = AcceptAllSemanticProvider()

    result = propose_ai_plan(
        source,
        provider=provider,
        model="fake-model",
        include_rules=False,
    )

    assert provider.calls == 2
    assert len(result.candidates) == 1
    assert result.candidates[0].action.type == ActionType.NORMALIZE_CATEGORIES
    assert len(result.plan.actions) == 1


def test_grouped_review_records_bulk_provenance_and_audit(tmp_path: Path) -> None:
    source = tmp_path / "study.csv"
    source.write_text(
        "subject_id,a,b\n"
        "S001,NA,NA\n"
        "S002,1,2\n"
        "S003,3,4\n",
        encoding="utf-8",
    )
    original = source.read_bytes()
    plan = propose_actions(source)
    missing = [
        action
        for action in plan.actions
        if action.type == ActionType.NORMALIZE_MISSING
    ]
    assert len(missing) == 2

    plan_path = tmp_path / "plan.json"
    reviewed_path = tmp_path / "reviewed.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "review-grouped",
            str(source),
            str(plan_path),
            "--output",
            str(reviewed_path),
        ],
        input="a\nYES\n",
    )
    assert result.exit_code == 0, result.output
    assert source.read_bytes() == original

    reviewed = CleaningPlan.model_validate_json(
        reviewed_path.read_text(encoding="utf-8")
    )
    assert all(
        action.decision == ActionDecision.APPROVED
        for action in reviewed.actions
    )
    assert reviewed.review_session is not None
    assert reviewed.review_session.completed_at is not None
    assert reviewed.review_session.duration_seconds is not None
    assert reviewed.review_session.duration_seconds >= 0
    assert len(reviewed.review_session.events) == 1
    event = reviewed.review_session.events[0]
    assert event.scope == "family"
    assert event.decision == "approved"
    assert event.member_count == 2
    assert event.bulk_confirmed is True
    assert event.affected_count == 2

    loaded = load_dataset(source)
    _cleaned, changeset = apply_plan(source, loaded.tables, reviewed)
    assert changeset.review_session is not None
    assert changeset.review_session.events[0].bulk_confirmed is True


def test_grouped_review_can_drill_down_individually(tmp_path: Path) -> None:
    source = tmp_path / "study.csv"
    source.write_text(
        "subject_id,a,b\n"
        "S001,NA,NA\n"
        "S002,1,2\n"
        "S003,3,4\n",
        encoding="utf-8",
    )
    plan = propose_actions(source)
    plan_path = tmp_path / "plan.json"
    reviewed_path = tmp_path / "reviewed.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "review-grouped",
            str(source),
            str(plan_path),
            "--output",
            str(reviewed_path),
        ],
        input="i\na\nr\n",
    )
    assert result.exit_code == 0, result.output

    reviewed = CleaningPlan.model_validate_json(
        reviewed_path.read_text(encoding="utf-8")
    )
    decisions = {action.decision for action in reviewed.actions}
    assert decisions == {ActionDecision.APPROVED, ActionDecision.REJECTED}
    assert reviewed.review_session is not None
    scopes = [event.scope for event in reviewed.review_session.events]
    assert scopes == ["family", "individual", "individual"]
    assert reviewed.review_session.events[0].decision == "inspect_individually"


class MalformedOllama(OllamaProvider):
    def __init__(self) -> None:
        super().__init__("http://example.invalid")

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        assert method == "POST"
        assert path == "/api/chat"
        return {"message": {"content": "{ definitely not valid json"}}


def test_real_ollama_provider_wraps_malformed_structured_output() -> None:
    provider = MalformedOllama()
    with pytest.raises(AIProviderError, match="failed local schema validation"):
        provider.generate_structured(
            model="fake-model",
            system_prompt="system",
            user_prompt="user",
            response_model=AIPlannerResponse,
        )
