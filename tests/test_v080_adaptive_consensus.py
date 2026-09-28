from __future__ import annotations

import json
from pathlib import Path

from data_cleaning_assistant.actions import ActionType
from data_cleaning_assistant.ai_candidates import build_ai_candidates
from data_cleaning_assistant.ai_planner import propose_ai_plan
from data_cleaning_assistant.ai_provider import AIProviderError
from data_cleaning_assistant.engine import profile_dataset
from data_cleaning_assistant.io import load_dataset


def _write_typo_dataset(path: Path) -> None:
    path.write_text(
        "subject_id,label\n"
        "S001,Peripheral Blood\n"
        "S002,Peripheral Blood\n"
        "S003,Peripheral Blod\n"
        "S004,Peripheral Blood\n",
        encoding="utf-8",
    )


def _candidate_id(path: Path) -> str:
    loaded = load_dataset(path)
    candidates = build_ai_candidates(profile_dataset(path), loaded.tables)
    candidate = next(
        item for item in candidates
        if item.action.type == ActionType.NORMALIZE_CATEGORIES
        and item.action.column == "label"
    )
    return candidate.id


def _payload(candidate_id: str, decision: str, rationale: str = "fixture") -> dict:
    return {
        "summary": "fixture review",
        "selections": [
            {
                "candidate_id": candidate_id,
                "decision": decision,
                "rationale": rationale,
                "confidence": 0.9,
            }
        ],
    }


class SequenceProvider:
    name = "fake"

    def __init__(self, payloads: list[dict | Exception]) -> None:
        self.payloads = list(payloads)
        self.calls = 0
        self.system_prompts: list[str] = []

    def generate_structured(
        self, *, model, system_prompt, user_prompt, response_model
    ):
        assert model == "fake-model"
        assert "candidates" in user_prompt
        self.calls += 1
        self.system_prompts.append(system_prompt)
        item = self.payloads.pop(0)
        if isinstance(item, Exception):
            raise item
        return response_model.model_validate(item)


def test_two_agreeing_accepts_require_only_two_reviews(tmp_path: Path) -> None:
    source = tmp_path / "study.csv"
    _write_typo_dataset(source)
    candidate_id = _candidate_id(source)
    provider = SequenceProvider(
        [
            _payload(candidate_id, "accept", "clear typo"),
            _payload(candidate_id, "accept", "clear typo again"),
        ]
    )

    result = propose_ai_plan(
        source,
        provider=provider,
        model="fake-model",
        include_rules=False,
    )

    assert provider.calls == 2
    assert result.review_call_count == 2
    assert result.consensus_disagreement_count == 0
    assert len(result.plan.actions) == 1
    assert result.plan.deferred_reviews == []
    assert result.response.selections[0].decision.value == "accept"
    assert result.response.selections[0].rationale.startswith(
        "Adaptive consensus 2/2 accept"
    )
    assert all("evidence-proportional judgment" in prompt for prompt in provider.system_prompts)


def test_disagreement_uses_third_review_and_majority_accepts(tmp_path: Path) -> None:
    source = tmp_path / "study.csv"
    _write_typo_dataset(source)
    candidate_id = _candidate_id(source)
    provider = SequenceProvider(
        [
            _payload(candidate_id, "defer", "uncertain"),
            _payload(candidate_id, "accept", "clear typo"),
            _payload(candidate_id, "accept", "clear typo"),
        ]
    )

    result = propose_ai_plan(
        source,
        provider=provider,
        model="fake-model",
        include_rules=False,
    )

    assert provider.calls == 3
    assert result.review_call_count == 3
    assert result.consensus_disagreement_count == 1
    assert len(result.plan.actions) == 1
    assert result.response.selections[0].decision.value == "accept"
    assert result.response.selections[0].rationale.startswith(
        "Adaptive consensus 2/3 accept"
    )


def test_disagreement_uses_third_review_and_majority_defers(tmp_path: Path) -> None:
    source = tmp_path / "study.csv"
    _write_typo_dataset(source)
    candidate_id = _candidate_id(source)
    provider = SequenceProvider(
        [
            _payload(candidate_id, "accept", "looks like typo"),
            _payload(candidate_id, "defer", "semantic uncertainty"),
            _payload(candidate_id, "defer", "semantic uncertainty"),
        ]
    )

    result = propose_ai_plan(
        source,
        provider=provider,
        model="fake-model",
        include_rules=False,
    )

    assert provider.calls == 3
    assert result.plan.actions == []
    assert len(result.plan.deferred_reviews) == 1
    assert result.plan.deferred_reviews[0].rationale.startswith(
        "Adaptive consensus 2/3 defer"
    )


def test_review_error_fails_closed_to_defer(tmp_path: Path) -> None:
    source = tmp_path / "study.csv"
    _write_typo_dataset(source)
    candidate_id = _candidate_id(source)
    provider = SequenceProvider(
        [
            _payload(candidate_id, "accept", "clear typo"),
            AIProviderError("fixture structured-output failure"),
        ]
    )

    result = propose_ai_plan(
        source,
        provider=provider,
        model="fake-model",
        include_rules=False,
    )

    assert provider.calls == 2
    assert result.plan.actions == []
    assert len(result.plan.deferred_reviews) == 1
    assert result.plan.deferred_reviews[0].confidence == 0.0
    assert "failed closed" in result.plan.deferred_reviews[0].rationale
    assert "model calls failed" in result.response.summary


def test_unknown_candidate_vote_cannot_create_action(tmp_path: Path) -> None:
    source = tmp_path / "study.csv"
    _write_typo_dataset(source)
    candidate_id = _candidate_id(source)
    provider = SequenceProvider(
        [
            _payload("C999", "accept", "invented candidate"),
            _payload(candidate_id, "accept", "clear typo"),
        ]
    )

    result = propose_ai_plan(
        source,
        provider=provider,
        model="fake-model",
        include_rules=False,
    )

    assert result.plan.actions == []
    assert len(result.plan.deferred_reviews) == 1
    assert result.rejected
    assert "Review 1 referenced an unknown candidate ID" in result.rejected[0].reason


class AcceptAllCandidatesProvider:
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def generate_structured(
        self, *, model, system_prompt, user_prompt, response_model
    ):
        assert model == "fake-model"
        self.calls += 1
        payload = json.loads(user_prompt.split("\n\n", 1)[1])
        selections = [
            {
                "candidate_id": candidate["id"],
                "decision": "accept",
                "rationale": "fixture accept",
                "confidence": 0.9,
            }
            for candidate in payload["candidates"]
        ]
        return response_model.model_validate(
            {"summary": "accept all fixture candidates", "selections": selections}
        )


def test_hybrid_plan_suppresses_ai_category_mapping_already_covered_by_rule(
    tmp_path: Path,
) -> None:
    source = tmp_path / "study.csv"
    source.write_text(
        "subject_id,sex\n"
        "S001,Female\n"
        "S002,female\n"
        "S003,Male\n"
        "S004,MALE\n",
        encoding="utf-8",
    )
    provider = AcceptAllCandidatesProvider()

    result = propose_ai_plan(
        source,
        provider=provider,
        model="fake-model",
        include_rules=True,
    )

    category_actions = [
        action
        for action in result.plan.actions
        if action.type == ActionType.NORMALIZE_CATEGORIES
        and action.column == "sex"
    ]
    assert len(category_actions) == 1
    assert category_actions[0].origin.value == "rule"
    assert category_actions[0].parameters == {
        "mapping": {"female": "Female", "MALE": "Male"}
    }
    assert provider.calls == 2
