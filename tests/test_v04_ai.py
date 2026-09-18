from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from data_cleaning_assistant.actions import ActionOrigin, ActionType
from data_cleaning_assistant.ai_candidates import build_ai_candidates
from data_cleaning_assistant.ai_context import build_ai_context
from data_cleaning_assistant.ai_models import AIPlannerResponse
from data_cleaning_assistant.ai_planner import propose_ai_plan, resolve_model
from data_cleaning_assistant.ai_provider import OllamaProvider
from data_cleaning_assistant.cli import app
from data_cleaning_assistant.engine import profile_dataset
from data_cleaning_assistant.io import load_dataset
from data_cleaning_assistant.planner import propose_actions
from data_cleaning_assistant.transforms import apply_plan


class FakeProvider:
    name = "fake"

    def __init__(self, payload: dict, models: list[str] | None = None) -> None:
        self.payload = payload
        self.models = models if models is not None else ["fake-model"]
        self.last_user_prompt = ""

    def list_models(self) -> list[str]:
        return self.models

    def generate_structured(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_model,
    ):
        assert system_prompt
        assert model in self.models
        self.last_user_prompt = user_prompt
        return response_model.model_validate(self.payload)


def _write_fuzzy_dataset(path: Path) -> None:
    path.write_text(
        "patient_id,group\n"
        "P001,Baseline\n"
        "P002,Basline\n"
        "P003,Followup\n"
        "P004,Baseline\n",
        encoding="utf-8",
    )


def _candidate_for(source: Path, action_type: ActionType, column: str):
    loaded = load_dataset(source)
    report = profile_dataset(source)
    candidates = build_ai_candidates(report, loaded.tables)
    return next(
        candidate
        for candidate in candidates
        if candidate.action.type == action_type and candidate.action.column == column
    )


def test_ai_context_excludes_path_samples_and_identifier_values(tmp_path: Path) -> None:
    source = tmp_path / "secret_study.csv"
    source.write_text(
        "patient_id,value\n"
        "987654321,10\n"
        "987654322,20\n"
        "987654323,30\n",
        encoding="utf-8",
    )

    loaded = load_dataset(source)
    report = profile_dataset(source)
    candidates = build_ai_candidates(report, loaded.tables)
    context = build_ai_context(report, candidates)
    payload = context.model_dump_json()

    assert "secret_study.csv" not in payload
    assert "987654321" not in payload
    patient = next(
        column
        for table in context.tables
        for column in table.columns
        if column.name == "patient_id"
    )
    assert patient.inferred_type == "identifier"
    assert patient.numeric_summary is None


def test_candidate_builder_creates_bounded_fuzzy_merge(tmp_path: Path) -> None:
    source = tmp_path / "study.csv"
    _write_fuzzy_dataset(source)

    candidate = _candidate_for(source, ActionType.NORMALIZE_CATEGORIES, "group")

    assert candidate.id.startswith("C")
    assert candidate.action.parameters == {"mapping": {"Basline": "Baseline"}}
    assert candidate.context.table == "study"
    assert candidate.context.column == "group"


def test_ai_selects_candidate_without_authoring_table_column_or_parameters(
    tmp_path: Path,
) -> None:
    source = tmp_path / "study.csv"
    _write_fuzzy_dataset(source)
    candidate = _candidate_for(source, ActionType.NORMALIZE_CATEGORIES, "group")

    provider = FakeProvider(
        {
            "summary": "The one-character typo is safe to normalize.",
            "selections": [
                {
                    "candidate_id": candidate.id,
                    "decision": "accept",
                    "rationale": "Basline is clearly a typo of the more frequent Baseline label.",
                    "confidence": 0.95,
                }
            ],
        }
    )

    result = propose_ai_plan(
        source,
        provider=provider,
        model="fake-model",
        include_rules=False,
    )

    assert len(result.plan.actions) == 1
    action = result.plan.actions[0]
    assert action.origin == ActionOrigin.AI
    assert action.table == "study"
    assert action.column == "group"
    assert action.parameters == {"mapping": {"Basline": "Baseline"}}
    assert not result.rejected

    loaded = load_dataset(source)
    cleaned, changeset = apply_plan(
        source,
        loaded.tables,
        result.plan,
        approved_ids={action.id},
    )
    assert cleaned["study"]["group"].tolist() == [
        "Baseline",
        "Baseline",
        "Followup",
        "Baseline",
    ]
    assert changeset.affected_cell_count == 1


def test_fuzzy_semantic_difference_can_be_deferred(tmp_path: Path) -> None:
    source = tmp_path / "stage.csv"
    source.write_text(
        "patient_id,stage\n"
        "P001,Stage II\n"
        "P002,Stage III\n"
        "P003,Stage II\n"
        "P004,Stage III\n",
        encoding="utf-8",
    )
    candidate = _candidate_for(source, ActionType.NORMALIZE_CATEGORIES, "stage")

    provider = FakeProvider(
        {
            "summary": "The stages are distinct.",
            "selections": [
                {
                    "candidate_id": candidate.id,
                    "decision": "defer",
                    "rationale": "Stage II and Stage III are scientifically distinct.",
                    "confidence": 0.99,
                }
            ],
        }
    )

    result = propose_ai_plan(
        source,
        provider=provider,
        model="fake-model",
        include_rules=False,
    )

    assert result.plan.actions == []
    assert not result.rejected


def test_ai_cannot_create_row_deletion_candidate(tmp_path: Path) -> None:
    source = tmp_path / "duplicates.csv"
    source.write_text(
        "patient_id,status\n"
        "P001,Active\n"
        "P001,Active\n"
        "P002,Inactive\n",
        encoding="utf-8",
    )
    loaded = load_dataset(source)
    report = profile_dataset(source)
    candidates = build_ai_candidates(report, loaded.tables)

    assert all(
        candidate.action.type != ActionType.REMOVE_EXACT_DUPLICATES
        for candidate in candidates
    )


def test_unknown_candidate_id_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "study.csv"
    _write_fuzzy_dataset(source)
    provider = FakeProvider(
        {
            "summary": "Invent an ID.",
            "selections": [
                {
                    "candidate_id": "C999",
                    "decision": "accept",
                    "rationale": "Unknown candidate.",
                    "confidence": 1.0,
                }
            ],
        }
    )

    result = propose_ai_plan(
        source,
        provider=provider,
        model="fake-model",
        include_rules=False,
    )

    assert result.plan.actions == []
    assert len(result.rejected) == 1
    assert "unknown candidate ID" in result.rejected[0].reason


def test_model_resolution_requires_choice_when_multiple_are_installed() -> None:
    provider = FakeProvider(
        {"summary": "", "selections": []},
        models=["model-a", "model-b"],
    )
    with pytest.raises(ValueError, match="Multiple local models"):
        resolve_model(provider, None)
    assert resolve_model(provider, "model-b") == "model-b"


class StubOllama(OllamaProvider):
    def __init__(self) -> None:
        super().__init__("http://example.invalid")
        self.sent: dict | None = None

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        assert method == "POST"
        assert path == "/api/chat"
        self.sent = payload
        return {
            "message": {
                "content": json.dumps(
                    {
                        "summary": "No candidates accepted.",
                        "selections": [],
                    }
                )
            }
        }


def test_ollama_provider_sends_json_schema_and_disables_streaming_and_thinking() -> None:
    provider = StubOllama()
    response = provider.generate_structured(
        model="test-model",
        system_prompt="system",
        user_prompt="user",
        response_model=AIPlannerResponse,
    )

    assert response.summary == "No candidates accepted."
    assert provider.sent is not None
    assert provider.sent["stream"] is False
    assert provider.sent["think"] is False
    assert provider.sent["format"]["type"] == "object"
    assert provider.sent["options"]["temperature"] == 0.1


def test_ai_plan_cli_writes_candidate_selected_plan(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "study.csv"
    _write_fuzzy_dataset(source)
    candidate = _candidate_for(source, ActionType.NORMALIZE_CATEGORIES, "group")
    output = tmp_path / "ai_plan.json"
    context_out = tmp_path / "ai_context.json"

    provider = FakeProvider(
        {
            "summary": "Merge the typo pair.",
            "selections": [
                {
                    "candidate_id": candidate.id,
                    "decision": "accept",
                    "rationale": "Evidence-supported typo correction.",
                    "confidence": 0.9,
                }
            ],
        }
    )
    monkeypatch.setattr(
        "data_cleaning_assistant.cli.OllamaProvider",
        lambda url=None, think=False: provider,
    )

    result = CliRunner().invoke(
        app,
        [
            "ai-plan",
            str(source),
            "--model",
            "fake-model",
            "--output",
            str(output),
            "--context-out",
            str(context_out),
            "--no-rules",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert output.exists()
    assert context_out.exists()
    assert '"origin": "ai"' in output.read_text(encoding="utf-8")
    assert '"id": "C001"' in context_out.read_text(encoding="utf-8")


def test_no_candidate_fast_path_does_not_call_provider(tmp_path: Path) -> None:
    source = tmp_path / "duplicates.csv"
    source.write_text(
        "patient_id,status\n"
        "P001,Active\n"
        "P001,Active\n"
        "P002,Inactive\n",
        encoding="utf-8",
    )

    class NeverCalledProvider:
        name = "fake"

        def list_models(self) -> list[str]:
            return ["fake-model"]

        def generate_structured(self, **kwargs):
            raise AssertionError("Provider should not be called when there are no AI candidates.")

    result = propose_ai_plan(
        source,
        provider=NeverCalledProvider(),
        model="fake-model",
        include_rules=False,
    )

    assert result.candidates == []
    assert result.plan.actions == []
    assert result.response.selections == []
    assert "No AI-review candidates" in result.response.summary
    assert result.rejected == []


def test_zero_padding_candidate_has_deterministic_equivalence_evidence(tmp_path: Path) -> None:
    source = tmp_path / "visits.csv"
    source.write_text(
        "subject_id,visit_label\n"
        "S001,Visit 1\n"
        "S002,Visit 1\n"
        "S003,Visit 01\n"
        "S004,Visit 1\n",
        encoding="utf-8",
    )

    candidate = _candidate_for(source, ActionType.NORMALIZE_CATEGORIES, "visit_label")

    assert candidate.action.parameters == {"mapping": {"Visit 01": "Visit 1"}}
    assert candidate.context.evidence["basis"] == "numeric_zero_padding_equivalence"
    assert candidate.context.evidence["numeric_zero_padding_equivalent"] is True


def test_contextual_missingness_is_flagged_but_not_actionable_by_default(
    tmp_path: Path,
) -> None:
    source = tmp_path / "outcomes.csv"
    source.write_text(
        "subject_id,outcome_category\n"
        "S001,Response\n"
        "S002,Unknown\n"
        "S003,Progression\n"
        "S004,Response\n",
        encoding="utf-8",
    )

    report = profile_dataset(source)
    issue = next(
        item
        for item in report.issues
        if item.type == "hidden_missing_values" and item.column == "outcome_category"
    )
    assert issue.evidence["contextual_tokens"] == {"Unknown": 1}
    assert issue.evidence["structural_tokens"] == {}

    loaded = load_dataset(source)
    candidates = build_ai_candidates(report, loaded.tables)
    assert not any(
        candidate.action.type == ActionType.NORMALIZE_MISSING
        and candidate.action.column == "outcome_category"
        for candidate in candidates
    )

    rule_plan = propose_actions(source)
    assert not any(
        action.type == ActionType.NORMALIZE_MISSING
        and action.column == "outcome_category"
        for action in rule_plan.actions
    )
