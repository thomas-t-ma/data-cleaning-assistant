from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from data_cleaning_assistant.actions import ActionOrigin, ActionType
from data_cleaning_assistant.ai_context import build_ai_context
from data_cleaning_assistant.ai_models import AIPlannerResponse
from data_cleaning_assistant.ai_planner import propose_ai_plan, resolve_model
from data_cleaning_assistant.ai_provider import OllamaProvider
from data_cleaning_assistant.cli import app
from data_cleaning_assistant.engine import profile_dataset
from data_cleaning_assistant.io import load_dataset
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


def test_ai_context_excludes_path_samples_and_identifier_values(tmp_path: Path) -> None:
    source = tmp_path / "secret_study.csv"
    source.write_text(
        "patient_id,value\n"
        "987654321,10\n"
        "987654322,20\n"
        "987654323,30\n",
        encoding="utf-8",
    )

    context = build_ai_context(profile_dataset(source))
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


def test_ai_fuzzy_category_merge_survives_validation_and_is_executable(
    tmp_path: Path,
) -> None:
    source = tmp_path / "study.csv"
    _write_fuzzy_dataset(source)
    original = source.read_bytes()

    provider = FakeProvider(
        {
            "summary": "Basline appears to be a typo of Baseline.",
            "actions": [
                {
                    "type": "normalize_categories",
                    "table": "study",
                    "column": "group",
                    "parameters": {"mapping": {"Basline": "Baseline"}},
                    "rationale": (
                        "The profiler identified this exact pair as highly similar, "
                        "and both labels occur in the same categorical field."
                    ),
                    "confidence": 0.91,
                    "issue_ids": [],
                }
            ],
            "deferred_issue_ids": [],
        }
    )

    result = propose_ai_plan(
        source,
        provider=provider,
        model="fake-model",
        include_rules=True,
    )

    assert source.read_bytes() == original
    ai_actions = [action for action in result.plan.actions if action.origin == ActionOrigin.AI]
    assert len(ai_actions) == 1
    assert ai_actions[0].type == ActionType.NORMALIZE_CATEGORIES
    assert ai_actions[0].parameters == {"mapping": {"Basline": "Baseline"}}
    assert not result.rejected

    loaded = load_dataset(source)
    cleaned, changeset = apply_plan(
        source,
        loaded.tables,
        result.plan,
        approved_ids={ai_actions[0].id},
    )
    assert cleaned["study"]["group"].tolist() == [
        "Baseline",
        "Baseline",
        "Followup",
        "Baseline",
    ]
    assert changeset.affected_cell_count == 1


def test_ai_cannot_merge_observed_categories_without_detector_pair(tmp_path: Path) -> None:
    source = tmp_path / "study.csv"
    _write_fuzzy_dataset(source)
    provider = FakeProvider(
        {
            "summary": "Attempt an unsupported merge.",
            "actions": [
                {
                    "type": "normalize_categories",
                    "table": "study",
                    "column": "group",
                    "parameters": {"mapping": {"Followup": "Baseline"}},
                    "rationale": "These look related.",
                    "confidence": 0.99,
                    "issue_ids": [],
                }
            ],
            "deferred_issue_ids": [],
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
    assert "not supported by detector candidate pairs" in result.rejected[0].reason


def test_ai_cannot_invent_missing_tokens(tmp_path: Path) -> None:
    source = tmp_path / "missing.csv"
    source.write_text(
        "patient_id,status\n"
        "P001,N/A\n"
        "P002,Active\n"
        "P003,Inactive\n",
        encoding="utf-8",
    )
    provider = FakeProvider(
        {
            "summary": "Attempt to classify a real category as missing.",
            "actions": [
                {
                    "type": "normalize_missing",
                    "table": "missing",
                    "column": "status",
                    "parameters": {"values": ["Active"]},
                    "rationale": "Treat Active as missing.",
                    "confidence": 1.0,
                    "issue_ids": [],
                }
            ],
            "deferred_issue_ids": [],
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
    assert "only normalize tokens already detected" in result.rejected[0].reason


def test_ai_category_cycles_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "study.csv"
    _write_fuzzy_dataset(source)
    provider = FakeProvider(
        {
            "summary": "Attempt a swap.",
            "actions": [
                {
                    "type": "normalize_categories",
                    "table": "study",
                    "column": "group",
                    "parameters": {
                        "mapping": {
                            "Basline": "Baseline",
                            "Baseline": "Basline",
                        }
                    },
                    "rationale": "Swap the two labels.",
                    "confidence": 0.9,
                    "issue_ids": [],
                }
            ],
            "deferred_issue_ids": [],
        }
    )

    result = propose_ai_plan(
        source,
        provider=provider,
        model="fake-model",
        include_rules=False,
    )

    assert result.plan.actions == []
    assert "cycles or chains" in result.rejected[0].reason


def test_model_resolution_requires_choice_when_multiple_are_installed() -> None:
    provider = FakeProvider(
        {"summary": "", "actions": [], "deferred_issue_ids": []},
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
                        "summary": "No changes.",
                        "actions": [],
                        "deferred_issue_ids": [],
                    }
                )
            }
        }


def test_ollama_provider_sends_json_schema_and_disables_streaming() -> None:
    provider = StubOllama()
    response = provider.generate_structured(
        model="test-model",
        system_prompt="system",
        user_prompt="user",
        response_model=AIPlannerResponse,
    )

    assert response.summary == "No changes."
    assert provider.sent is not None
    assert provider.sent["stream"] is False
    assert provider.sent["think"] is False
    assert provider.sent["format"]["type"] == "object"
    assert provider.sent["options"]["temperature"] == 0.1


def test_ai_plan_cli_writes_plan_and_exact_context(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "study.csv"
    _write_fuzzy_dataset(source)
    output = tmp_path / "ai_plan.json"
    context_out = tmp_path / "ai_context.json"

    provider = FakeProvider(
        {
            "summary": "Merge the typo pair.",
            "actions": [
                {
                    "type": "normalize_categories",
                    "table": "study",
                    "column": "group",
                    "parameters": {"mapping": {"Basline": "Baseline"}},
                    "rationale": "Evidence-supported typo correction.",
                    "confidence": 0.9,
                    "issue_ids": [],
                }
            ],
            "deferred_issue_ids": [],
        }
    )
    monkeypatch.setattr(
        "data_cleaning_assistant.cli.OllamaProvider",
        lambda url=None, think=False: provider,
    )

    runner = CliRunner()
    result = runner.invoke(
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
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert output.exists()
    assert context_out.exists()
    assert "AI-assisted plan written" in result.stdout
    assert '"origin": "ai"' in output.read_text(encoding="utf-8")


def test_incorrect_model_issue_id_does_not_block_valid_evidence(tmp_path: Path) -> None:
    source = tmp_path / "study.csv"
    _write_fuzzy_dataset(source)
    provider = FakeProvider(
        {
            "summary": "Correct the typo.",
            "actions": [
                {
                    "type": "normalize_categories",
                    "table": "study",
                    "column": "group",
                    "parameters": {"mapping": {"Basline": "Baseline"}},
                    "rationale": "The supplied detector pair supports the correction.",
                    "confidence": 0.9,
                    "issue_ids": ["issue_invented_by_model"],
                }
            ],
            "deferred_issue_ids": [],
        }
    )

    result = propose_ai_plan(
        source,
        provider=provider,
        model="fake-model",
        include_rules=False,
    )

    assert len(result.plan.actions) == 1
    assert not result.rejected
    assert "issue_invented_by_model" not in result.plan.actions[0].issue_ids
    assert result.plan.actions[0].issue_ids


def test_ai_row_deletion_is_always_rejected(tmp_path: Path) -> None:
    source = tmp_path / "duplicates.csv"
    source.write_text(
        "patient_id,status\n"
        "P001,Active\n"
        "P001,Active\n"
        "P002,Inactive\n",
        encoding="utf-8",
    )
    provider = FakeProvider(
        {
            "summary": "Remove the duplicate.",
            "actions": [
                {
                    "type": "remove_exact_duplicates",
                    "table": "duplicates",
                    "column": None,
                    "parameters": {"keep": "first"},
                    "rationale": "The rows are exact duplicates.",
                    "confidence": 1.0,
                    "issue_ids": [],
                }
            ],
            "deferred_issue_ids": [],
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
    assert "not allowed to propose row deletion" in result.rejected[0].reason
    assert result.rejected[0].parameters == {"keep": "first"}
