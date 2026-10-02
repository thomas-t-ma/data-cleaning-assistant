from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from data_cleaning_assistant.cli import app
from data_cleaning_assistant.guided import GuidedStage, load_guided_session
from data_cleaning_assistant.io import load_dataset


class GuidedFakeProvider:
    name = "ollama"
    base_url = "http://fake.local"

    def __init__(self, models: list[str] | None = None) -> None:
        self.models = models or ["fake-model"]
        self.calls = 0

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
        self.calls += 1
        context = json.loads(user_prompt.split("\n\n", 1)[1])
        return response_model.model_validate(
            {
                "summary": "Accept bounded typo-shaped candidates.",
                "selections": [
                    {
                        "candidate_id": candidate["id"],
                        "decision": "accept",
                        "rationale": "Synthetic fixture contains a clear one-edit typo.",
                        "confidence": 0.95,
                    }
                    for candidate in context["candidates"]
                ],
            }
        )


def _write_fuzzy_dataset(path: Path) -> None:
    path.write_text(
        "patient_id,group\n"
        "P001,Baseline\n"
        "P002,Basline\n"
        "P003,Followup\n"
        "P004,Baseline\n",
        encoding="utf-8",
    )


def _write_schema(path: Path) -> None:
    path.write_text(
        "version: 1\n"
        "tables:\n"
        "  study:\n"
        "    grain_key: [patient_id]\n",
        encoding="utf-8",
    )


def test_guided_ai_plan_runs_bounded_semantic_review_and_verifies(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "study.csv"
    schema = tmp_path / "schema.yaml"
    workspace = tmp_path / "run"
    _write_fuzzy_dataset(source)
    _write_schema(schema)

    provider = GuidedFakeProvider()
    monkeypatch.setattr(
        "data_cleaning_assistant.cli.OllamaProvider",
        lambda url=None, think=False: provider,
    )

    result = CliRunner().invoke(
        app,
        [
            "guided-clean",
            str(source),
            "--workspace",
            str(workspace),
            "--schema",
            str(schema),
            "--model",
            "fake-model",
        ],
        input="a\n",
    )

    assert result.exit_code == 0, result.output
    assert "GUIDED CLEANING COMPLETE" in result.output
    assert provider.calls >= 2

    session = load_guided_session(workspace)
    assert session.stage == GuidedStage.VERIFIED
    assert session.planning_mode == "ai"
    assert session.provider == "ollama"
    assert session.model == "fake-model"
    assert session.ai_provenance_path is not None

    cleaned = load_dataset(session.cleaned_path).tables
    frame = next(iter(cleaned.values()))
    assert frame["group"].tolist() == [
        "Baseline",
        "Baseline",
        "Followup",
        "Baseline",
    ]

    provenance = Path(session.ai_provenance_path).read_text(encoding="utf-8")
    assert '"model": "fake-model"' in provenance
    assert '"review_call_count": 2' in provenance
    assert "P001" not in provenance
    assert "study.csv" not in provenance


def test_guided_ai_model_resolution_fails_before_workspace_creation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "study.csv"
    workspace = tmp_path / "run"
    _write_fuzzy_dataset(source)

    provider = GuidedFakeProvider(models=["other-model"])
    monkeypatch.setattr(
        "data_cleaning_assistant.cli.OllamaProvider",
        lambda url=None, think=False: provider,
    )

    result = CliRunner().invoke(
        app,
        [
            "guided-clean",
            str(source),
            "--workspace",
            str(workspace),
            "--model",
            "missing-model",
        ],
    )

    assert result.exit_code != 0
    assert "not installed" in result.output
    assert not workspace.exists()


def test_guided_rejects_ollama_options_without_model(
    tmp_path: Path,
) -> None:
    source = tmp_path / "study.csv"
    workspace = tmp_path / "run"
    _write_fuzzy_dataset(source)

    result = CliRunner().invoke(
        app,
        [
            "guided-clean",
            str(source),
            "--workspace",
            str(workspace),
            "--think",
        ],
    )

    assert result.exit_code != 0
    assert "Ollama options require a model" in result.output
    assert not workspace.exists()
