from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from data_cleaning_assistant.actions import ActionType
from data_cleaning_assistant.benchmark import (
    BenchmarkCase,
    BenchmarkCaseResult,
    BenchmarkEffect,
    BenchmarkSummary,
    run_ai_benchmark,
)
from data_cleaning_assistant.cli import app


class SequenceProvider:
    name = "fake"

    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = list(payloads)
        self.prompts: list[str] = []

    def list_models(self) -> list[str]:
        return ["fake-model"]

    def generate_structured(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_model,
    ):
        assert model == "fake-model"
        assert system_prompt
        self.prompts.append(user_prompt)
        payload = self.payloads.pop(0)
        return response_model.model_validate(payload)


def test_benchmark_scores_correct_required_action() -> None:
    case = BenchmarkCase(
        id="typo",
        description="Hidden answer key: Basline should become Baseline.",
        csv_text=(
            "patient_id,group\n"
            "P001,Baseline\n"
            "P002,Basline\n"
            "P003,Baseline\n"
        ),
        required=[
            BenchmarkEffect(
                type=ActionType.NORMALIZE_CATEGORIES,
                column="group",
                parameters={"mapping": {"Basline": "Baseline"}},
            )
        ],
    )
    provider = SequenceProvider(
        [
            {
                "summary": "Correct the obvious typo.",
                "actions": [
                    {
                        "type": "normalize_categories",
                        "table": "dataset",
                        "column": "group",
                        "parameters": {"mapping": {"Basline": "Baseline"}},
                        "rationale": "The profiler surfaced this exact fuzzy pair.",
                        "confidence": 0.9,
                        "issue_ids": [],
                    }
                ],
                "deferred_issue_ids": [],
            }
        ]
    )

    summary = run_ai_benchmark(
        provider=provider,
        model="fake-model",
        cases=[case],
    )

    assert summary.pass_rate == 1.0
    assert summary.required_recall == 1.0
    assert summary.accepted_precision == 1.0
    assert summary.forbidden_accepted == 0
    assert "Hidden answer key" not in provider.prompts[0]


def test_benchmark_flags_semantically_unsafe_detector_permitted_merge() -> None:
    case = BenchmarkCase(
        id="stage",
        description="Stage II and Stage III must remain distinct.",
        csv_text=(
            "patient_id,stage\n"
            "P001,Stage II\n"
            "P002,Stage III\n"
            "P003,Stage II\n"
            "P004,Stage III\n"
        ),
        forbidden=[
            BenchmarkEffect(
                type=ActionType.NORMALIZE_CATEGORIES,
                column="stage",
                parameters=None,
            )
        ],
    )
    provider = SequenceProvider(
        [
            {
                "summary": "Incorrectly merge the stages.",
                "actions": [
                    {
                        "type": "normalize_categories",
                        "table": "dataset",
                        "column": "stage",
                        "parameters": {"mapping": {"Stage III": "Stage II"}},
                        "rationale": "The labels are lexically similar.",
                        "confidence": 0.9,
                        "issue_ids": [],
                    }
                ],
                "deferred_issue_ids": [],
            }
        ]
    )

    summary = run_ai_benchmark(
        provider=provider,
        model="fake-model",
        cases=[case],
    )

    assert summary.pass_rate == 0.0
    assert summary.forbidden_raw_proposals == 1
    assert summary.forbidden_accepted == 1
    assert summary.accepted_precision == 0.0


def test_benchmark_counts_validator_catch_for_identifier_coercion() -> None:
    case = BenchmarkCase(
        id="identifier",
        description="Identifiers must not be numerically coerced.",
        csv_text=(
            "patient_id,value\n"
            "001,10\n"
            "002,20\n"
            "bad,30\n"
            "004,40\n"
            "005,50\n"
        ),
        forbidden=[
            BenchmarkEffect(
                type=ActionType.COERCE_NUMERIC,
                column="patient_id",
                parameters=None,
            )
        ],
    )
    provider = SequenceProvider(
        [
            {
                "summary": "Attempt identifier coercion.",
                "actions": [
                    {
                        "type": "coerce_numeric",
                        "table": "dataset",
                        "column": "patient_id",
                        "parameters": {"unparseable": "keep"},
                        "rationale": "Most values look numeric.",
                        "confidence": 0.9,
                        "issue_ids": [],
                    }
                ],
                "deferred_issue_ids": [],
            }
        ]
    )

    summary = run_ai_benchmark(
        provider=provider,
        model="fake-model",
        cases=[case],
    )

    assert summary.pass_rate == 1.0
    assert summary.forbidden_raw_proposals == 1
    assert summary.forbidden_accepted == 0
    assert summary.forbidden_rejected == 1
    assert summary.forbidden_validator_catch_rate == 1.0
    assert summary.validator_rejections == 1


def test_ai_benchmark_cli_writes_json(tmp_path: Path, monkeypatch) -> None:
    class FakeOllama:
        name = "ollama"
        base_url = "http://localhost:11434"

        def list_models(self) -> list[str]:
            return ["fake-model"]

    summary = BenchmarkSummary(
        model="fake-model",
        provider="ollama",
        runs_per_case=1,
        case_runs=1,
        passed_case_runs=1,
        pass_rate=1.0,
        required_total=1,
        required_hits=1,
        required_recall=1.0,
        accepted_total=1,
        accepted_correct=1,
        accepted_precision=1.0,
        forbidden_raw_proposals=0,
        forbidden_accepted=0,
        forbidden_rejected=0,
        forbidden_validator_catch_rate=None,
        validator_rejections=0,
        mean_latency_seconds=0.5,
        results=[
            BenchmarkCaseResult(
                case_id="case",
                description="case",
                run_index=1,
                latency_seconds=0.5,
                raw_proposals=1,
                accepted_actions=1,
                validator_rejections=0,
                required_total=1,
                required_hits=1,
                required_misses=0,
                forbidden_raw_proposals=0,
                forbidden_accepted=0,
                forbidden_rejected=0,
                unexpected_accepted=0,
                passed=True,
            )
        ],
    )

    monkeypatch.setattr(
        "data_cleaning_assistant.cli.OllamaProvider",
        lambda url=None: FakeOllama(),
    )
    monkeypatch.setattr(
        "data_cleaning_assistant.cli.run_ai_benchmark",
        lambda **kwargs: summary,
    )

    output = tmp_path / "benchmark.json"
    result = CliRunner().invoke(
        app,
        [
            "ai-benchmark",
            "--model",
            "fake-model",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Case-run pass rate" in result.stdout
    assert output.exists()
    assert '"model": "fake-model"' in output.read_text(encoding="utf-8")
