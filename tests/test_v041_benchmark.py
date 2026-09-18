from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from data_cleaning_assistant.actions import ActionType
from data_cleaning_assistant.ai_candidates import build_ai_candidates
from data_cleaning_assistant.benchmark import (
    BenchmarkCase,
    BenchmarkCaseResult,
    BenchmarkEffect,
    BenchmarkSummary,
    run_ai_benchmark,
)
from data_cleaning_assistant.cli import app
from data_cleaning_assistant.engine import profile_dataset
from data_cleaning_assistant.io import load_dataset


class CandidateProvider:
    name = "fake"

    def __init__(self, decision: str = "accept") -> None:
        self.decision = decision
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
        self.prompts.append(user_prompt)
        context_candidates = []
        marker = '"candidates":'
        assert marker in user_prompt

        # Tests use one candidate per custom case, so C001 is sufficient and deliberately
        # mirrors the short IDs the real model sees.
        context_candidates.append(
            {
                "candidate_id": "C001",
                "decision": self.decision,
                "rationale": "Synthetic benchmark decision.",
                "confidence": 0.9,
            }
        )
        return response_model.model_validate(
            {
                "summary": "Reviewed bounded candidate.",
                "selections": context_candidates,
            }
        )


def test_benchmark_scores_correct_required_candidate() -> None:
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
    provider = CandidateProvider("accept")

    summary = run_ai_benchmark(
        provider=provider,
        model="fake-model",
        cases=[case],
    )

    assert summary.pass_rate == 1.0
    assert summary.required_recall == 1.0
    assert summary.accepted_precision == 1.0
    assert "Hidden answer key" not in provider.prompts[0]


def test_benchmark_flags_semantically_unsafe_candidate_acceptance() -> None:
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

    summary = run_ai_benchmark(
        provider=CandidateProvider("accept"),
        model="fake-model",
        cases=[case],
    )

    assert summary.pass_rate == 0.0
    assert summary.forbidden_raw_proposals == 1
    assert summary.forbidden_accepted == 1
    assert summary.accepted_precision == 0.0


def test_benchmark_passes_when_semantically_unsafe_candidate_is_deferred() -> None:
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

    summary = run_ai_benchmark(
        provider=CandidateProvider("defer"),
        model="fake-model",
        cases=[case],
    )

    assert summary.pass_rate == 1.0
    assert summary.forbidden_accepted == 0


def test_duplicate_and_identifier_unsafe_actions_are_not_candidate_surface(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text(
        "patient_id,status\nP001,Active\nP001,Active\nP002,Inactive\n",
        encoding="utf-8",
    )
    identifier = tmp_path / "identifier.csv"
    identifier.write_text(
        "patient_id,value\n001,10\n002,20\nbad,30\n004,40\n005,50\n",
        encoding="utf-8",
    )

    for source in [duplicate, identifier]:
        loaded = load_dataset(source)
        candidates = build_ai_candidates(profile_dataset(source), loaded.tables)
        assert all(
            candidate.action.type != ActionType.REMOVE_EXACT_DUPLICATES
            and not (
                candidate.action.type == ActionType.COERCE_NUMERIC
                and candidate.action.column == "patient_id"
            )
            for candidate in candidates
        )


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
        lambda url=None, think=False: FakeOllama(),
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
