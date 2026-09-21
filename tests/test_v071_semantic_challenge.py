"""Offline semantic benchmark tests; fake local providers only, no Ollama in CI."""

from __future__ import annotations

import json
from pathlib import Path

from data_cleaning_assistant.ai_models import (
    AICandidateSelection,
    AIPlannerResponse,
    CandidateDecision,
)
from data_cleaning_assistant.planner import file_sha256
from test_v07_real_data_challenge import ROOT, _setup, _successful


class FakeProvider:
    name = "offline_fake"

    def __init__(self, *, invert: bool = False, omit_defer: bool = False) -> None:
        self.invert = invert
        self.omit_defer = omit_defer
        self.prompts: list[str] = []

    def generate_structured(
        self, *, model, system_prompt, user_prompt, response_model
    ) -> AIPlannerResponse:
        assert model == "fake:local"
        assert response_model is AIPlannerResponse
        self.prompts.append(user_prompt)
        context = json.loads(user_prompt.split("\n\n", 1)[1])
        assert {table["name"] for table in context["tables"]} == {"BenchmarkLabels"}
        assert {issue["table"] for issue in context["issues"]} == {"BenchmarkLabels"}
        assert context["relationships"] == []
        assert {c["table"] for c in context["candidates"]} == {"BenchmarkLabels"}
        assert "case_id" not in user_prompt
        assert "answer_key.json" not in user_prompt
        assert "diagnosis_key" not in user_prompt

        selections = []
        for candidate in context["candidates"]:
            mapping = candidate["proposed_change"]["mapping"]
            pair = frozenset(mapping) | frozenset(mapping.values())
            decision = (
                CandidateDecision.ACCEPT if pair == {
                    "Peripheral Blood", "Peripheral Blod"
                } else CandidateDecision.DEFER
            )
            if self.invert:
                decision = (
                    CandidateDecision.DEFER if decision == CandidateDecision.ACCEPT
                    else CandidateDecision.ACCEPT
                )
            if self.omit_defer and decision == CandidateDecision.DEFER:
                continue
            selections.append(AICandidateSelection(
                candidate_id=candidate["id"],
                decision=decision,
                rationale="Offline fixture model response.",
                confidence=0.9,
            ))
        return AIPlannerResponse(summary="Offline fixture", selections=selections)


def _challenge(tmp_path: Path) -> tuple[Path, Path, Path]:
    source, schema, directory = _setup(tmp_path)
    _successful(
        str(ROOT / "examples" / "commpass_challenge.py"),
        "make", "--source", str(source),
        "--schema", str(schema), "--out", str(directory),
    )
    return source, schema, directory


def test_isolated_semantic_model_pass_and_unmodified_source(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    from score_commpass_semantics import prepare, run

    source, _schema, challenge = _challenge(tmp_path)
    before = file_sha256(source)
    challenge_hash = file_sha256(challenge / "commpass_challenge.xlsx")
    context, candidates, answers = prepare(challenge)
    assert len(candidates) == len(answers) == 2
    assert {candidate.action.column for candidate in candidates} == {
        "specimen_source", "assay_label"
    }
    assert {t.name for t in context.tables} == {"BenchmarkLabels"}
    output = challenge / "semantic_score.json"
    provider = FakeProvider()
    result = run(challenge, provider, "fake:local", 3, output)
    assert result["passing_runs"] == 3
    assert result["correct_decisions"] == 6
    assert result["total_decisions"] == 6
    assert len(provider.prompts) == 3
    assert all(item["protocol_ok"] for item in result["results"])
    assert file_sha256(source) == before
    assert file_sha256(challenge / "commpass_challenge.xlsx") == challenge_hash
    assert output.exists()


def test_wrong_accept_of_distinct_assays_fails(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    from score_commpass_semantics import run

    _, _, challenge = _challenge(tmp_path)
    result = run(
        challenge, FakeProvider(invert=True), "fake:local", 2,
        challenge / "wrong.json",
    )
    assert result["passing_runs"] == 0
    assert result["correct_decisions"] == 0
    for item in result["results"]:
        assert item["passed"] is False


def test_omitted_defer_does_not_count_as_correct(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    from score_commpass_semantics import run

    _, _, challenge = _challenge(tmp_path)
    result = run(
        challenge, FakeProvider(omit_defer=True), "fake:local", 1,
        challenge / "missing.json",
    )
    assert result["passing_runs"] == 0
    assert result["results"][0]["protocol_ok"] is False
    assert result["results"][0]["correct"] == 1
    assert any(
        item["actual"] == "missing"
        for item in result["results"][0]["decisions"]
    )


def test_unknown_duplicate_and_missing_ids_fail_protocol(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    from score_commpass_semantics import prepare, score_response

    _, _, challenge = _challenge(tmp_path)
    _, candidates, answers = prepare(challenge)
    good = [
        AICandidateSelection(
            candidate_id=candidate.id,
            decision=CandidateDecision(answers[candidate.id]),
            rationale="fixture",
            confidence=1.0,
        )
        for candidate in candidates
    ]
    response = AIPlannerResponse(
        summary="fake",
        selections=good + [good[0], AICandidateSelection(
            candidate_id="not_a_candidate",
            decision=CandidateDecision.DEFER,
            rationale="extra",
            confidence=0.1,
        )],
    )
    scored = score_response(response, candidates, answers)
    assert not scored["passed"]
    assert not scored["protocol_ok"]
    assert scored["unknown_candidate_ids"] == ["not_a_candidate"]
    assert scored["duplicate_candidate_ids"] == [good[0].candidate_id]


def test_rejects_reusing_score_output_and_mismatched_hash(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    from score_commpass_semantics import prepare, run

    _, _, challenge = _challenge(tmp_path)
    output = challenge / "score.json"
    run(challenge, FakeProvider(), "fake:local", 1, output)
    try:
        run(challenge, FakeProvider(), "fake:local", 1, output)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Scoring must never overwrite prior run output")

    answer = challenge / "answer_key.json"
    contents = json.loads(answer.read_text(encoding="utf-8"))
    contents["challenge_sha256"] = "0" * 64
    answer.write_text(json.dumps(contents), encoding="utf-8")
    try:
        prepare(challenge)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("Hash mismatch should fail closed")
