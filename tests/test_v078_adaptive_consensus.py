"""Offline guards for the frozen adaptive-consensus validation."""

from __future__ import annotations

from pathlib import Path

from data_cleaning_assistant.semantic_consensus_validation import adaptive_consensus_cases
from data_cleaning_assistant.semantic_holdout import semantic_neighbor_holdout_cases
from data_cleaning_assistant.semantic_stress import semantic_cases
from data_cleaning_assistant.semantic_validation import frozen_validation_cases


def _pair(case) -> frozenset[str]:
    return frozenset((case.canonical, case.variant))


def test_adaptive_consensus_corpus_is_balanced_and_fresh() -> None:
    cases = adaptive_consensus_cases()
    assert len(cases) == 100
    assert len({case.id for case in cases}) == 100
    assert sum(case.expected == "accept" for case in cases) == 50
    assert sum(case.expected == "defer" for case in cases) == 50

    prior = semantic_cases() + semantic_neighbor_holdout_cases() + frozen_validation_cases()
    prior_pairs = {_pair(case) for case in prior}
    assert not ({_pair(case) for case in cases} & prior_pairs)


def test_adaptive_consensus_all_safe_cases_reach_candidate_review(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from semantic_stress_test import _prepare_case

    safe_failures = []
    unsafe_screened = []
    for case in adaptive_consensus_cases():
        folder = tmp_path / case.id
        folder.mkdir()
        item = _prepare_case(case, folder)
        if item["candidate"] is None:
            entry = (case.id, item["reason"])
            if case.expected == "accept":
                safe_failures.append(entry)
            else:
                unsafe_screened.append(entry)

    # A safe correction that never reaches review is lost automation coverage.
    assert safe_failures == []

    # Unsafe pairs may be screened out by deterministic candidate construction.
    # That is a valid fail-closed outcome and the live report records it separately.
    assert all(reason in {"not_exposed", "ambiguous_candidate_set"} for _, reason in unsafe_screened)


def test_consensus_majority_and_fail_closed(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from adaptive_consensus_validation import _consensus_from_votes

    def vote(decision: str, ok: bool = True):
        return {
            "decision": decision,
            "protocol_ok": ok,
            "rationale": "",
            "confidence": 0.9,
            "latency_seconds": 0.1,
            "error": None if ok else "fixture",
        }

    assert _consensus_from_votes(
        [vote("accept"), vote("defer"), vote("accept")]
    ) == ("accept", "majority_2_1")
    assert _consensus_from_votes(
        [vote("accept"), vote("defer"), vote("defer")]
    ) == ("defer", "majority_1_2")
    assert _consensus_from_votes(
        [vote("accept"), vote("accept", ok=False)]
    ) == ("defer", "fail_closed_protocol_or_model_error")
