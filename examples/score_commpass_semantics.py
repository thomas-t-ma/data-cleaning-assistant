"""Score local Ollama semantic decisions on the synthetic CoMMpass challenge.

Only BenchmarkLabels names/candidates are passed to the model. The model
never receives patient records, the GDC table profiles, answer-key labels,
or permission to apply changes. Use a fresh --output for each experiment.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from data_cleaning_assistant.actions import ActionType
from data_cleaning_assistant.ai_candidates import build_ai_candidates
from data_cleaning_assistant.ai_context import build_ai_context
from data_cleaning_assistant.ai_models import AIPlannerResponse, CandidateDecision
from data_cleaning_assistant.ai_planner import SYSTEM_PROMPT, _build_user_prompt, resolve_model
from data_cleaning_assistant.ai_provider import OllamaProvider, StructuredAIProvider
from data_cleaning_assistant.engine import profile_dataset
from data_cleaning_assistant.io import load_dataset
from data_cleaning_assistant.planner import file_sha256
from data_cleaning_assistant.study_schema import load_study_schema

from commpass_challenge import MANIFEST, SCHEMA, SYNTHETIC, WORKBOOK

TYPO = frozenset(("Peripheral Blood", "Peripheral Blod"))


def _pair(candidate: Any) -> frozenset[str]:
    mapping = candidate.action.parameters.get("mapping", {})
    if (
        candidate.action.type != ActionType.NORMALIZE_CATEGORIES
        or len(mapping) != 1
    ):
        raise ValueError("Expected exactly one bounded category mapping")
    old, new = next(iter(mapping.items()))
    return frozenset((str(old), str(new)))


def prepare(challenge_dir: Path) -> tuple[Any, list[Any], dict[str, str]]:
    """Fail closed before contacting the model or exposing source-derived data."""
    root = challenge_dir.expanduser().resolve()
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or manifest.get("synthetic_sheet") != SYNTHETIC:
        raise ValueError("This is not the expected seeded CoMMpass challenge")
    if file_sha256(root / WORKBOOK) != manifest["challenge_sha256"]:
        raise ValueError("Challenge workbook hash mismatch")
    if file_sha256(Path(manifest["source_path"])) != manifest["source_sha256"]:
        raise ValueError("Original public workbook hash mismatch")
    if file_sha256(root / SCHEMA) != manifest["source_schema_sha256"]:
        raise ValueError("Challenge schema hash mismatch")

    schema = load_study_schema(root / SCHEMA)
    data = load_dataset(root / WORKBOOK)
    schema.validate_dataset(data.tables)
    report = profile_dataset(root / WORKBOOK, schema=schema)
    all_candidates = build_ai_candidates(report, data.tables, schema=schema)
    selected = [
        candidate for candidate in all_candidates
        if candidate.action.table == SYNTHETIC
    ]
    distinct = frozenset(manifest["protected_distinct_labels"])
    by_pair = {_pair(candidate): candidate for candidate in selected}
    if (
        len(selected) != 2 or len(by_pair) != 2
        or set(by_pair) != {TYPO, distinct}
    ):
        raise ValueError(
            "Synthetic candidate set changed; require exactly the typo and "
            "the distinct numbered-assay pair. No model call was made."
        )
    typo_mapping = by_pair[TYPO].action.parameters["mapping"]
    if typo_mapping != {"Peripheral Blod": "Peripheral Blood"}:
        raise ValueError("Typo normalization direction changed; refusing to score")

    # Use the SAME deterministic candidate construction and SAME prompt as
    # ai-plan, but strictly restrict the model context to the synthetic sheet.
    context = build_ai_context(report, selected)
    context.tables = [t for t in context.tables if t.name == SYNTHETIC]
    source_ids = {issue_id for candidate in selected for issue_id in candidate.action.issue_ids}
    context.issues = [
        issue for issue in context.issues
        if issue.table == SYNTHETIC and issue.id in source_ids
    ]
    context.relationships = []
    context.candidates = [candidate.context for candidate in selected]
    if len(context.tables) != 1 or len(context.candidates) != 2:
        raise ValueError("Synthetic-only model context construction failed")
    expectation = {
        by_pair[TYPO].id: CandidateDecision.ACCEPT.value,
        by_pair[distinct].id: CandidateDecision.DEFER.value,
    }
    return context, selected, expectation


def score_response(
    response: AIPlannerResponse,
    candidates: list[Any],
    expectation: dict[str, str],
) -> dict[str, Any]:
    """Score literal returned decisions, not a synthesized accepted plan.

    Unknown/duplicate/omitted selections are scored as incorrect protocol
    responses. Missing responses do NOT count as successful deferrals.
    """
    by_id = {candidate.id: candidate for candidate in candidates}
    records: list[dict[str, Any]] = []
    unknown: list[str] = []
    duplicates: list[str] = []
    seen: dict[str, Any] = {}
    for selection in response.selections:
        if selection.candidate_id not in by_id:
            unknown.append(selection.candidate_id)
            continue
        if selection.candidate_id in seen:
            duplicates.append(selection.candidate_id)
            continue
        seen[selection.candidate_id] = selection

    for candidate in candidates:
        selection = seen.get(candidate.id)
        actual = selection.decision.value if selection else "missing"
        required = expectation[candidate.id]
        mapping = candidate.action.parameters["mapping"]
        row = {
            "candidate_id": candidate.id,
            "column": candidate.action.column,
            "mapping": mapping,  # synthetic labels only
            "basis": candidate.context.evidence.get("basis"),
            "expected": required,
            "actual": actual,
            "correct": actual == required and candidate.id not in duplicates,
            "confidence": selection.confidence if selection else None,
            "rationale": selection.rationale if selection else "No model selection",
        }
        records.append(row)
    correct = sum(row["correct"] for row in records)
    protocol_ok = not unknown and not duplicates and len(seen) == len(candidates)
    return {
        "correct": correct,
        "total": len(candidates),
        "passed": correct == len(candidates) and protocol_ok,
        "protocol_ok": protocol_ok,
        "unknown_candidate_ids": unknown,
        "duplicate_candidate_ids": duplicates,
        "decisions": records,
    }


def run(
    challenge_dir: Path,
    provider: StructuredAIProvider,
    model: str,
    runs: int,
    output: Path,
) -> dict[str, Any]:
    if runs < 1:
        raise ValueError("--runs must be at least 1")
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"Score already exists: {output}. Choose a new --output to preserve history."
        )
    context, candidates, expectation = prepare(challenge_dir)
    prompt = _build_user_prompt(context)
    # No test answer, source identifiers, source path, or other GDC tables go
    # into the model prompt. We score using answer-key expectations only AFTER
    # the returned model decisions are fixed.
    if any(
        item in prompt
        for item in (
            "case_id", "diagnosis_key", "followup_key", "labtest_key",
            "commpass_pilot.xlsx", "answer_key.json",
        )
    ):
        raise ValueError("Unexpected original-data identifier in model context")

    results: list[dict[str, Any]] = []
    for index in range(1, runs + 1):
        started = time.perf_counter()
        try:
            response = provider.generate_structured(
                model=model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                response_model=AIPlannerResponse,
            )
            result = score_response(response, candidates, expectation)
            result["model_summary"] = response.summary
            result["error"] = None
        except Exception as exc:
            # Make errors count as failures, not false deferrals.
            result = {
                "correct": 0, "total": len(candidates),
                "passed": False, "protocol_ok": False,
                "decisions": [], "error": f"{type(exc).__name__}: {exc}",
            }
        result["run"] = index
        result["latency_seconds"] = round(time.perf_counter() - started, 3)
        results.append(result)
        print(
            f"Run {index}/{runs}: {result['correct']}/{result['total']} correct; "
            f"pass={result['passed']}; {result['latency_seconds']}s",
            flush=True,
        )
    payload = {
        "version": 1,
        "purpose": "Synthetic-label semantic accept/defer benchmark, no data edits",
        "provider": provider.name,
        "model": model,
        "challenge_sha256": file_sha256(challenge_dir / WORKBOOK),
        "runs": runs,
        "passing_runs": sum(result["passed"] for result in results),
        "pass_rate": sum(result["passed"] for result in results) / runs,
        "correct_decisions": sum(result["correct"] for result in results),
        "total_decisions": sum(result["total"] for result in results),
        "mean_latency_seconds": (
            sum(result["latency_seconds"] for result in results) / runs
        ),
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Semantic score: {output}")
    print("Only synthetic label metadata were sent to the local model; no edits were made.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--model", help="Exact installed Ollama model; omit if only one")
    parser.add_argument("--url", help="Ollama base URL (defaults to local Ollama)")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--think", action="store_true", help="Enable Ollama thinking mode")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provider = OllamaProvider(args.url, think=args.think)
    model = resolve_model(provider, args.model)
    run(args.challenge, provider, model, args.runs, args.output)


if __name__ == "__main__":
    main()
