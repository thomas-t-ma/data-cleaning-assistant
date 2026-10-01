"""Independent, known-answer, synthetic semantic stress test for local Ollama.

No CoMMpass workbook, public patient records, or clinical files are loaded.
Each fixture is a separate one-column CSV in a temporary directory.
The expected decision and case ID never appear in the model prompt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from data_cleaning_assistant.actions import ActionType
from data_cleaning_assistant.ai_candidates import build_ai_candidates
from data_cleaning_assistant.ai_context import build_ai_context
from data_cleaning_assistant.ai_models import AIPlannerResponse
from data_cleaning_assistant.ai_planner import SYSTEM_PROMPT, _build_user_prompt, resolve_model
from data_cleaning_assistant.ai_provider import OllamaProvider, StructuredAIProvider
from data_cleaning_assistant.engine import profile_dataset
from data_cleaning_assistant.io import load_dataset
from data_cleaning_assistant.semantic_stress import SemanticCase, semantic_cases


# Experimental benchmark-only variant. Do not change the production planner
# prompt on the strength of this stress test. Compare against the frozen
# baseline on the same corpus and the same local model first.
EVIDENCE_CALIBRATED_ADDENDUM = """
Use evidence-proportional judgment when reviewing a bounded category candidate.
A merely imaginable difference is not evidence of a scientific distinction.
For a commonplace full-word misspelling with a single nonnumeric character edit,
unchanged surrounding words, and a more frequent correctly spelled form, accept
when there is no concrete competing code, definition, or semantic distinction.
Do not invent historical spellings, abbreviations, procedures, or coding meanings
to explain a deferral; if one is relevant, identify evidence present in the prompt.
Preserve distinct numeric IDs, units, doses, stages, timepoints, analytes, marker
names, treatments, outcome categories, and contextual missingness labels.
These protections take priority over high lexical similarity or label frequency.
If the supplied study dictionary explicitly establishes that two codes differ,
defer even when a numeric-zero-padding heuristic suggests equivalence.
If it explicitly establishes they are equivalent, that specific evidence
outweighs unsupported speculation about different meanings.
Never infer that two assay labels are equivalent from spelling similarity alone.
Return the supplied candidate ID with accept or defer; never invent a mapping.
"""


def _system_prompt(variant: str) -> str:
    if variant == "baseline":
        return SYSTEM_PROMPT
    if variant == "evidence_calibrated":
        return SYSTEM_PROMPT + "\n" + EVIDENCE_CALIBRATED_ADDENDUM
    raise ValueError("Unknown prompt variant: " + variant)


def _prepare_case(case: SemanticCase, folder: Path) -> dict[str, Any]:
    """Use production profiling + production bounded-candidate construction."""
    path = folder / "fixture.csv"
    pd.DataFrame(
        {
            case.column: (
                [case.canonical] * case.canonical_count
                + [case.variant] * case.variant_count
            )
        }
    ).to_csv(path, index=False)
    report = profile_dataset(path)
    candidates = build_ai_candidates(report, load_dataset(path).tables)
    pair = {case.canonical, case.variant}
    matching = [
        item for item in candidates
        if item.action.column == case.column
        and len(item.action.parameters.get("mapping", {})) == 1
        and (
            set(item.action.parameters["mapping"])
            | set(item.action.parameters["mapping"].values())
        ) == pair
    ]
    if not matching:
        return {"case": case, "candidate": None, "reason": "not_exposed"}
    if len(matching) != 1:
        return {"case": case, "candidate": None, "reason": "ambiguous_candidate_set"}
    candidate = matching[0]
    # A whitespace variant may independently produce a bounded TRIM_WHITESPACE
    # proposal and a category mapping. That is not a second semantic mapping.
    # Permit only this known overlap; any other extra action is a coverage failure.
    other_candidates = [item for item in candidates if item is not candidate]
    if any(
        item.action.type != ActionType.TRIM_WHITESPACE
        or item.action.column != case.column
        for item in other_candidates
    ):
        return {"case": case, "candidate": None, "reason": "ambiguous_candidate_set"}
    if (
        case.expected == "accept"
        and candidate.action.parameters["mapping"]
        != {case.variant: case.canonical}
    ):
        return {"case": case, "candidate": None, "reason": "wrong_mapping_direction"}

    context = build_ai_context(report, [candidate])
    context.tables = [table for table in context.tables if table.name == path.stem]
    linked_issue_ids = set(candidate.action.issue_ids)
    context.issues = [
        issue for issue in context.issues
        if issue.table == path.stem and issue.id in linked_issue_ids
    ]
    context.relationships = []
    if case.context:
        # Fixture metadata is substantive evidence, not the hidden expected
        # accept/defer label. The same label pair can have opposite ground truth
        # under distinct explicitly supplied dictionary rules.
        context.candidates[0].question += (
            "\nStudy-specific fixture metadata: " + case.context
        )
    if len(context.tables) != 1 or len(context.candidates) != 1:
        raise ValueError("Synthetic-only prompt construction failed")
    prompt = _build_user_prompt(context)
    # A model call must not have the hidden score or answer-key bookkeeping.
    if (
        case.id in json.dumps(context.model_dump(mode="json"))
        or "expected_decision" in prompt
        or "answer_key" in prompt
    ):
        raise ValueError("Hidden answer-key metadata reached the model prompt")
    return {"case": case, "candidate": candidate, "prompt": prompt, "reason": None}


def _score_reply(
    response: AIPlannerResponse, candidate_id: str, expected: str
) -> dict[str, Any]:
    selections = response.selections
    matching = [s for s in selections if s.candidate_id == candidate_id]
    unknown = [s.candidate_id for s in selections if s.candidate_id != candidate_id]
    protocol_ok = len(matching) == 1 and not unknown
    selection = matching[0] if matching else None
    actual = selection.decision.value if selection is not None else "missing"
    return {
        "actual": actual,
        "correct": actual == expected and protocol_ok,
        "protocol_ok": protocol_ok,
        "unknown_candidate_ids": unknown,
        "duplicate_candidate_id": len(matching) > 1,
        "rationale": selection.rationale if selection else "Missing candidate response",
        "confidence": selection.confidence if selection else None,
        "model_summary": response.summary,
    }


def run_suite(
    provider: StructuredAIProvider | None,
    model: str | None,
    *,
    runs: int,
    output: Path,
    audit_only: bool = False,
    cases: list[SemanticCase] | None = None,
    prompt_variant: str = "baseline",
) -> dict[str, Any]:
    if runs < 1:
        raise ValueError("--runs must be at least 1")
    system_prompt = _system_prompt(prompt_variant)
    if not audit_only and (provider is None or not model):
        raise ValueError("A model and provider are required unless --audit-only is used")
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"Benchmark score already exists: {output}. Choose a new --output."
        )
    selected = list(cases if cases is not None else semantic_cases())
    if not selected or len({case.id for case in selected}) != len(selected):
        raise ValueError("Corpus must contain distinct case IDs")
    if any(
        case.expected not in {"accept", "defer"}
        or case.canonical == case.variant
        or min(case.canonical_count, case.variant_count) < 1
        for case in selected
    ):
        raise ValueError("Invalid corpus: expected accept/defer and two different labels")
    fingerprint = hashlib.sha256(
        json.dumps([asdict(case) for case in selected], sort_keys=True).encode("utf-8")
    ).hexdigest()

    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="synthetic-semantic-") as temporary:
        prepared = []
        for case in selected:
            folder = Path(temporary) / case.id
            folder.mkdir()
            prepared.append(_prepare_case(case, folder))
        for run_index in range(1, runs + 1):
            for index, item in enumerate(prepared, start=1):
                case: SemanticCase = item["case"]
                record: dict[str, Any] = {
                    "run": run_index,
                    "id": case.id,
                    "group": case.group,
                    "column": case.column,
                    "labels": [case.canonical, case.variant],
                    "expected": case.expected,
                    "context_supplied": bool(case.context),
                    "candidate_exposed": item["candidate"] is not None,
                    "coverage_reason": item["reason"],
                    "actual": "not_evaluated" if audit_only else "not_exposed",
                    "correct": False,
                    "protocol_ok": None,
                    "latency_seconds": None,
                    "confidence": None,
                    "rationale": None,
                    "error": None,
                }
                if item["candidate"] is not None:
                    candidate = item["candidate"]
                    record["proposed_mapping"] = candidate.action.parameters["mapping"]
                    record["evidence_basis"] = candidate.context.evidence.get("basis")
                    if not audit_only:
                        started = time.perf_counter()
                        try:
                            response = provider.generate_structured(
                                model=model,
                                system_prompt=system_prompt,
                                user_prompt=item["prompt"],
                                response_model=AIPlannerResponse,
                            )
                            record.update(_score_reply(response, candidate.id, case.expected))
                        except Exception as exc:  # noqa: BLE001 - stress test records all model/runtime failures
                            record["actual"] = "model_error"
                            record["protocol_ok"] = False
                            record["error"] = f"{type(exc).__name__}: {exc}"
                        record["latency_seconds"] = round(
                            time.perf_counter() - started, 3
                        )
                records.append(record)
                status = (
                    "not_exposed" if not record["candidate_exposed"]
                    else "audit_only" if audit_only
                    else "correct" if record["correct"]
                    else record["actual"]
                )
                print(
                    f"Run {run_index}/{runs} case {index}/{len(selected)} "
                    f"{case.id}: {status}",
                    flush=True,
                )

    surfaced = [row for row in records if row["candidate_exposed"]]
    attempted = [row for row in surfaced if row["actual"] != "not_evaluated"]
    safe = [row for row in records if row["expected"] == "accept"]
    unsafe = [row for row in records if row["expected"] == "defer"]
    unsafe_merges = sum(
        row["actual"] == "accept" for row in unsafe
    )
    misses = sum(
        row["actual"] == "defer" for row in safe
    )
    protocol_failures = sum(row["protocol_ok"] is False for row in records)
    model_errors = sum(row["actual"] == "model_error" for row in records)
    model_correct = sum(row["correct"] for row in records)
    latencies = [
        row["latency_seconds"] for row in records
        if row["latency_seconds"] is not None
    ]
    groups: dict[str, dict[str, int]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["group"]].append(row)
    for name, rows in sorted(grouped.items()):
        groups[name] = {
            "total": len(rows),
            "exposed": sum(row["candidate_exposed"] for row in rows),
            "correct": sum(row["correct"] for row in rows),
            "unsafe_merges": sum(
                row["expected"] == "defer" and row["actual"] == "accept"
                for row in rows
            ),
            "missed_safe_corrections": sum(
                row["expected"] == "accept" and row["actual"] == "defer"
                for row in rows
            ),
        }
    payload = {
        "version": 1,
        "purpose": "Synthetic-only independent semantic stress test; no edits",
        "prompt_variant": prompt_variant,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "model": model if not audit_only else None,
        "provider": provider.name if provider and not audit_only else None,
        "audit_only": audit_only,
        "corpus_sha256": fingerprint,
        "unique_cases": len(selected),
        "safe_cases": sum(case.expected == "accept" for case in selected),
        "unsafe_cases": sum(case.expected == "defer" for case in selected),
        "runs": runs,
        "total_case_runs": len(records),
        "exposed_case_runs": len(surfaced),
        "not_exposed_case_runs": len(records) - len(surfaced),
        "safe_coverage": sum(row["candidate_exposed"] for row in safe),
        "safe_total": len(safe),
        "unsafe_exposure": sum(row["candidate_exposed"] for row in unsafe),
        "unsafe_total": len(unsafe),
        "model_calls": len(attempted),
        "correct_model_decisions": model_correct,
        "model_decision_denominator": len(attempted),
        "unsafe_merges_accepted": unsafe_merges,
        "missed_safe_corrections": misses,
        "protocol_failures": protocol_failures,
        "model_errors": model_errors,
        "mean_call_latency_seconds": (
            sum(latencies) / len(latencies) if latencies else None
        ),
        "strict_suite_pass": (
            not audit_only
            and len(surfaced) == len(records)
            and model_correct == len(records)
            and protocol_failures == 0
        ),
        "coverage_reasons": dict(
            Counter(row["coverage_reason"] for row in records if row["coverage_reason"])
        ),
        "by_group": groups,
        "results": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"Exposed: {payload['exposed_case_runs']}/{payload['total_case_runs']}; "
        f"correct model decisions: {model_correct}/{len(attempted)}; "
        f"unsafe accepts: {unsafe_merges}; missed safe edits: {misses}.",
        flush=True,
    )
    print(f"Score saved: {output}")
    print("Synthetic fixtures only; original study files untouched.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Exact installed Ollama model; omit if only one")
    parser.add_argument("--url", help="Ollama URL; defaults to local")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--think", action="store_true", help="Optional thinking-mode experiment")
    parser.add_argument(
        "--prompt-variant",
        choices=("baseline", "evidence_calibrated"),
        default="baseline",
        help="Compare the frozen production prompt against an experimental benchmark-only addendum",
    )
    parser.add_argument(
        "--audit-only", action="store_true",
        help="Check detector/candidate coverage without calling Ollama",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    provider = None if args.audit_only else OllamaProvider(args.url, think=args.think)
    model = None if args.audit_only else resolve_model(provider, args.model)
    run_suite(
        provider, model, runs=args.runs, output=args.output,
        audit_only=args.audit_only,
        prompt_variant=args.prompt_variant,
    )


if __name__ == "__main__":
    main()
