"""Benchmark-only two-stage semantic veto experiment for qwen3.5:4b.

Stage 1 uses the existing evidence_calibrated review prompt.
Stage 2 runs only when stage 1 ACCEPTS a single_edit_typo_pattern candidate.
The veto asks whether both differing terms are independently valid scientific/
clinical terms with different meanings. DISTINCT or UNCERTAIN forces DEFER.

No production planner behavior is changed and no data edits are applied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from data_cleaning_assistant.ai_models import AIPlannerResponse
from data_cleaning_assistant.ai_planner import resolve_model
from data_cleaning_assistant.ai_provider import OllamaProvider, StructuredAIProvider
from data_cleaning_assistant.semantic_holdout import semantic_neighbor_holdout_cases

from semantic_stress_test import _prepare_case, _score_reply, _system_prompt


AUDITOR_SYSTEM_PROMPT = """You are a semantic-collision auditor for scientific data cleaning.

A primary reviewer has proposed accepting a bounded one-character normalization.
Your only job is to decide whether the two differing terms are:
- SAME_TYPO: one is clearly a misspelling/nonword form of the other;
- DISTINCT: both are established valid scientific, clinical, anatomical, laboratory,
  procedural, or ordinary terms with different meanings;
- UNCERTAIN: the evidence is insufficient to confidently choose SAME_TYPO or DISTINCT.

Rules:
- Ignore label frequency as evidence of semantic equivalence.
- A one-character edit is NOT evidence that two terms mean the same thing.
- Use ordinary scientific/medical knowledge when the terms are established concepts.
- Do not invent hypothetical meanings, codes, abbreviations, historical spellings, or
  study-specific categories.
- If both spellings are recognizable valid terms with different meanings, choose DISTINCT.
- If the variant is an obvious malformed spelling/nonword of the canonical term and has no
  independently recognized meaning, choose SAME_TYPO.
- If you genuinely cannot tell, choose UNCERTAIN.
- Return the supplied candidate ID exactly and do not create or modify mappings.
"""


class NeighborAuditResponse(BaseModel):
    candidate_id: str
    verdict: Literal["same_typo", "distinct", "uncertain"]
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


def _audit_user_prompt(candidate) -> str:
    payload = {
        "candidate_id": candidate.id,
        "column": candidate.action.column,
        "proposed_mapping": candidate.action.parameters.get("mapping", {}),
        "evidence": candidate.context.evidence,
    }
    return (
        "Audit this already-bounded single-edit candidate for semantic collision. "
        "Do not infer or alter any mapping.\n\n"
        + json.dumps(payload, indent=2, sort_keys=True)
    )


def run_veto_suite(
    provider: StructuredAIProvider,
    model: str,
    *,
    runs: int,
    output: Path,
    cases=None,
) -> dict:
    if runs < 1:
        raise ValueError("--runs must be at least 1")
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Benchmark score already exists: {output}")

    cases = list(cases if cases is not None else semantic_neighbor_holdout_cases())
    if not cases:
        raise ValueError("At least one case is required")
    primary_system = _system_prompt("evidence_calibrated")
    records = []

    with tempfile.TemporaryDirectory(prefix="semantic-veto-") as temporary:
        prepared = []
        for case in cases:
            folder = Path(temporary) / case.id
            folder.mkdir()
            prepared.append(_prepare_case(case, folder))

        for run_index in range(1, runs + 1):
            for index, item in enumerate(prepared, start=1):
                case = item["case"]
                candidate = item["candidate"]
                if candidate is None:
                    raise RuntimeError(
                        f"Holdout candidate unexpectedly not exposed: {case.id} "
                        f"({item['reason']})"
                    )

                record = {
                    "run": run_index,
                    "id": case.id,
                    "group": case.group,
                    "column": case.column,
                    "labels": [case.canonical, case.variant],
                    "expected": case.expected,
                    "evidence_basis": candidate.context.evidence.get("basis"),
                    "primary_actual": None,
                    "primary_rationale": None,
                    "primary_confidence": None,
                    "audit_called": False,
                    "audit_verdict": None,
                    "audit_rationale": None,
                    "audit_confidence": None,
                    "final_actual": None,
                    "correct": False,
                    "protocol_ok": True,
                    "primary_latency_seconds": None,
                    "audit_latency_seconds": None,
                    "error": None,
                }

                started = time.perf_counter()
                try:
                    primary = provider.generate_structured(
                        model=model,
                        system_prompt=primary_system,
                        user_prompt=item["prompt"],
                        response_model=AIPlannerResponse,
                    )
                    scored = _score_reply(primary, candidate.id, case.expected)
                    record["primary_actual"] = scored["actual"]
                    record["primary_rationale"] = scored["rationale"]
                    record["primary_confidence"] = scored["confidence"]
                    record["protocol_ok"] = scored["protocol_ok"]
                except Exception as exc:
                    record["error"] = f"primary {type(exc).__name__}: {exc}"
                    record["protocol_ok"] = False
                    record["final_actual"] = "model_error"
                record["primary_latency_seconds"] = round(
                    time.perf_counter() - started, 3
                )

                if (
                    record["final_actual"] is None
                    and record["protocol_ok"]
                    and record["primary_actual"] == "accept"
                    and record["evidence_basis"] == "single_edit_typo_pattern"
                ):
                    record["audit_called"] = True
                    started = time.perf_counter()
                    try:
                        audit = provider.generate_structured(
                            model=model,
                            system_prompt=AUDITOR_SYSTEM_PROMPT,
                            user_prompt=_audit_user_prompt(candidate),
                            response_model=NeighborAuditResponse,
                        )
                        if audit.candidate_id != candidate.id:
                            raise ValueError(
                                f"Auditor returned unknown candidate ID {audit.candidate_id!r}"
                            )
                        record["audit_verdict"] = audit.verdict
                        record["audit_rationale"] = audit.rationale
                        record["audit_confidence"] = audit.confidence
                        record["final_actual"] = (
                            "accept" if audit.verdict == "same_typo" else "defer"
                        )
                    except Exception as exc:
                        record["error"] = f"audit {type(exc).__name__}: {exc}"
                        record["protocol_ok"] = False
                        record["final_actual"] = "model_error"
                    record["audit_latency_seconds"] = round(
                        time.perf_counter() - started, 3
                    )

                if record["final_actual"] is None:
                    record["final_actual"] = record["primary_actual"]

                record["correct"] = (
                    record["protocol_ok"]
                    and record["final_actual"] == case.expected
                )
                records.append(record)
                print(
                    f"Run {run_index}/{runs} case {index}/{len(cases)} {case.id}: "
                    f"primary={record['primary_actual']} "
                    f"audit={record['audit_verdict']} "
                    f"final={record['final_actual']}",
                    flush=True,
                )

    safe = [row for row in records if row["expected"] == "accept"]
    risky = [row for row in records if row["expected"] == "defer"]
    primary_unsafe = sum(row["primary_actual"] == "accept" for row in risky)
    final_unsafe = sum(row["final_actual"] == "accept" for row in risky)
    primary_safe = sum(row["primary_actual"] == "accept" for row in safe)
    final_safe = sum(row["final_actual"] == "accept" for row in safe)
    protocol_failures = sum(not row["protocol_ok"] for row in records)
    model_errors = sum(row["final_actual"] == "model_error" for row in records)
    auditor_counts = Counter(
        row["audit_verdict"] for row in records if row["audit_called"]
    )
    primary_latencies = [
        row["primary_latency_seconds"] for row in records
        if row["primary_latency_seconds"] is not None
    ]
    audit_latencies = [
        row["audit_latency_seconds"] for row in records
        if row["audit_latency_seconds"] is not None
    ]

    payload = {
        "version": 1,
        "purpose": "Benchmark-only two-stage semantic-neighbor veto; no edits",
        "model": model,
        "provider": provider.name,
        "runs": runs,
        "unique_cases": len(cases),
        "total_case_runs": len(records),
        "primary_prompt_variant": "evidence_calibrated",
        "primary_prompt_sha256": hashlib.sha256(
            primary_system.encode("utf-8")
        ).hexdigest(),
        "auditor_prompt_sha256": hashlib.sha256(
            AUDITOR_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "primary_safe_accepts": primary_safe,
        "final_safe_accepts": final_safe,
        "safe_total": len(safe),
        "primary_unsafe_accepts": primary_unsafe,
        "final_unsafe_accepts": final_unsafe,
        "unsafe_total": len(risky),
        "correct_final_decisions": sum(row["correct"] for row in records),
        "audits_called": sum(row["audit_called"] for row in records),
        "audit_verdict_counts": dict(auditor_counts),
        "protocol_failures": protocol_failures,
        "model_errors": model_errors,
        "mean_primary_latency_seconds": (
            sum(primary_latencies) / len(primary_latencies)
            if primary_latencies else None
        ),
        "mean_audit_latency_seconds": (
            sum(audit_latencies) / len(audit_latencies)
            if audit_latencies else None
        ),
        "results": records,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"Final correct: {payload['correct_final_decisions']}/{len(records)}; "
        f"safe accepts: {final_safe}/{len(safe)}; "
        f"unsafe accepts: {final_unsafe}/{len(risky)}; "
        f"audits: {payload['audits_called']}.",
        flush=True,
    )
    print(f"Score saved: {output}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Exact installed Ollama model; omit if only one")
    parser.add_argument("--url", help="Ollama URL; defaults to local")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    provider = OllamaProvider(args.url, think=args.think)
    model = resolve_model(provider, args.model)
    run_veto_suite(provider, model, runs=args.runs, output=args.output)


if __name__ == "__main__":
    main()
