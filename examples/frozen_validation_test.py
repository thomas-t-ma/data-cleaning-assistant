"""One-shot evaluation of frozen semantic architectures on 100 fresh cases.

The default live run evaluates baseline, evidence_calibrated, and blind_veto in one
invocation and writes one report only after all requested modes finish. This reduces
the temptation to tune between architecture results.

No production data are loaded or edited.
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

from data_cleaning_assistant.ai_models import AIPlannerResponse
from data_cleaning_assistant.ai_planner import resolve_model
from data_cleaning_assistant.ai_provider import OllamaProvider, StructuredAIProvider
from data_cleaning_assistant.semantic_validation import frozen_validation_cases

from semantic_stress_test import _prepare_case, _score_reply, _system_prompt
from semantic_veto_test import (
    AUDITOR_SYSTEM_PROMPT,
    NeighborAuditResponse,
    _audit_user_prompt,
)


MODES = ("baseline", "evidence_calibrated", "blind_veto")


def _final_correct(expected: str, final_actual: str, protocol_ok: bool) -> bool:
    if not protocol_ok:
        return False
    if expected == "accept":
        return final_actual == "accept"
    return final_actual in {"defer", "not_exposed"}


def _summarize_mode(records: list[dict[str, Any]]) -> dict[str, Any]:
    safe = [row for row in records if row["expected"] == "accept"]
    risky = [row for row in records if row["expected"] == "defer"]
    exposed = [row for row in records if row["candidate_exposed"]]
    model_calls = sum(row["primary_called"] for row in records)
    audit_calls = sum(row["audit_called"] for row in records)

    by_group: dict[str, dict[str, int]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["group"]].append(row)
    for group, items in sorted(grouped.items()):
        group_safe = [row for row in items if row["expected"] == "accept"]
        group_risky = [row for row in items if row["expected"] == "defer"]
        by_group[group] = {
            "total": len(items),
            "exposed": sum(row["candidate_exposed"] for row in items),
            "correct": sum(row["correct"] for row in items),
            "safe_accepts": sum(row["final_actual"] == "accept" for row in group_safe),
            "safe_total": len(group_safe),
            "unsafe_accepts": sum(row["final_actual"] == "accept" for row in group_risky),
            "unsafe_total": len(group_risky),
            "deferred": sum(row["final_actual"] == "defer" for row in items),
            "not_exposed": sum(row["final_actual"] == "not_exposed" for row in items),
        }

    primary_latencies = [
        row["primary_latency_seconds"]
        for row in records
        if row["primary_latency_seconds"] is not None
    ]
    audit_latencies = [
        row["audit_latency_seconds"]
        for row in records
        if row["audit_latency_seconds"] is not None
    ]

    case_decisions: dict[str, list[str]] = defaultdict(list)
    for row in records:
        case_decisions[row["id"]].append(row["final_actual"])
    stable_cases = sum(len(set(decisions)) == 1 for decisions in case_decisions.values())

    return {
        "case_runs": len(records),
        "exposed_case_runs": len(exposed),
        "model_calls": model_calls,
        "audit_calls": audit_calls,
        "correct_final_decisions": sum(row["correct"] for row in records),
        "safe_accepts": sum(row["final_actual"] == "accept" for row in safe),
        "safe_total": len(safe),
        "safe_deferred": sum(row["final_actual"] == "defer" for row in safe),
        "safe_not_exposed": sum(row["final_actual"] == "not_exposed" for row in safe),
        "unsafe_accepts": sum(row["final_actual"] == "accept" for row in risky),
        "unsafe_total": len(risky),
        "unsafe_deferred": sum(row["final_actual"] == "defer" for row in risky),
        "unsafe_screened": sum(row["final_actual"] == "not_exposed" for row in risky),
        "protocol_failures": sum(not row["protocol_ok"] for row in records),
        "model_errors": sum(row["final_actual"] == "model_error" for row in records),
        "stable_cases_across_runs": stable_cases,
        "unique_cases": len(case_decisions),
        "mean_primary_latency_seconds": (
            sum(primary_latencies) / len(primary_latencies)
            if primary_latencies else None
        ),
        "mean_audit_latency_seconds": (
            sum(audit_latencies) / len(audit_latencies)
            if audit_latencies else None
        ),
        "audit_verdict_counts": dict(
            Counter(
                row["audit_verdict"]
                for row in records
                if row["audit_verdict"] is not None
            )
        ),
        "by_group": by_group,
    }


def run_validation(
    provider: StructuredAIProvider | None,
    model: str | None,
    *,
    runs: int,
    modes: tuple[str, ...],
    output: Path,
    audit_only: bool = False,
) -> dict[str, Any]:
    if runs < 1:
        raise ValueError("--runs must be at least 1")
    unknown = sorted(set(modes) - set(MODES))
    if unknown:
        raise ValueError(f"Unknown modes: {', '.join(unknown)}")
    if not modes:
        raise ValueError("At least one mode is required")
    if not audit_only and (provider is None or not model):
        raise ValueError("A model and provider are required for a live run")

    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"Validation report already exists: {output}. Choose a new --output."
        )

    cases = frozen_validation_cases()
    corpus_sha = hashlib.sha256(
        json.dumps([asdict(case) for case in cases], sort_keys=True).encode("utf-8")
    ).hexdigest()

    with tempfile.TemporaryDirectory(prefix="frozen-semantic-validation-") as temporary:
        prepared = []
        for case in cases:
            folder = Path(temporary) / case.id
            folder.mkdir()
            prepared.append(_prepare_case(case, folder))

        coverage = []
        for item in prepared:
            candidate = item["candidate"]
            coverage.append(
                {
                    "id": item["case"].id,
                    "group": item["case"].group,
                    "expected": item["case"].expected,
                    "candidate_exposed": candidate is not None,
                    "coverage_reason": item["reason"],
                    "evidence_basis": (
                        candidate.context.evidence.get("basis")
                        if candidate is not None else None
                    ),
                }
            )

        if audit_only:
            payload = {
                "version": 1,
                "purpose": "Frozen 100-case semantic validation coverage audit",
                "audit_only": True,
                "corpus_sha256": corpus_sha,
                "unique_cases": len(cases),
                "safe_cases": sum(case.expected == "accept" for case in cases),
                "unsafe_cases": sum(case.expected == "defer" for case in cases),
                "exposed_cases": sum(row["candidate_exposed"] for row in coverage),
                "safe_exposed": sum(
                    row["candidate_exposed"] and row["expected"] == "accept"
                    for row in coverage
                ),
                "unsafe_exposed": sum(
                    row["candidate_exposed"] and row["expected"] == "defer"
                    for row in coverage
                ),
                "coverage_reasons": dict(
                    Counter(
                        row["coverage_reason"]
                        for row in coverage
                        if row["coverage_reason"]
                    )
                ),
                "by_group": {
                    group: {
                        "total": len(items),
                        "exposed": sum(row["candidate_exposed"] for row in items),
                    }
                    for group, items in sorted(
                        (
                            (group, [row for row in coverage if row["group"] == group])
                            for group in {row["group"] for row in coverage}
                        ),
                        key=lambda pair: pair[0],
                    )
                },
                "coverage": coverage,
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(
                f"Coverage: {payload['exposed_cases']}/{len(cases)}; "
                f"safe exposed {payload['safe_exposed']}/{payload['safe_cases']}; "
                f"unsafe exposed {payload['unsafe_exposed']}/{payload['unsafe_cases']}.",
                flush=True,
            )
            print(f"Report saved: {output}")
            return payload

        all_records: list[dict[str, Any]] = []
        for mode in modes:
            primary_variant = (
                "evidence_calibrated" if mode == "blind_veto" else mode
            )
            primary_system = _system_prompt(primary_variant)

            for run_index in range(1, runs + 1):
                for index, item in enumerate(prepared, start=1):
                    case = item["case"]
                    candidate = item["candidate"]
                    record: dict[str, Any] = {
                        "mode": mode,
                        "run": run_index,
                        "id": case.id,
                        "group": case.group,
                        "column": case.column,
                        "labels": [case.canonical, case.variant],
                        "expected": case.expected,
                        "context_supplied": bool(case.context),
                        "candidate_exposed": candidate is not None,
                        "coverage_reason": item["reason"],
                        "evidence_basis": (
                            candidate.context.evidence.get("basis")
                            if candidate is not None else None
                        ),
                        "primary_called": False,
                        "primary_actual": None,
                        "primary_rationale": None,
                        "primary_confidence": None,
                        "primary_latency_seconds": None,
                        "audit_called": False,
                        "audit_verdict": None,
                        "audit_rationale": None,
                        "audit_confidence": None,
                        "audit_latency_seconds": None,
                        "final_actual": "not_exposed",
                        "correct": False,
                        "protocol_ok": True,
                        "error": None,
                    }

                    if candidate is not None:
                        record["primary_called"] = True
                        started = time.perf_counter()
                        try:
                            response = provider.generate_structured(
                                model=model,
                                system_prompt=primary_system,
                                user_prompt=item["prompt"],
                                response_model=AIPlannerResponse,
                            )
                            scored = _score_reply(
                                response, candidate.id, case.expected
                            )
                            record["primary_actual"] = scored["actual"]
                            record["primary_rationale"] = scored["rationale"]
                            record["primary_confidence"] = scored["confidence"]
                            record["protocol_ok"] = scored["protocol_ok"]
                            record["final_actual"] = scored["actual"]
                        except Exception as exc:
                            record["protocol_ok"] = False
                            record["final_actual"] = "model_error"
                            record["error"] = f"primary {type(exc).__name__}: {exc}"
                        record["primary_latency_seconds"] = round(
                            time.perf_counter() - started, 3
                        )

                        if (
                            mode == "blind_veto"
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
                                        "Auditor returned an unexpected candidate ID"
                                    )
                                record["audit_verdict"] = audit.verdict
                                record["audit_rationale"] = audit.rationale
                                record["audit_confidence"] = audit.confidence
                                record["final_actual"] = (
                                    "accept"
                                    if audit.verdict == "same_typo"
                                    else "defer"
                                )
                            except Exception as exc:
                                record["protocol_ok"] = False
                                record["final_actual"] = "model_error"
                                record["error"] = (
                                    f"audit {type(exc).__name__}: {exc}"
                                )
                            record["audit_latency_seconds"] = round(
                                time.perf_counter() - started, 3
                            )

                    record["correct"] = _final_correct(
                        case.expected,
                        record["final_actual"],
                        record["protocol_ok"],
                    )
                    all_records.append(record)
                    print(
                        f"{mode} run {run_index}/{runs} "
                        f"case {index}/{len(cases)} {case.id}: "
                        f"{record['final_actual']}",
                        flush=True,
                    )

    summaries = {
        mode: _summarize_mode(
            [row for row in all_records if row["mode"] == mode]
        )
        for mode in modes
    }

    payload = {
        "version": 1,
        "purpose": "Frozen 100-case architecture validation; synthetic only; no edits",
        "audit_only": False,
        "corpus_sha256": corpus_sha,
        "model": model,
        "provider": provider.name,
        "runs": runs,
        "modes": list(modes),
        "unique_cases": len(cases),
        "safe_cases": sum(case.expected == "accept" for case in cases),
        "unsafe_cases": sum(case.expected == "defer" for case in cases),
        "coverage": {
            "exposed_cases": sum(row["candidate_exposed"] for row in coverage),
            "safe_exposed": sum(
                row["candidate_exposed"] and row["expected"] == "accept"
                for row in coverage
            ),
            "unsafe_exposed": sum(
                row["candidate_exposed"] and row["expected"] == "defer"
                for row in coverage
            ),
            "coverage_reasons": dict(
                Counter(
                    row["coverage_reason"]
                    for row in coverage
                    if row["coverage_reason"]
                )
            ),
        },
        "prompt_hashes": {
            "baseline": hashlib.sha256(
                _system_prompt("baseline").encode("utf-8")
            ).hexdigest(),
            "evidence_calibrated": hashlib.sha256(
                _system_prompt("evidence_calibrated").encode("utf-8")
            ).hexdigest(),
            "blind_auditor": hashlib.sha256(
                AUDITOR_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
        },
        "mode_summaries": summaries,
        "results": all_records,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\nFROZEN VALIDATION SUMMARY", flush=True)
    for mode in modes:
        summary = summaries[mode]
        print(
            f"{mode}: correct={summary['correct_final_decisions']}/"
            f"{summary['case_runs']}; "
            f"safe_accepts={summary['safe_accepts']}/{summary['safe_total']}; "
            f"unsafe_accepts={summary['unsafe_accepts']}/{summary['unsafe_total']}; "
            f"stable={summary['stable_cases_across_runs']}/"
            f"{summary['unique_cases']}",
            flush=True,
        )
    print(f"Report saved: {output}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Exact installed Ollama model; omit if only one")
    parser.add_argument("--url", help="Ollama URL; defaults to local")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--think", action="store_true")
    parser.add_argument(
        "--modes",
        default="baseline,evidence_calibrated,blind_veto",
        help="Comma-separated frozen modes; default evaluates all before inspection",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit production candidate coverage without calling Ollama",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    modes = tuple(part.strip() for part in args.modes.split(",") if part.strip())
    provider = None if args.audit_only else OllamaProvider(args.url, think=args.think)
    model = None if args.audit_only else resolve_model(provider, args.model)
    run_validation(
        provider,
        model,
        runs=args.runs,
        modes=modes,
        output=args.output,
        audit_only=args.audit_only,
    )


if __name__ == "__main__":
    main()
