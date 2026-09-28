"""Fresh one-shot validation of adaptive consensus for semantic review.

Policy frozen before first live run:
1. Run the existing evidence-calibrated reviewer twice.
2. If both valid decisions agree, use that decision.
3. If they disagree, run one third identical review and take 2-of-3 majority.
4. Any protocol/model error fails closed to DEFER/human review.

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
from data_cleaning_assistant.semantic_consensus_validation import adaptive_consensus_cases

from semantic_stress_test import _prepare_case, _score_reply, _system_prompt


def _review_once(
    provider: StructuredAIProvider,
    model: str,
    *,
    system_prompt: str,
    user_prompt: str,
    candidate_id: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = provider.generate_structured(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=AIPlannerResponse,
        )
        scored = _score_reply(response, candidate_id, expected="defer")
        # expected above is irrelevant here; only actual/protocol fields are used.
        return {
            "decision": scored["actual"],
            "protocol_ok": scored["protocol_ok"],
            "rationale": scored["rationale"],
            "confidence": scored["confidence"],
            "latency_seconds": round(time.perf_counter() - started, 3),
            "error": None,
        }
    except Exception as exc:
        return {
            "decision": "model_error",
            "protocol_ok": False,
            "rationale": "",
            "confidence": None,
            "latency_seconds": round(time.perf_counter() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _consensus_from_votes(votes: list[dict[str, Any]]) -> tuple[str, str]:
    if any(not vote["protocol_ok"] for vote in votes):
        return "defer", "fail_closed_protocol_or_model_error"
    decisions = [vote["decision"] for vote in votes]
    accepts = decisions.count("accept")
    defers = decisions.count("defer")
    if accepts > defers:
        return "accept", f"majority_{accepts}_{defers}"
    return "defer", f"majority_{accepts}_{defers}"


def _summarize(records: list[dict[str, Any]], decision_field: str) -> dict[str, Any]:
    safe = [row for row in records if row["expected"] == "accept"]
    risky = [row for row in records if row["expected"] == "defer"]
    return {
        "correct": sum(
            row[decision_field] == row["expected"]
            for row in records
        ),
        "total": len(records),
        "safe_accepts": sum(row[decision_field] == "accept" for row in safe),
        "safe_total": len(safe),
        "unsafe_accepts": sum(row[decision_field] == "accept" for row in risky),
        "unsafe_total": len(risky),
        "deferred": sum(row[decision_field] == "defer" for row in records),
    }


def run_consensus_validation(
    provider: StructuredAIProvider | None,
    model: str | None,
    *,
    output: Path,
    audit_only: bool = False,
) -> dict[str, Any]:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"Validation report already exists: {output}. Choose a new --output."
        )
    cases = adaptive_consensus_cases()
    corpus_sha = hashlib.sha256(
        json.dumps([asdict(case) for case in cases], sort_keys=True).encode("utf-8")
    ).hexdigest()

    with tempfile.TemporaryDirectory(prefix="adaptive-consensus-") as temporary:
        prepared = []
        for case in cases:
            folder = Path(temporary) / case.id
            folder.mkdir()
            prepared.append(_prepare_case(case, folder))

        coverage = [
            {
                "id": item["case"].id,
                "group": item["case"].group,
                "expected": item["case"].expected,
                "candidate_exposed": item["candidate"] is not None,
                "coverage_reason": item["reason"],
                "evidence_basis": (
                    item["candidate"].context.evidence.get("basis")
                    if item["candidate"] is not None else None
                ),
            }
            for item in prepared
        ]

        if audit_only:
            payload = {
                "version": 1,
                "purpose": "Fresh adaptive-consensus validation coverage audit",
                "audit_only": True,
                "corpus_sha256": corpus_sha,
                "unique_cases": len(cases),
                "safe_cases": sum(c.expected == "accept" for c in cases),
                "unsafe_cases": sum(c.expected == "defer" for c in cases),
                "exposed_cases": sum(x["candidate_exposed"] for x in coverage),
                "safe_exposed": sum(
                    x["candidate_exposed"] and x["expected"] == "accept"
                    for x in coverage
                ),
                "unsafe_exposed": sum(
                    x["candidate_exposed"] and x["expected"] == "defer"
                    for x in coverage
                ),
                "coverage_reasons": dict(
                    Counter(
                        x["coverage_reason"] for x in coverage
                        if x["coverage_reason"]
                    )
                ),
                "coverage": coverage,
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(
                f"Coverage: {payload['exposed_cases']}/{len(cases)}; "
                f"safe={payload['safe_exposed']}/{payload['safe_cases']}; "
                f"unsafe={payload['unsafe_exposed']}/{payload['unsafe_cases']}.",
                flush=True,
            )
            print(f"Report saved: {output}", flush=True)
            return payload

        if provider is None or not model:
            raise ValueError("A provider/model is required for a live validation run")

        system_prompt = _system_prompt("evidence_calibrated")
        records: list[dict[str, Any]] = []

        for index, item in enumerate(prepared, start=1):
            case = item["case"]
            candidate = item["candidate"]
            record: dict[str, Any] = {
                "id": case.id,
                "group": case.group,
                "column": case.column,
                "labels": [case.canonical, case.variant],
                "expected": case.expected,
                "candidate_exposed": candidate is not None,
                "coverage_reason": item["reason"],
                "evidence_basis": (
                    candidate.context.evidence.get("basis")
                    if candidate is not None else None
                ),
                "votes": [],
                "third_call_used": False,
                "first_vote_decision": "not_exposed",
                "consensus_decision": "defer",
                "consensus_reason": "safe_case_not_exposed",
            }

            if candidate is None:
                if case.expected == "defer":
                    record["consensus_reason"] = "unsafe_case_screened_deterministically"
                records.append(record)
                continue

            vote1 = _review_once(
                provider,
                model,
                system_prompt=system_prompt,
                user_prompt=item["prompt"],
                candidate_id=candidate.id,
            )
            vote2 = _review_once(
                provider,
                model,
                system_prompt=system_prompt,
                user_prompt=item["prompt"],
                candidate_id=candidate.id,
            )
            record["votes"] = [vote1, vote2]
            record["first_vote_decision"] = (
                vote1["decision"] if vote1["protocol_ok"] else "defer"
            )

            if not vote1["protocol_ok"] or not vote2["protocol_ok"]:
                record["consensus_decision"] = "defer"
                record["consensus_reason"] = "fail_closed_protocol_or_model_error"
            elif vote1["decision"] == vote2["decision"]:
                record["consensus_decision"] = vote1["decision"]
                record["consensus_reason"] = "two_vote_agreement"
            else:
                vote3 = _review_once(
                    provider,
                    model,
                    system_prompt=system_prompt,
                    user_prompt=item["prompt"],
                    candidate_id=candidate.id,
                )
                record["votes"].append(vote3)
                record["third_call_used"] = True
                final, reason = _consensus_from_votes(record["votes"])
                record["consensus_decision"] = final
                record["consensus_reason"] = reason

            records.append(record)
            print(
                f"case {index}/{len(cases)} {case.id}: "
                f"votes={[v['decision'] for v in record['votes']]} "
                f"final={record['consensus_decision']}",
                flush=True,
            )

    first_vote_summary = _summarize(records, "first_vote_decision")
    consensus_summary = _summarize(records, "consensus_decision")
    total_calls = sum(len(row["votes"]) for row in records)
    third_calls = sum(row["third_call_used"] for row in records)
    protocol_failures = sum(
        not vote["protocol_ok"]
        for row in records
        for vote in row["votes"]
    )
    latencies = [
        vote["latency_seconds"]
        for row in records
        for vote in row["votes"]
        if vote["latency_seconds"] is not None
    ]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["group"]].append(row)
    by_group = {}
    for group, items in sorted(grouped.items()):
        safe = [x for x in items if x["expected"] == "accept"]
        risky = [x for x in items if x["expected"] == "defer"]
        by_group[group] = {
            "total": len(items),
            "safe_accepts": sum(x["consensus_decision"] == "accept" for x in safe),
            "safe_total": len(safe),
            "unsafe_accepts": sum(x["consensus_decision"] == "accept" for x in risky),
            "unsafe_total": len(risky),
            "third_calls": sum(x["third_call_used"] for x in items),
        }

    payload = {
        "version": 1,
        "purpose": "Fresh frozen adaptive consensus validation; synthetic only; no edits",
        "audit_only": False,
        "corpus_sha256": corpus_sha,
        "model": model,
        "provider": provider.name,
        "review_policy": {
            "prompt_variant": "evidence_calibrated",
            "first_reviews": 2,
            "third_review_on_disagreement": True,
            "final_rule": "majority_vote",
            "error_policy": "fail_closed_to_defer",
        },
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "unique_cases": len(cases),
        "safe_cases": sum(c.expected == "accept" for c in cases),
        "unsafe_cases": sum(c.expected == "defer" for c in cases),
        "coverage": {
            "exposed_cases": sum(x["candidate_exposed"] for x in coverage),
            "safe_exposed": sum(
                x["candidate_exposed"] and x["expected"] == "accept"
                for x in coverage
            ),
            "unsafe_exposed": sum(
                x["candidate_exposed"] and x["expected"] == "defer"
                for x in coverage
            ),
            "coverage_reasons": dict(
                Counter(
                    x["coverage_reason"] for x in coverage
                    if x["coverage_reason"]
                )
            ),
        },
        "first_vote_summary": first_vote_summary,
        "consensus_summary": consensus_summary,
        "total_model_calls": total_calls,
        "third_calls": third_calls,
        "protocol_failures": protocol_failures,
        "mean_call_latency_seconds": (
            sum(latencies) / len(latencies) if latencies else None
        ),
        "by_group": by_group,
        "results": records,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\nADAPTIVE CONSENSUS SUMMARY", flush=True)
    print(
        f"first vote: correct={first_vote_summary['correct']}/{len(cases)}, "
        f"safe={first_vote_summary['safe_accepts']}/{first_vote_summary['safe_total']}, "
        f"unsafe={first_vote_summary['unsafe_accepts']}/{first_vote_summary['unsafe_total']}",
        flush=True,
    )
    print(
        f"consensus: correct={consensus_summary['correct']}/{len(cases)}, "
        f"safe={consensus_summary['safe_accepts']}/{consensus_summary['safe_total']}, "
        f"unsafe={consensus_summary['unsafe_accepts']}/{consensus_summary['unsafe_total']}; "
        f"calls={total_calls}, third_calls={third_calls}",
        flush=True,
    )
    print(f"Report saved: {output}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Exact installed Ollama model; omit if only one")
    parser.add_argument("--url", help="Ollama URL; defaults to local")
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    provider = None if args.audit_only else OllamaProvider(args.url, think=args.think)
    model = None if args.audit_only else resolve_model(provider, args.model)
    run_consensus_validation(
        provider,
        model,
        output=args.output,
        audit_only=args.audit_only,
    )


if __name__ == "__main__":
    main()
