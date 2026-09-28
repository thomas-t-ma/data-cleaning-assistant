"""Local-only real-workflow pilot harness for v0.8+.

This harness does not apply data edits. It captures a privacy-conscious trace of
candidate decisions, then scores those decisions against a later human review.

Typical workflow:
    python examples/workflow_pilot.py start --source DATA.xlsx --out pilot_outputs/workflow_01 --model qwen3.5:4b
    data-cleaner review DATA.xlsx pilot_outputs/workflow_01/ai_plan.json -o pilot_outputs/workflow_01/reviewed_plan.json
    python examples/workflow_pilot.py score --pilot pilot_outputs/workflow_01 --reviewed-plan pilot_outputs/workflow_01/reviewed_plan.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from data_cleaning_assistant.actions import ActionDecision, CleaningPlan
from data_cleaning_assistant.ai_planner import (
    _action_is_covered_by_existing,
    propose_ai_plan,
    resolve_model,
)
from data_cleaning_assistant.ai_provider import OllamaProvider, StructuredAIProvider
from data_cleaning_assistant.engine import profile_dataset
from data_cleaning_assistant.planner import file_sha256
from data_cleaning_assistant.study_schema import StudySchema, load_study_schema

MANIFEST = "pilot_manifest.json"
PROFILE = "profile.json"
CONTEXT = "ai_context.json"
PLAN = "ai_plan.json"
TRACE = "decision_trace.json"
SCORE = "workflow_score.json"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _source_descriptor(source: Path) -> dict[str, Any]:
    return {
        "name": source.name,
        "suffix": source.suffix.lower(),
        "sha256": file_sha256(source),
    }


def _schema_descriptor(schema_path: Path | None) -> dict[str, Any] | None:
    if schema_path is None:
        return None
    schema_path = schema_path.expanduser().resolve()
    return {
        "name": schema_path.name,
        "sha256": file_sha256(schema_path),
    }


def _selection_by_id(result) -> dict[str, Any]:
    return {selection.candidate_id: selection for selection in result.response.selections}


def _trace_payload(result) -> dict[str, Any]:
    selections = _selection_by_id(result)
    rows: list[dict[str, Any]] = []
    for candidate in result.candidates:
        selection = selections.get(candidate.id)
        covering = [
            action.id
            for action in result.plan.actions
            if _action_is_covered_by_existing(candidate.action, action)
        ]
        rows.append(
            {
                "candidate_id": candidate.id,
                "type": candidate.action.type.value,
                "table": candidate.action.table,
                "column": candidate.action.column,
                "parameters": candidate.action.parameters,
                "basis": candidate.context.evidence.get("basis"),
                "question": candidate.context.question,
                "decision": selection.decision.value if selection else "missing",
                "rationale": selection.rationale if selection else "No consensus selection",
                "confidence": selection.confidence if selection else None,
                "covered_by_plan_action_ids": covering,
            }
        )

    return {
        "version": 1,
        "purpose": "Local real-workflow pilot decision trace; no data edits",
        "provider": result.provider,
        "model": result.model,
        "review_call_count": result.review_call_count,
        "candidate_disagreement_count": result.consensus_disagreement_count,
        "covered_ai_action_count": result.covered_ai_action_count,
        "model_summary": result.response.summary,
        "candidate_count": len(result.candidates),
        "candidates": rows,
    }


def start_pilot(
    source: Path,
    out: Path,
    *,
    provider: StructuredAIProvider,
    model: str,
    schema: StudySchema | None = None,
    schema_path: Path | None = None,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    out = out.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if out.exists():
        raise FileExistsError(
            f"Pilot output already exists: {out}. Choose a fresh directory to preserve history."
        )

    out.mkdir(parents=True, exist_ok=False)
    report = profile_dataset(source, schema=schema)
    result = propose_ai_plan(
        source,
        provider=provider,
        model=model,
        include_rules=True,
        schema=schema,
    )

    _write_json(out / PROFILE, report.model_dump(mode="json"))
    (out / CONTEXT).write_text(result.context.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (out / PLAN).write_text(result.plan.model_dump_json(indent=2) + "\n", encoding="utf-8")
    trace = _trace_payload(result)
    _write_json(out / TRACE, trace)

    manifest = {
        "version": 1,
        "purpose": "Local real-workflow usefulness pilot; source is never modified",
        "source": _source_descriptor(source),
        # The absolute path is local-only convenience for resuming the pilot.
        # pilot_outputs is gitignored; do not publish this manifest if the path is sensitive.
        "source_path": str(source),
        "schema": _schema_descriptor(schema_path),
        "model": model,
        "provider": provider.name,
        "files": {
            "profile": PROFILE,
            "ai_context": CONTEXT,
            "ai_plan": PLAN,
            "decision_trace": TRACE,
        },
        "no_edits_applied": True,
    }
    _write_json(out / MANIFEST, manifest)

    print("REAL-WORKFLOW PILOT STARTED")
    print(f"Source hash: {manifest['source']['sha256']}")
    print(f"Profile issues: {len(report.issues)}")
    print(f"Plan actions: {len(result.plan.actions)}")
    print(f"AI candidates: {trace['candidate_count']}")
    print(
        "Consensus: "
        f"{result.review_call_count} model call(s), "
        f"{result.consensus_disagreement_count} candidate disagreement(s)"
    )
    print(f"AI actions already covered by deterministic plan: {result.covered_ai_action_count}")
    print(f"Deferred AI candidates: {len(result.plan.deferred_reviews)}")
    print("No edits were applied.")
    print(f"Pilot directory: {out}")
    return {"manifest": manifest, "trace": trace, "plan": result.plan}


def _human_action_status(reviewed: CleaningPlan, action_ids: list[str]) -> str:
    actions = {action.id: action for action in reviewed.actions}
    matched = [actions[action_id] for action_id in action_ids if action_id in actions]
    if not matched:
        return "unresolved"
    if any(action.decision == ActionDecision.APPROVED for action in matched):
        return "approved"
    if all(action.decision == ActionDecision.REJECTED for action in matched):
        return "rejected"
    return "unresolved"


def _human_candidate_status(
    candidate: dict[str, Any],
    reviewed: CleaningPlan,
) -> str:
    if candidate["decision"] == "defer":
        deferred = next(
            (
                item
                for item in reviewed.deferred_reviews
                if item.candidate_id == candidate["candidate_id"]
            ),
            None,
        )
        if deferred is None:
            return "unresolved"
        if deferred.status == "approved":
            return "approved"
        if deferred.status == "rejected":
            return "rejected"
        return "unresolved"

    if candidate["decision"] == "accept":
        return _human_action_status(
            reviewed,
            list(candidate.get("covered_by_plan_action_ids", [])),
        )
    return "unresolved"


def score_pilot(
    pilot: Path,
    reviewed_plan: Path,
    *,
    review_minutes: float | None = None,
    missed_issues: list[str] | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    pilot = pilot.expanduser().resolve()
    reviewed_plan = reviewed_plan.expanduser().resolve()
    manifest = json.loads((pilot / MANIFEST).read_text(encoding="utf-8"))
    trace = json.loads((pilot / TRACE).read_text(encoding="utf-8"))
    reviewed = CleaningPlan.model_validate_json(reviewed_plan.read_text(encoding="utf-8"))

    source = Path(manifest["source_path"])
    expected_hash = manifest["source"]["sha256"]
    if not source.is_file() or file_sha256(source) != expected_hash:
        raise ValueError(
            "Pilot source is missing or changed since the trace was created. "
            "Do not score against a different dataset version."
        )
    if reviewed.source_sha256 != expected_hash:
        raise ValueError("Reviewed plan is not bound to the pilot source hash.")

    rows: list[dict[str, Any]] = []
    for candidate in trace["candidates"]:
        human = _human_candidate_status(candidate, reviewed)
        model = candidate["decision"]
        if human == "unresolved":
            outcome = "unresolved"
        elif model == "accept" and human == "approved":
            outcome = "agree_accept"
        elif model == "accept" and human == "rejected":
            outcome = "human_rejected_ai_accept"
        elif model == "defer" and human == "rejected":
            outcome = "agree_defer"
        elif model == "defer" and human == "approved":
            outcome = "human_approved_ai_defer"
        else:
            outcome = "unresolved"
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "type": candidate["type"],
                "table": candidate["table"],
                "column": candidate["column"],
                "basis": candidate.get("basis"),
                "ai_decision": model,
                "human_decision": human,
                "outcome": outcome,
            }
        )

    resolved = [row for row in rows if row["outcome"] != "unresolved"]
    agreement = [row for row in resolved if row["outcome"] in {"agree_accept", "agree_defer"}]
    overreach = [row for row in rows if row["outcome"] == "human_rejected_ai_accept"]
    overcaution = [row for row in rows if row["outcome"] == "human_approved_ai_defer"]

    action_counts = {
        "approved": sum(a.decision == ActionDecision.APPROVED for a in reviewed.actions),
        "rejected": sum(a.decision == ActionDecision.REJECTED for a in reviewed.actions),
        "unresolved": sum(a.decision == ActionDecision.PROPOSED for a in reviewed.actions),
        "total": len(reviewed.actions),
    }
    deferred_counts = {
        "approved": sum(item.status == "approved" for item in reviewed.deferred_reviews),
        "rejected": sum(item.status == "rejected" for item in reviewed.deferred_reviews),
        "unresolved": sum(item.status == "pending" for item in reviewed.deferred_reviews),
        "total": len(reviewed.deferred_reviews),
    }

    payload = {
        "version": 1,
        "purpose": "Human-vs-AI workflow usefulness score; human review is not ground truth",
        "source_sha256": expected_hash,
        "model": trace["model"],
        "provider": trace["provider"],
        "candidate_count": len(rows),
        "resolved_candidate_reviews": len(resolved),
        "unresolved_candidate_reviews": len(rows) - len(resolved),
        "human_ai_agreement_count": len(agreement),
        "human_ai_agreement_rate_resolved": (
            len(agreement) / len(resolved) if resolved else None
        ),
        "human_rejected_ai_accept_count": len(overreach),
        "human_approved_ai_defer_count": len(overcaution),
        "review_call_count": trace["review_call_count"],
        "candidate_disagreement_count": trace["candidate_disagreement_count"],
        "covered_ai_action_count": trace["covered_ai_action_count"],
        "plan_action_review": action_counts,
        "deferred_candidate_review": deferred_counts,
        "review_minutes": review_minutes,
        "missed_issues_reported_by_reviewer": list(missed_issues or []),
        "candidate_outcomes": rows,
        "interpretation_note": (
            "This report measures agreement with one human review and workflow burden. "
            "It does not establish scientific correctness, clinical safety, or recall of "
            "issues the current detectors never surfaced."
        ),
    }

    output = (
        output.expanduser().resolve()
        if output is not None
        else pilot / SCORE
    )
    if output.exists():
        raise FileExistsError(
            f"Workflow score already exists: {output}. Choose a fresh --output to preserve history."
        )
    _write_json(output, payload)

    print("WORKFLOW PILOT SCORE")
    print(
        f"Candidate decisions resolved: {len(resolved)}/{len(rows)}; "
        f"human/AI agreement: {len(agreement)}/{len(resolved) if resolved else 0}"
    )
    print(f"Human rejected AI accepts: {len(overreach)}")
    print(f"Human approved AI deferrals: {len(overcaution)}")
    print(
        "Plan actions reviewed: "
        f"{action_counts['approved']} approved, "
        f"{action_counts['rejected']} rejected, "
        f"{action_counts['unresolved']} unresolved"
    )
    print(
        "Deferred AI reviews: "
        f"{deferred_counts['approved']} approved, "
        f"{deferred_counts['rejected']} rejected, "
        f"{deferred_counts['unresolved']} unresolved"
    )
    if review_minutes is not None:
        print(f"Human review time: {review_minutes:.1f} minute(s)")
    print(f"Reviewer-reported missed issues: {len(missed_issues or [])}")
    print("Human review is not treated as scientific ground truth.")
    print(f"Score written to: {output}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    starter = commands.add_parser("start", help="Create a fresh read-only workflow pilot")
    starter.add_argument("--source", type=Path, required=True)
    starter.add_argument("--out", type=Path, required=True)
    starter.add_argument("--schema", type=Path)
    starter.add_argument("--model", help="Exact installed Ollama model; omit if only one")
    starter.add_argument("--url", help="Ollama base URL; defaults to local")
    starter.add_argument("--think", action="store_true")

    scorer = commands.add_parser("score", help="Score AI decisions against a reviewed plan")
    scorer.add_argument("--pilot", type=Path, required=True)
    scorer.add_argument("--reviewed-plan", type=Path, required=True)
    scorer.add_argument("--review-minutes", type=float)
    scorer.add_argument(
        "--missed-issue",
        action="append",
        default=[],
        help="Reviewer-observed issue the system did not surface; repeat as needed",
    )
    scorer.add_argument("--output", type=Path)

    args = parser.parse_args()
    if args.command == "start":
        schema = load_study_schema(args.schema) if args.schema else None
        provider = OllamaProvider(args.url, think=args.think)
        model = resolve_model(provider, args.model)
        start_pilot(
            args.source,
            args.out,
            provider=provider,
            model=model,
            schema=schema,
            schema_path=args.schema,
        )
    else:
        score_pilot(
            args.pilot,
            args.reviewed_plan,
            review_minutes=args.review_minutes,
            missed_issues=args.missed_issue,
            output=args.output,
        )


if __name__ == "__main__":
    main()
