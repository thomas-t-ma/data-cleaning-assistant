"""Summarize a v0.9 plan/review for public workflow-pilot comparison."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from data_cleaning_assistant.actions import CleaningPlan


def summarize(plan_path: Path, context_path: Path | None = None) -> dict:
    plan = CleaningPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    type_counts = Counter(action.type.value for action in plan.actions)
    origin_counts = Counter(action.origin.value for action in plan.actions)
    decision_counts = Counter(action.decision.value for action in plan.actions)

    candidate_count = None
    review_batch_count = None
    candidate_types = Counter()
    candidate_bases = Counter()
    candidate_targets = Counter()
    candidate_rows = []
    if context_path is not None:
        context_payload = json.loads(context_path.read_text(encoding="utf-8"))
        if "aggregate_context" in context_payload:
            context = context_payload["aggregate_context"]
            review_batch_count = len(context_payload.get("review_batches", []))
        else:
            context = context_payload
            review_batch_count = 1 if context.get("candidates") else 0
        candidates = context.get("candidates", [])
        candidate_count = len(candidates)
        for candidate in candidates:
            candidate_type = str(candidate.get("type", "unknown"))
            candidate_types[candidate_type] += 1
            evidence = candidate.get("evidence") or {}
            basis = str(evidence.get("basis") or "unspecified")
            candidate_bases[basis] += 1
            target = (
                f"{candidate.get('table', '?')}.{candidate.get('column') or '(table)'}"
            )
            candidate_targets[target] += 1
            candidate_rows.append(
                {
                    "id": candidate.get("id"),
                    "type": candidate_type,
                    "target": target,
                    "basis": basis,
                    "proposed_change": candidate.get("proposed_change"),
                    "evidence": evidence,
                    "question": candidate.get("question"),
                }
            )

    deferred_bases = Counter()
    deferred_targets = Counter()
    deferred_examples = []
    for item in plan.deferred_reviews:
        evidence = item.evidence or {}
        basis = str(evidence.get("basis") or "unspecified")
        deferred_bases[basis] += 1
        action = item.proposed_action
        target = (
            f"{action.table}.{action.column}"
            if action.column is not None
            else action.table
        )
        deferred_targets[target] += 1
        deferred_examples.append(
            {
                "candidate_id": item.candidate_id,
                "target": target,
                "basis": basis,
                "parameters": action.parameters,
                "confidence": item.confidence,
                "rationale": item.rationale,
            }
        )

    review = plan.review_session
    event_scopes = Counter()
    event_decisions = Counter()
    family_events = []
    if review is not None:
        for event in review.events:
            event_scopes[event.scope] += 1
            event_decisions[event.decision] += 1
            if event.scope == "family":
                family_events.append(
                    {
                        "action_type": (
                            event.action_type.value if event.action_type is not None else None
                        ),
                        "member_count": event.member_count,
                        "affected_count": event.affected_count,
                        "decision": event.decision,
                        "bulk_confirmed": event.bulk_confirmed,
                        "duration_seconds": event.duration_seconds,
                    }
                )

    payload = {
        "actions": len(plan.actions),
        "action_types": dict(sorted(type_counts.items())),
        "action_origins": dict(sorted(origin_counts.items())),
        "action_decisions": dict(sorted(decision_counts.items())),
        "deferred_ai_reviews": len(plan.deferred_reviews),
        "deferred_ai_bases": dict(sorted(deferred_bases.items())),
        "top_deferred_ai_targets": [
            {"target": target, "count": count}
            for target, count in deferred_targets.most_common(20)
        ],
        "deferred_ai_examples": deferred_examples[:50],
        "semantic_ai_context_candidates": candidate_count,
        "semantic_review_batches": review_batch_count,
        "candidate_types": dict(sorted(candidate_types.items())),
        "candidate_bases": dict(sorted(candidate_bases.items())),
        "top_candidate_targets": [
            {"target": target, "count": count}
            for target, count in candidate_targets.most_common(20)
        ],
        "candidate_examples": candidate_rows[:50],
        "review_duration_seconds": (
            review.duration_seconds if review is not None else None
        ),
        "review_event_scopes": dict(sorted(event_scopes.items())),
        "review_event_decisions": dict(sorted(event_decisions.items())),
        "family_events": family_events,
    }

    print("V0.9 WORKFLOW SUMMARY")
    print(f"Actions: {payload['actions']}")
    print(f"Deferred AI reviews: {payload['deferred_ai_reviews']}")
    if payload["deferred_ai_reviews"]:
        print("Deferred AI evidence bases:")
        for name, count in payload["deferred_ai_bases"].items():
            print(f"  {name}: {count}")
        print("Deferred AI examples:")
        for item in payload["deferred_ai_examples"][:15]:
            print(
                f"  {item['candidate_id']} | {item['target']} | "
                f"{item['basis']} | {item['parameters']} | "
                f"confidence={item['confidence']:.2f}"
            )
            print(f"    rationale: {item['rationale']}")
    if candidate_count is not None:
        print(f"Semantic AI context candidates: {candidate_count}")
        print(f"Semantic review batches: {review_batch_count}")
        print("Candidate evidence bases:")
        for name, count in payload["candidate_bases"].items():
            print(f"  {name}: {count}")
        print("Top semantic candidate targets:")
        for item in payload["top_candidate_targets"]:
            print(f"  {item['target']}: {item['count']}")
        print("Candidate examples:")
        for candidate in payload["candidate_examples"][:20]:
            print(
                f"  {candidate['id']} | {candidate['target']} | "
                f"{candidate['basis']} | {candidate['proposed_change']}"
            )
    print("Action types:")
    for name, count in payload["action_types"].items():
        print(f"  {name}: {count}")
    print("Action origins:")
    for name, count in payload["action_origins"].items():
        print(f"  {name}: {count}")
    print("Action decisions:")
    for name, count in payload["action_decisions"].items():
        print(f"  {name}: {count}")

    if review is not None:
        print(f"Review duration: {review.duration_seconds:.1f}s")
        print("Review event scopes:")
        for name, count in payload["review_event_scopes"].items():
            print(f"  {name}: {count}")
        print("Family decisions:")
        for event in family_events:
            print(
                "  "
                f"{event['action_type']}: {event['member_count']} action(s), "
                f"{event['affected_count']} affected, "
                f"decision={event['decision']}, "
                f"bulk_confirmed={event['bulk_confirmed']}"
            )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = summarize(args.plan, args.context)
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(
                f"Output already exists: {args.output}. Choose a fresh path."
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Summary written to: {args.output}")


if __name__ == "__main__":
    main()
