"""Summarize a v0.9 plan/review for public workflow-pilot comparison."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from data_cleaning_assistant.actions import ActionDecision, CleaningPlan


def summarize(plan_path: Path, context_path: Path | None = None) -> dict:
    plan = CleaningPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    type_counts = Counter(action.type.value for action in plan.actions)
    origin_counts = Counter(action.origin.value for action in plan.actions)
    decision_counts = Counter(action.decision.value for action in plan.actions)

    candidate_count = None
    if context_path is not None:
        context = json.loads(context_path.read_text(encoding="utf-8"))
        candidate_count = len(context.get("candidates", []))

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
        "semantic_ai_context_candidates": candidate_count,
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
    if candidate_count is not None:
        print(f"Semantic AI context candidates: {candidate_count}")
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
