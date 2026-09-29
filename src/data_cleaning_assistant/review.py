from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .actions import CleaningAction, CleaningPlan, ReviewEvent, ReviewSession
from .transforms import preview_action


@dataclass
class ActionFamilyMember:
    action: CleaningAction
    affected_count: int


@dataclass
class ActionFamily:
    key: str
    origin: str
    action_type: str
    parameters: dict[str, Any]
    members: list[ActionFamilyMember]

    @property
    def affected_count(self) -> int:
        return sum(member.affected_count for member in self.members)

    @property
    def action_ids(self) -> list[str]:
        return [member.action.id for member in self.members]

    @property
    def targets(self) -> list[str]:
        return [action_target(member.action) for member in self.members]


def action_target(action: CleaningAction) -> str:
    if action.column is None:
        return action.table
    return f"{action.table}.{action.column}"


def group_actions(
    tables,
    plan: CleaningPlan,
) -> list[ActionFamily]:
    """Group exact mechanical transformations for compact human review."""
    groups: dict[str, ActionFamily] = {}
    for action in plan.actions:
        params = json.dumps(action.parameters, sort_keys=True, default=str)
        key = f"{action.origin.value}|{action.type.value}|{params}"
        family = groups.get(key)
        if family is None:
            family = ActionFamily(
                key=key,
                origin=action.origin.value,
                action_type=action.type.value,
                parameters=action.parameters,
                members=[],
            )
            groups[key] = family
        preview = preview_action(tables, action, sample_limit=1)
        family.members.append(
            ActionFamilyMember(action=action, affected_count=preview.affected_count)
        )

    families = list(groups.values())
    for family in families:
        family.members.sort(
            key=lambda member: (-member.affected_count, action_target(member.action))
        )
    families.sort(
        key=lambda family: (
            -len(family.members),
            family.action_type,
            str(family.parameters),
        )
    )
    return families


def begin_review_session(plan: CleaningPlan) -> float:
    """Start a new review session and return a perf-counter timestamp."""
    plan.review_session = ReviewSession(started_at=datetime.now(UTC))
    return time.perf_counter()


def record_review_event(
    plan: CleaningPlan,
    *,
    scope: str,
    actions: list[CleaningAction],
    affected_count: int,
    decision: str,
    duration_seconds: float,
    bulk_confirmed: bool = False,
    targets_preview: list[str] | None = None,
) -> None:
    if plan.review_session is None:
        raise ValueError("Review session has not been started.")
    first = actions[0] if actions else None
    plan.review_session.events.append(
        ReviewEvent(
            scope=scope,
            action_ids=[action.id for action in actions],
            action_type=first.type if first is not None else None,
            origin=first.origin if first is not None else None,
            parameters=first.parameters if first is not None else {},
            member_count=len(actions),
            affected_count=affected_count,
            decision=decision,
            bulk_confirmed=bulk_confirmed,
            duration_seconds=max(0.0, duration_seconds),
            targets_preview=list(targets_preview or [])[:20],
        )
    )


def finish_review_session(plan: CleaningPlan, started_perf: float) -> None:
    if plan.review_session is None:
        return
    plan.review_session.completed_at = datetime.now(UTC)
    plan.review_session.duration_seconds = max(0.0, time.perf_counter() - started_perf)
