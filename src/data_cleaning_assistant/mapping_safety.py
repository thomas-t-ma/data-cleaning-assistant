from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable

from .actions import ActionDecision, ActionType, CleaningAction, CleaningPlan


@dataclass(frozen=True)
class MappingConflict:
    """One source category proposed to map to multiple canonical targets."""

    table: str
    column: str | None
    source: str
    targets: tuple[str, ...]
    action_ids: tuple[str, ...]


def _mapping_items(action: CleaningAction) -> list[tuple[object, object]]:
    if action.type != ActionType.NORMALIZE_CATEGORIES:
        return []
    mapping = action.parameters.get("mapping")
    if not isinstance(mapping, dict):
        return []
    return list(mapping.items())


def _mapping_action_id(
    action: CleaningAction,
    source: object,
    target: object,
) -> str:
    payload = json.dumps(
        {
            "parent_action_id": action.parent_action_id or action.id,
            "table": action.table,
            "column": action.column,
            "source": source,
            "target": target,
        },
        sort_keys=True,
        default=str,
    )
    suffix = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{action.id}__map_{suffix}"


def atomize_category_mappings(plan: CleaningPlan) -> tuple[int, int]:
    """Split multi-mapping category actions into independently reviewable actions.

    Returns:
        A tuple of (source actions split, additional actions created).
    """

    expanded: list[CleaningAction] = []
    split_actions = 0

    for action in plan.actions:
        items = _mapping_items(action)
        if len(items) <= 1:
            expanded.append(action)
            continue

        split_actions += 1
        parent_id = action.parent_action_id or action.id

        for source, target in items:
            parameters = deepcopy(action.parameters)
            parameters["mapping"] = {source: target}

            child = action.model_copy(deep=True)
            child.id = _mapping_action_id(action, source, target)
            child.parent_action_id = parent_id
            child.parameters = parameters
            expanded.append(child)

    original_count = len(plan.actions)
    plan.actions = expanded
    return split_actions, len(expanded) - original_count


def find_mapping_conflicts(
    actions: Iterable[CleaningAction],
) -> list[MappingConflict]:
    """Find category sources that point to more than one target."""

    grouped: dict[
        tuple[str, str | None, str],
        dict[str, list[str]],
    ] = defaultdict(lambda: defaultdict(list))

    for action in actions:
        for source, target in _mapping_items(action):
            key = (action.table, action.column, str(source))
            grouped[key][str(target)].append(action.id)

    conflicts: list[MappingConflict] = []
    for (table, column, source), target_actions in grouped.items():
        if len(target_actions) <= 1:
            continue
        action_ids = sorted(
            action_id
            for ids in target_actions.values()
            for action_id in ids
        )
        conflicts.append(
            MappingConflict(
                table=table,
                column=column,
                source=source,
                targets=tuple(sorted(target_actions)),
                action_ids=tuple(action_ids),
            )
        )

    conflicts.sort(
        key=lambda conflict: (
            conflict.table,
            conflict.column or "",
            conflict.source,
            conflict.targets,
        )
    )
    return conflicts


def review_mapping_conflicts(plan: CleaningPlan) -> list[MappingConflict]:
    """Find unresolved conflicts that should be surfaced during review."""

    candidates = [
        action
        for action in plan.actions
        if action.decision != ActionDecision.REJECTED
    ]
    candidates.extend(
        item.proposed_action
        for item in plan.deferred_reviews
        if item.status == "pending"
    )
    return find_mapping_conflicts(candidates)


def approval_mapping_conflicts(
    plan: CleaningPlan,
    candidates: Iterable[CleaningAction],
) -> list[MappingConflict]:
    """Find conflicts that would exist if candidate actions were approved now."""

    by_id = {
        action.id: action
        for action in plan.actions
        if action.decision == ActionDecision.APPROVED
    }
    for action in candidates:
        by_id[action.id] = action
    return find_mapping_conflicts(by_id.values())


def selected_mapping_conflicts(
    actions: Iterable[CleaningAction],
) -> list[MappingConflict]:
    """Find unsafe conflicts among actions selected for execution."""

    return find_mapping_conflicts(actions)
