from __future__ import annotations

from copy import deepcopy

from .actions import (
    ActionDecision,
    CleaningAction,
    CleaningPlan,
    PlanAmendment,
)
from .mapping_safety import selected_mapping_conflicts


def _find_action(plan: CleaningPlan, action_id: str) -> CleaningAction:
    matches = [action for action in plan.actions if action.id == action_id]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one action with id {action_id!r}; found {len(matches)}."
        )
    return matches[0]


def amend_action_decision(
    plan: CleaningPlan,
    *,
    action_id: str,
    decision: ActionDecision,
    reason: str,
) -> PlanAmendment:
    action = _find_action(plan, action_id)
    before = {"decision": action.decision.value}
    action.decision = decision

    if decision == ActionDecision.APPROVED:
        approved = [
            candidate
            for candidate in plan.actions
            if candidate.decision == ActionDecision.APPROVED
        ]
        conflicts = selected_mapping_conflicts(approved)
        if conflicts:
            action.decision = ActionDecision(before["decision"])
            details = "; ".join(
                f"{conflict.table}.{conflict.column or '(table)'}: "
                f"{conflict.source!r} -> {list(conflict.targets)!r}"
                for conflict in conflicts[:8]
            )
            raise ValueError(
                "The requested decision would create conflicting approved "
                f"category mappings. {details}"
            )

    amendment = PlanAmendment(
        kind="decision",
        action_id=action.id,
        before=before,
        after={"decision": action.decision.value},
        reason=reason,
    )
    plan.amendments.append(amendment)
    return amendment


def amend_mapping_target(
    plan: CleaningPlan,
    *,
    action_id: str,
    target: str,
    reason: str,
) -> PlanAmendment:
    action = _find_action(plan, action_id)
    mapping = action.parameters.get("mapping")
    if not isinstance(mapping, dict) or len(mapping) != 1:
        raise ValueError(
            "Mapping-target amendments require one atomic category mapping action."
        )

    source, old_target = next(iter(mapping.items()))
    before_parameters = deepcopy(action.parameters)
    action.parameters = deepcopy(action.parameters)
    action.parameters["mapping"] = {source: target}

    approved = [
        candidate
        for candidate in plan.actions
        if candidate.decision == ActionDecision.APPROVED
    ]
    conflicts = selected_mapping_conflicts(approved)
    if conflicts:
        action.parameters = before_parameters
        details = "; ".join(
            f"{conflict.table}.{conflict.column or '(table)'}: "
            f"{conflict.source!r} -> {list(conflict.targets)!r}"
            for conflict in conflicts[:8]
        )
        raise ValueError(
            "The requested target would create conflicting approved "
            f"category mappings. {details}"
        )

    amendment = PlanAmendment(
        kind="mapping_target",
        action_id=action.id,
        before={"mapping": {source: old_target}},
        after={"mapping": {source: target}},
        reason=reason,
    )
    plan.amendments.append(amendment)
    return amendment


def reset_action_for_backtracking(
    plan: CleaningPlan,
    *,
    action_id: str,
) -> PlanAmendment:
    return amend_action_decision(
        plan,
        action_id=action_id,
        decision=ActionDecision.PROPOSED,
        reason="Interactive review backtracking",
    )
