from __future__ import annotations

from copy import deepcopy

from .actions import (
    ActionDecision,
    CleaningAction,
    CleaningPlan,
    DeferredReview,
    PlanAmendment,
)
from .mapping_safety import approval_mapping_conflicts, selected_mapping_conflicts


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



def _find_deferred(
    plan: CleaningPlan,
    candidate_id: str,
) -> DeferredReview:
    matches = [
        item
        for item in plan.deferred_reviews
        if item.candidate_id == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one deferred candidate with id "
            f"{candidate_id!r}; found {len(matches)}."
        )
    return matches[0]


def promote_deferred_candidate(
    plan: CleaningPlan,
    *,
    candidate_id: str,
) -> CleaningAction:
    """Approve one deferred candidate while retaining reversible provenance."""

    deferred = _find_deferred(plan, candidate_id)
    action = deferred.proposed_action

    conflicts = approval_mapping_conflicts(plan, [action])
    if conflicts:
        details = "; ".join(
            f"{conflict.table}.{conflict.column or '(table)'}: "
            f"{conflict.source!r} -> {list(conflict.targets)!r}"
            for conflict in conflicts[:8]
        )
        raise ValueError(
            "Approving the deferred candidate would create conflicting "
            f"category mappings. {details}"
        )

    matching = next(
        (
            existing
            for existing in plan.actions
            if existing.type == action.type
            and existing.table == action.table
            and existing.column == action.column
            and existing.parameters == action.parameters
        ),
        None,
    )

    if matching is not None:
        deferred.promoted_action_id = matching.id
        deferred.promoted_action_created = False
        deferred.promoted_action_previous_decision = matching.decision
        matching.decision = ActionDecision.APPROVED
        recorded_action = matching
    else:
        approved_action = action.model_copy(deep=True)
        approved_action.decision = ActionDecision.APPROVED
        approved_action.description = (
            "Manually approved deferred AI candidate: " + deferred.rationale
        )
        plan.actions.append(approved_action)
        deferred.promoted_action_id = approved_action.id
        deferred.promoted_action_created = True
        deferred.promoted_action_previous_decision = None
        recorded_action = approved_action

    deferred.status = "approved"
    return recorded_action


def reject_deferred_candidate(
    plan: CleaningPlan,
    *,
    candidate_id: str,
) -> DeferredReview:
    deferred = _find_deferred(plan, candidate_id)
    deferred.status = "rejected"
    deferred.promoted_action_id = None
    deferred.promoted_action_created = False
    deferred.promoted_action_previous_decision = None
    return deferred


def reset_deferred_for_backtracking(
    plan: CleaningPlan,
    *,
    candidate_id: str,
) -> PlanAmendment | None:
    """Undo a prior deferred-candidate decision and return it to pending."""

    deferred = _find_deferred(plan, candidate_id)
    before = {
        "status": deferred.status,
        "promoted_action_id": deferred.promoted_action_id,
        "promoted_action_created": deferred.promoted_action_created,
        "promoted_action_previous_decision": (
            deferred.promoted_action_previous_decision.value
            if deferred.promoted_action_previous_decision is not None
            else None
        ),
    }

    if deferred.status == "pending":
        return None

    if deferred.status == "approved" and deferred.promoted_action_id is not None:
        promoted = next(
            (
                action
                for action in plan.actions
                if action.id == deferred.promoted_action_id
            ),
            None,
        )
        if promoted is not None:
            if deferred.promoted_action_created:
                plan.actions.remove(promoted)
            else:
                promoted.decision = (
                    deferred.promoted_action_previous_decision
                    or ActionDecision.PROPOSED
                )

    deferred.status = "pending"
    deferred.promoted_action_id = None
    deferred.promoted_action_created = False
    deferred.promoted_action_previous_decision = None

    amendment = PlanAmendment(
        kind="deferred_status",
        deferred_candidate_id=deferred.candidate_id,
        before=before,
        after={
            "status": "pending",
            "promoted_action_id": None,
            "promoted_action_created": False,
            "promoted_action_previous_decision": None,
        },
        reason="Interactive deferred-review backtracking",
    )
    plan.amendments.append(amendment)
    return amendment
