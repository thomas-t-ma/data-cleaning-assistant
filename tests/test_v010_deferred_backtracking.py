from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from data_cleaning_assistant.actions import (
    ActionDecision,
    ActionOrigin,
    ActionType,
    CleaningAction,
    CleaningPlan,
    DeferredReview,
)
from data_cleaning_assistant.cli import app
from data_cleaning_assistant.planner import file_sha256
from data_cleaning_assistant.plan_amendments import (
    promote_deferred_candidate,
    reset_deferred_for_backtracking,
)


def _category_action(
    *,
    action_id: str,
    source: str,
    target: str,
    decision: ActionDecision = ActionDecision.PROPOSED,
) -> CleaningAction:
    return CleaningAction(
        id=action_id,
        type=ActionType.NORMALIZE_CATEGORIES,
        table="study",
        column="label",
        title="Normalize categories",
        description="Deferred category normalization.",
        parameters={"mapping": {source: target}},
        confidence=0.5,
        decision=decision,
        origin=ActionOrigin.AI,
    )


def _deferred(
    *,
    candidate_id: str,
    action: CleaningAction,
) -> DeferredReview:
    return DeferredReview(
        candidate_id=candidate_id,
        proposed_action=action,
        question="Are these labels equivalent?",
        rationale="AI left this semantic change for human judgment.",
        confidence=0.5,
    )


def _plan(
    source: Path,
    *,
    actions: list[CleaningAction] | None = None,
    deferred: list[DeferredReview] | None = None,
) -> CleaningPlan:
    return CleaningPlan(
        source_path=str(source.resolve()),
        source_sha256=file_sha256(source),
        actions=list(actions or []),
        deferred_reviews=list(deferred or []),
    )


def test_backtracking_removes_action_created_by_deferred_approval(
    tmp_path: Path,
) -> None:
    source = tmp_path / "study.csv"
    source.write_text("subject_id,label\nS001,alhpa\n", encoding="utf-8")
    action = _category_action(
        action_id="ai_alpha",
        source="alhpa",
        target="alpha",
    )
    plan = _plan(
        source,
        deferred=[_deferred(candidate_id="candidate_alpha", action=action)],
    )

    promoted = promote_deferred_candidate(
        plan,
        candidate_id="candidate_alpha",
    )

    item = plan.deferred_reviews[0]
    assert item.status == "approved"
    assert item.promoted_action_created is True
    assert item.promoted_action_id == promoted.id
    assert len(plan.actions) == 1
    assert plan.actions[0].decision == ActionDecision.APPROVED

    amendment = reset_deferred_for_backtracking(
        plan,
        candidate_id="candidate_alpha",
    )

    assert amendment is not None
    assert item.status == "pending"
    assert item.promoted_action_id is None
    assert item.promoted_action_created is False
    assert plan.actions == []
    assert amendment.kind == "deferred_status"
    assert amendment.before["status"] == "approved"
    assert amendment.after["status"] == "pending"


def test_backtracking_restores_existing_action_previous_decision(
    tmp_path: Path,
) -> None:
    source = tmp_path / "study.csv"
    source.write_text("subject_id,label\nS001,alhpa\n", encoding="utf-8")
    existing = _category_action(
        action_id="existing_alpha",
        source="alhpa",
        target="alpha",
        decision=ActionDecision.REJECTED,
    )
    deferred_action = _category_action(
        action_id="ai_alpha",
        source="alhpa",
        target="alpha",
    )
    plan = _plan(
        source,
        actions=[existing],
        deferred=[
            _deferred(
                candidate_id="candidate_alpha",
                action=deferred_action,
            )
        ],
    )

    promoted = promote_deferred_candidate(
        plan,
        candidate_id="candidate_alpha",
    )
    item = plan.deferred_reviews[0]

    assert promoted.id == existing.id
    assert promoted.decision == ActionDecision.APPROVED
    assert item.promoted_action_created is False
    assert item.promoted_action_previous_decision == ActionDecision.REJECTED

    reset_deferred_for_backtracking(
        plan,
        candidate_id="candidate_alpha",
    )

    assert len(plan.actions) == 1
    assert plan.actions[0].id == existing.id
    assert plan.actions[0].decision == ActionDecision.REJECTED
    assert item.status == "pending"


def test_grouped_deferred_review_can_backtrack_approval(
    tmp_path: Path,
) -> None:
    source = tmp_path / "study.csv"
    source.write_text(
        "subject_id,label\n"
        "S001,alhpa\n"
        "S002,betta\n",
        encoding="utf-8",
    )
    alpha = _category_action(
        action_id="ai_alpha",
        source="alhpa",
        target="alpha",
    )
    beta = _category_action(
        action_id="ai_beta",
        source="betta",
        target="beta",
    )
    plan = _plan(
        source,
        deferred=[
            _deferred(candidate_id="candidate_alpha", action=alpha),
            _deferred(candidate_id="candidate_beta", action=beta),
        ],
    )

    plan_path = tmp_path / "plan.json"
    reviewed_path = tmp_path / "reviewed.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "review-grouped",
            str(source),
            str(plan_path),
            "--output",
            str(reviewed_path),
        ],
        input="a\nb\nr\nr\n",
    )

    assert result.exit_code == 0, result.output
    assert "Previous deferred judgment undone" in result.output

    reviewed = CleaningPlan.model_validate_json(
        reviewed_path.read_text(encoding="utf-8")
    )
    assert reviewed.actions == []
    assert [item.status for item in reviewed.deferred_reviews] == [
        "rejected",
        "rejected",
    ]

    assert len(reviewed.amendments) == 1
    amendment = reviewed.amendments[0]
    assert amendment.kind == "deferred_status"
    assert amendment.deferred_candidate_id == "candidate_alpha"
    assert amendment.before["status"] == "approved"
    assert amendment.after["status"] == "pending"

    assert reviewed.review_session is not None
    decisions = [event.decision for event in reviewed.review_session.events]
    assert decisions == [
        "approved",
        "reset_to_pending",
        "rejected",
        "rejected",
    ]
