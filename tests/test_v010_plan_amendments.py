from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from data_cleaning_assistant.actions import (
    ActionDecision,
    ActionOrigin,
    ActionType,
    CleaningAction,
    CleaningPlan,
)
from data_cleaning_assistant.cli import app
from data_cleaning_assistant.mapping_safety import atomize_category_mappings
from data_cleaning_assistant.planner import file_sha256
from data_cleaning_assistant.plan_amendments import (
    amend_action_decision,
    amend_mapping_target,
)


def _category_action(
    *,
    action_id: str,
    mapping: dict[str, str],
    decision: ActionDecision = ActionDecision.PROPOSED,
) -> CleaningAction:
    return CleaningAction(
        id=action_id,
        type=ActionType.NORMALIZE_CATEGORIES,
        table="study",
        column="label",
        title="Normalize categories",
        description="Test category normalization.",
        parameters={"mapping": mapping},
        confidence=0.9,
        decision=decision,
        origin=ActionOrigin.RULE,
    )


def _plan(source: Path, actions: list[CleaningAction]) -> CleaningPlan:
    return CleaningPlan(
        source_path=str(source.resolve()),
        source_sha256=file_sha256(source),
        actions=actions,
    )


def test_amend_action_decision_records_provenance(tmp_path: Path) -> None:
    source = tmp_path / "study.csv"
    source.write_text("subject_id,label\nS001,alpha\n", encoding="utf-8")
    plan = _plan(
        source,
        [
            _category_action(
                action_id="action_alpha",
                mapping={"alpha": "Alpha"},
                decision=ActionDecision.REJECTED,
            )
        ],
    )

    amendment = amend_action_decision(
        plan,
        action_id="action_alpha",
        decision=ActionDecision.APPROVED,
        reason="Researcher confirmed canonical label.",
    )

    assert plan.actions[0].decision == ActionDecision.APPROVED
    assert amendment.kind == "decision"
    assert amendment.before == {"decision": "rejected"}
    assert amendment.after == {"decision": "approved"}
    assert amendment.reason == "Researcher confirmed canonical label."
    assert plan.amendments == [amendment]


def test_amend_mapping_target_records_old_and_new_target(tmp_path: Path) -> None:
    source = tmp_path / "study.csv"
    source.write_text("subject_id,label\nS001,alpha\n", encoding="utf-8")
    plan = _plan(
        source,
        [
            _category_action(
                action_id="action_alpha",
                mapping={"alpha": "ALPHA"},
                decision=ActionDecision.APPROVED,
            )
        ],
    )

    amendment = amend_mapping_target(
        plan,
        action_id="action_alpha",
        target="Alpha",
        reason="Study dictionary uses title case.",
    )

    assert plan.actions[0].parameters["mapping"] == {"alpha": "Alpha"}
    assert amendment.kind == "mapping_target"
    assert amendment.before == {"mapping": {"alpha": "ALPHA"}}
    assert amendment.after == {"mapping": {"alpha": "Alpha"}}


def test_plan_actions_and_amend_plan_work_without_json_editing(tmp_path: Path) -> None:
    source = tmp_path / "study.csv"
    source.write_text(
        "subject_id,label\n"
        "S001,alpha\n"
        "S002,beta\n",
        encoding="utf-8",
    )
    plan = _plan(
        source,
        [
            _category_action(
                action_id="action_multi",
                mapping={"alpha": "Alpha", "beta": "Beta"},
            )
        ],
    )
    plan_path = tmp_path / "plan.json"
    amended_path = tmp_path / "amended.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")

    atomized = plan.model_copy(deep=True)
    atomize_category_mappings(atomized)
    alpha_action = next(
        action
        for action in atomized.actions
        if action.parameters["mapping"] == {"alpha": "Alpha"}
    )

    runner = CliRunner()
    listing = runner.invoke(app, ["plan-actions", str(plan_path)])
    assert listing.exit_code == 0, listing.output
    assert alpha_action.id in listing.output
    assert "alpha" in listing.output
    assert "Alpha" in listing.output

    result = runner.invoke(
        app,
        [
            "amend-plan",
            str(plan_path),
            "--action-id",
            alpha_action.id,
            "--decision",
            "approved",
            "--reason",
            "Confirmed during review.",
            "--output",
            str(amended_path),
        ],
    )
    assert result.exit_code == 0, result.output

    amended = CleaningPlan.model_validate_json(
        amended_path.read_text(encoding="utf-8")
    )
    amended_alpha = next(
        action for action in amended.actions if action.id == alpha_action.id
    )
    assert amended_alpha.decision == ActionDecision.APPROVED
    assert len(amended.amendments) == 1
    assert amended.amendments[0].reason == "Confirmed during review."


def test_grouped_review_backtracks_and_revisits_previous_group(tmp_path: Path) -> None:
    source = tmp_path / "study.csv"
    source.write_text(
        "subject_id,label\n"
        "S001,alpha\n"
        "S002,beta\n",
        encoding="utf-8",
    )
    plan = _plan(
        source,
        [
            _category_action(
                action_id="action_alpha",
                mapping={"alpha": "Alpha"},
            ),
            _category_action(
                action_id="action_beta",
                mapping={"beta": "Beta"},
            ),
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
    assert "Previous decision undone" in result.output

    reviewed = CleaningPlan.model_validate_json(
        reviewed_path.read_text(encoding="utf-8")
    )
    assert all(
        action.decision == ActionDecision.REJECTED
        for action in reviewed.actions
    )
    assert len(reviewed.amendments) == 1
    amendment = reviewed.amendments[0]
    assert amendment.kind == "decision"
    assert amendment.before == {"decision": "approved"}
    assert amendment.after == {"decision": "proposed"}
    assert amendment.reason == "Interactive review backtracking"

    assert reviewed.review_session is not None
    decisions = [event.decision for event in reviewed.review_session.events]
    assert decisions == [
        "approved",
        "reset_to_proposed",
        "rejected",
        "rejected",
    ]
