from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from data_cleaning_assistant.actions import (
    ActionDecision,
    ActionOrigin,
    ActionType,
    CleaningAction,
    CleaningPlan,
)
from data_cleaning_assistant.cli import app
from data_cleaning_assistant.io import load_dataset
from data_cleaning_assistant.mapping_safety import (
    atomize_category_mappings,
    find_mapping_conflicts,
)
from data_cleaning_assistant.planner import file_sha256
from data_cleaning_assistant.transforms import apply_plan


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


def test_atomize_category_mappings_creates_one_action_per_mapping(tmp_path: Path) -> None:
    source = tmp_path / "study.csv"
    source.write_text(
        "subject_id,label\n"
        "S001,alpha\n"
        "S002,beta\n"
        "S003,gamma\n",
        encoding="utf-8",
    )
    original = _category_action(
        action_id="action_multi",
        mapping={
            "alpha": "Alpha",
            "beta": "Beta",
            "gamma": "Gamma",
        },
    )
    plan = _plan(source, [original])

    split_count, additional_actions = atomize_category_mappings(plan)

    assert split_count == 1
    assert additional_actions == 2
    assert len(plan.actions) == 3
    assert len({action.id for action in plan.actions}) == 3
    assert all(action.parent_action_id == "action_multi" for action in plan.actions)
    assert all(len(action.parameters["mapping"]) == 1 for action in plan.actions)
    assert {
        next(iter(action.parameters["mapping"].items()))
        for action in plan.actions
    } == {
        ("alpha", "Alpha"),
        ("beta", "Beta"),
        ("gamma", "Gamma"),
    }


def test_grouped_review_decides_each_mapping_independently(tmp_path: Path) -> None:
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
        input="a\nr\n",
    )

    assert result.exit_code == 0, result.output
    assert "Mapping-level review" in result.output

    reviewed = CleaningPlan.model_validate_json(
        reviewed_path.read_text(encoding="utf-8")
    )
    assert len(reviewed.actions) == 2
    assert all(
        action.parent_action_id == "action_multi"
        for action in reviewed.actions
    )

    decisions_by_source = {
        next(iter(action.parameters["mapping"])): action.decision
        for action in reviewed.actions
    }
    assert set(decisions_by_source.values()) == {
        ActionDecision.APPROVED,
        ActionDecision.REJECTED,
    }


def test_find_mapping_conflicts_ignores_duplicate_same_target() -> None:
    actions = [
        _category_action(
            action_id="a1",
            mapping={"Uclear": "Unclear"},
        ),
        _category_action(
            action_id="a2",
            mapping={"Uclear": "Unclear"},
        ),
        _category_action(
            action_id="a3",
            mapping={"Uclear": "unclear"},
        ),
    ]

    conflicts = find_mapping_conflicts(actions)

    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict.source == "Uclear"
    assert conflict.targets == ("Unclear", "unclear")
    assert set(conflict.action_ids) == {"a1", "a2", "a3"}


def test_grouped_review_blocks_second_conflicting_approval(tmp_path: Path) -> None:
    source = tmp_path / "study.csv"
    source.write_text(
        "subject_id,label\n"
        "S001,Uclear\n"
        "S002,Unclear\n"
        "S003,unclear\n",
        encoding="utf-8",
    )
    plan = _plan(
        source,
        [
            _category_action(
                action_id="action_upper",
                mapping={"Uclear": "Unclear"},
            ),
            _category_action(
                action_id="action_lower",
                mapping={"Uclear": "unclear"},
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
        input="a\na\nr\n",
    )

    assert result.exit_code == 0, result.output
    assert "Conflicting canonical targets detected" in result.output
    assert "Approval blocked" in result.output

    reviewed = CleaningPlan.model_validate_json(
        reviewed_path.read_text(encoding="utf-8")
    )
    assert sum(
        action.decision == ActionDecision.APPROVED
        for action in reviewed.actions
    ) == 1
    assert sum(
        action.decision == ActionDecision.REJECTED
        for action in reviewed.actions
    ) == 1


def test_apply_fails_closed_on_conflicting_approved_mappings(tmp_path: Path) -> None:
    source = tmp_path / "study.csv"
    source.write_text(
        "subject_id,label\n"
        "S001,Uclear\n",
        encoding="utf-8",
    )
    plan = _plan(
        source,
        [
            _category_action(
                action_id="action_upper",
                mapping={"Uclear": "Unclear"},
                decision=ActionDecision.APPROVED,
            ),
            _category_action(
                action_id="action_lower",
                mapping={"Uclear": "unclear"},
                decision=ActionDecision.APPROVED,
            ),
        ],
    )
    loaded = load_dataset(source)

    with pytest.raises(
        ValueError,
        match="Conflicting approved category mappings",
    ):
        apply_plan(source, loaded.tables, plan)
