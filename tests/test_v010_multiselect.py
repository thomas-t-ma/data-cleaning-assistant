from __future__ import annotations

from pathlib import Path

from data_cleaning_assistant.actions import ActionType
from data_cleaning_assistant.io import load_dataset
from data_cleaning_assistant.planner import propose_actions
from data_cleaning_assistant.transforms import apply_plan, undo_changeset


def test_multiselect_variants_get_component_level_action(tmp_path: Path) -> None:
    source = tmp_path / "multiselect.csv"
    source.write_text(
        "subject_id,language\n"
        'S001,"R;BUGS"\n'
        'S002,"R;Bugs"\n'
        'S003,"Python;BUGS"\n'
        'S004,"R;BUGS"\n',
        encoding="utf-8",
    )

    plan = propose_actions(source)

    actions = [
        action
        for action in plan.actions
        if action.type == ActionType.NORMALIZE_MULTISELECT
    ]
    assert len(actions) == 1
    action = actions[0]
    assert action.column == "language"
    assert action.parameters == {
        "delimiter": ";",
        "mapping": {"Bugs": "BUGS"},
    }
    assert not any(
        item.type == ActionType.NORMALIZE_CATEGORIES
        and item.column == "language"
        for item in plan.actions
    )


def test_multiselect_normalization_preserves_other_selected_components(
    tmp_path: Path,
) -> None:
    source = tmp_path / "multiselect.csv"
    source.write_text(
        "subject_id,language\n"
        'S001,"R; BUGS; Python"\n'
        'S002,"R; Bugs; Python"\n'
        'S003,"R; BUGS; Julia"\n'
        'S004,"R; BUGS; Python"\n',
        encoding="utf-8",
    )
    original = source.read_bytes()
    loaded = load_dataset(source)
    plan = propose_actions(source)
    action = next(
        item for item in plan.actions
        if item.type == ActionType.NORMALIZE_MULTISELECT
    )

    cleaned, changeset = apply_plan(
        source,
        loaded.tables,
        plan,
        approved_ids={action.id},
    )

    table = next(iter(cleaned.values()))
    assert table.loc[1, "language"] == "R; BUGS; Python"
    assert table.loc[0, "language"] == "R; BUGS; Python"
    assert changeset.executions[0].affected_count == 1
    assert source.read_bytes() == original

    restored = undo_changeset(cleaned, changeset)
    restored_table = next(iter(restored.values()))
    assert restored_table.loc[1, "language"] == "R; Bugs; Python"


def test_multiselect_detector_skips_free_response_semicolon_prose(
    tmp_path: Path,
) -> None:
    source = tmp_path / "comments.csv"
    source.write_text(
        "subject_id,comments\n"
        'S001,"No; repository was unavailable"\n'
        'S002,"no; repository had moved"\n'
        'S003,"The code ran; documentation was incomplete"\n'
        'S004,"The package installed; one optional dependency failed"\n'
        'S005,"Several scripts were present; paths were machine-specific"\n'
        'S006,"The archive opened; tests were not documented"\n',
        encoding="utf-8",
    )

    plan = propose_actions(source)

    assert not any(
        action.type == ActionType.NORMALIZE_MULTISELECT
        for action in plan.actions
    )
