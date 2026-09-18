from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from data_cleaning_assistant.actions import ActionType
from data_cleaning_assistant.io import load_dataset
from data_cleaning_assistant.planner import file_sha256, propose_actions
from data_cleaning_assistant.transforms import (
    apply_plan,
    preview_action,
    undo_changeset,
    write_recipe,
    write_tables,
)


def _write_messy_csv(path: Path) -> None:
    path.write_text(
        "patient_id,sex,age,visit_date\n"
        "P001,Female,35,2026-01-01\n"
        "P002,female,N/A,01/03/2026\n"
        'P003, Female ,41,"January 5, 2026"\n'
        "P004,Male,forty,bad-date\n",
        encoding="utf-8",
    )


def test_plan_proposes_reviewable_actions(tmp_path: Path) -> None:
    source = tmp_path / "messy.csv"
    _write_messy_csv(source)

    plan = propose_actions(source)
    action_types = {action.type for action in plan.actions}

    assert ActionType.NORMALIZE_CATEGORIES in action_types
    assert ActionType.NORMALIZE_MISSING in action_types
    assert ActionType.STANDARDIZE_DATES in action_types
    assert all(action.decision.value == "proposed" for action in plan.actions)
    assert all(action.reversible for action in plan.actions)


def test_preview_and_apply_never_mutate_source_frame(tmp_path: Path) -> None:
    source = tmp_path / "messy.csv"
    _write_messy_csv(source)
    loaded = load_dataset(source)
    original = loaded.tables[source.stem].copy(deep=True)
    plan = propose_actions(source)

    category = next(
        action
        for action in plan.actions
        if action.type == ActionType.NORMALIZE_CATEGORIES
    )
    missing = next(
        action
        for action in plan.actions
        if action.type == ActionType.NORMALIZE_MISSING
    )

    preview = preview_action(loaded.tables, category)
    assert preview.affected_count >= 1
    pd.testing.assert_frame_equal(loaded.tables[source.stem], original)

    cleaned, changeset = apply_plan(
        source,
        loaded.tables,
        plan,
        approved_ids={category.id, missing.id},
    )
    pd.testing.assert_frame_equal(loaded.tables[source.stem], original)
    assert len(changeset.executions) == 2
    assert changeset.affected_cell_count >= 2

    cleaned_frame = cleaned[source.stem]
    assert set(cleaned_frame["sex"].tolist()) >= {"Female", "Male"}
    assert cleaned_frame.loc[1, "age"] is None


def test_changeset_round_trip_restores_original_values(tmp_path: Path) -> None:
    source = tmp_path / "messy.csv"
    _write_messy_csv(source)
    loaded = load_dataset(source)
    original = loaded.tables[source.stem].copy(deep=True)
    plan = propose_actions(source)

    approved = {
        action.id
        for action in plan.actions
        if action.type in {
            ActionType.NORMALIZE_CATEGORIES,
            ActionType.NORMALIZE_MISSING,
            ActionType.STANDARDIZE_DATES,
        }
    }
    cleaned, changeset = apply_plan(
        source,
        loaded.tables,
        plan,
        approved_ids=approved,
    )

    restored = undo_changeset(cleaned, changeset)
    pd.testing.assert_frame_equal(restored[source.stem], original)


def test_stale_plan_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "messy.csv"
    _write_messy_csv(source)
    plan = propose_actions(source)
    loaded = load_dataset(source)

    source.write_text(
        source.read_text(encoding="utf-8") + "P999,Female,50,2026-02-01\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="changed since this cleaning plan"):
        apply_plan(source, loaded.tables, plan, approve_all=True)


def test_written_output_recipe_and_audit_hash(tmp_path: Path) -> None:
    source = tmp_path / "messy.csv"
    _write_messy_csv(source)
    loaded = load_dataset(source)
    plan = propose_actions(source)

    category = next(
        action
        for action in plan.actions
        if action.type == ActionType.NORMALIZE_CATEGORIES
    )
    cleaned, changeset = apply_plan(
        source,
        loaded.tables,
        plan,
        approved_ids={category.id},
    )

    output = write_tables(cleaned, tmp_path / "cleaned.csv")
    changeset.result_path = str(output)
    changeset.result_sha256 = file_sha256(output)

    recipe_path = write_recipe(plan, changeset, tmp_path / "recipe.yaml")
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))

    assert output.exists()
    assert changeset.result_sha256 == file_sha256(output)
    assert len(recipe["actions"]) == 1
    assert recipe["actions"][0]["type"] == ActionType.NORMALIZE_CATEGORIES.value


def test_written_csv_can_be_reloaded_and_undone(tmp_path: Path) -> None:
    source = tmp_path / "messy.csv"
    _write_messy_csv(source)
    original = load_dataset(source).tables[source.stem].copy(deep=True)
    plan = propose_actions(source)

    approved = {
        action.id
        for action in plan.actions
        if action.type in {
            ActionType.NORMALIZE_CATEGORIES,
            ActionType.NORMALIZE_MISSING,
        }
    }
    loaded = load_dataset(source)
    cleaned, changeset = apply_plan(
        source,
        loaded.tables,
        plan,
        approved_ids=approved,
    )

    cleaned_path = write_tables(cleaned, tmp_path / "renamed_cleaned.csv")
    reloaded = load_dataset(cleaned_path)
    restored = undo_changeset(reloaded.tables, changeset)

    restored_frame = next(iter(restored.values()))
    pd.testing.assert_frame_equal(restored_frame, original)
