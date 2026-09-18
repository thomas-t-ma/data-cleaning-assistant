from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner
import yaml

from data_cleaning_assistant.cli import app
from data_cleaning_assistant.io import load_dataset
from data_cleaning_assistant.planner import propose_actions
from data_cleaning_assistant.recipe import (
    check_recipe_compatibility,
    load_recipe,
    recipe_to_plan,
)
from data_cleaning_assistant.transforms import apply_plan, write_recipe


def _write_dataset(path: Path, *, include_sex: bool = True) -> None:
    columns = ["patient_id"]
    if include_sex:
        columns.append("sex")
    columns.append("status")

    rows = [
        ["P001", "Female", "Active"] if include_sex else ["P001", "Active"],
        ["P002", "female", "N/A"] if include_sex else ["P002", "N/A"],
        ["P003", " Female ", "Active"] if include_sex else ["P003", "Active"],
    ]
    text = ",".join(columns) + "\n"
    text += "\n".join(",".join(row) for row in rows) + "\n"
    path.write_text(text, encoding="utf-8")


def _make_recipe(source: Path, recipe_path: Path) -> None:
    loaded = load_dataset(source)
    plan = propose_actions(source)
    approved_ids = {
        action.id
        for action in plan.actions
        if action.column in {"sex", "status"}
    }
    _, changeset = apply_plan(
        source,
        loaded.tables,
        plan,
        approved_ids=approved_ids,
    )
    write_recipe(plan, changeset, recipe_path)


def test_recipe_replays_on_differently_named_single_table_dataset(tmp_path: Path) -> None:
    source = tmp_path / "study_wave_1.csv"
    target = tmp_path / "study_wave_2.csv"
    recipe_path = tmp_path / "cleaning.recipe.yaml"
    _write_dataset(source)
    _write_dataset(target)
    _make_recipe(source, recipe_path)

    recipe = load_recipe(recipe_path)
    target_tables = load_dataset(target).tables
    plan, compatibility = recipe_to_plan(target, recipe, target_tables)

    assert compatibility.compatible
    assert compatibility.table_mapping == {"study_wave_1": "study_wave_2"}
    assert plan.actions
    assert all(action.table == "study_wave_2" for action in plan.actions)

    cleaned, changeset = apply_plan(
        target,
        target_tables,
        plan,
        approve_all=True,
    )
    assert changeset.affected_cell_count > 0
    frame = cleaned["study_wave_2"]
    assert frame["sex"].tolist() == ["Female", "Female", "Female"]
    assert frame.loc[1, "status"] is None


def test_recipe_reports_missing_columns_instead_of_applying(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    target = tmp_path / "target.csv"
    recipe_path = tmp_path / "cleaning.recipe.yaml"
    _write_dataset(source)
    _write_dataset(target, include_sex=False)
    _make_recipe(source, recipe_path)

    recipe = load_recipe(recipe_path)
    target_tables = load_dataset(target).tables
    report = check_recipe_compatibility(recipe, target_tables)

    assert not report.compatible
    assert any(issue.column == "sex" for issue in report.issues)


def test_recipe_yaml_has_only_executed_actions(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    recipe_path = tmp_path / "cleaning.recipe.yaml"
    _write_dataset(source)

    loaded = load_dataset(source)
    plan = propose_actions(source)
    one_action = next(action for action in plan.actions if action.column == "sex")
    _, changeset = apply_plan(
        source,
        loaded.tables,
        plan,
        approved_ids={one_action.id},
    )
    write_recipe(plan, changeset, recipe_path)

    payload = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    assert len(payload["actions"]) == 1
    assert payload["actions"][0]["column"] == "sex"


def test_replay_cli_is_preview_only_without_apply(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    target = tmp_path / "target.csv"
    recipe_path = tmp_path / "cleaning.recipe.yaml"
    output = tmp_path / "should_not_exist.csv"
    _write_dataset(source)
    _write_dataset(target)
    _make_recipe(source, recipe_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["replay", str(target), str(recipe_path), "-o", str(output)],
    )

    assert result.exit_code == 0
    assert "Preview only" in result.stdout
    assert not output.exists()


def test_replay_cli_applies_only_with_explicit_apply(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    target = tmp_path / "target.csv"
    recipe_path = tmp_path / "cleaning.recipe.yaml"
    output = tmp_path / "cleaned.csv"
    _write_dataset(source)
    _write_dataset(target)
    _make_recipe(source, recipe_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "replay",
            str(target),
            str(recipe_path),
            "--apply",
            "-o",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert output.exists()
    assert output.with_name("cleaned.csv.audit.json").exists()
    assert output.with_name("cleaned.csv.recipe.yaml").exists()
