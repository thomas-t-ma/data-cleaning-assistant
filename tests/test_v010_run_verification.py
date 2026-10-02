from __future__ import annotations

from pathlib import Path

import yaml
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
from data_cleaning_assistant.planner import file_sha256
from data_cleaning_assistant.transforms import (
    apply_plan,
    write_cleaned_dataset,
    write_recipe,
)
from data_cleaning_assistant.verification import verify_run


def _action(
    *,
    action_id: str,
    action_type: ActionType,
    parameters: dict,
) -> CleaningAction:
    return CleaningAction(
        id=action_id,
        type=action_type,
        table="study",
        column="label",
        title="test",
        description="test",
        parameters=parameters,
        confidence=1.0,
        decision=ActionDecision.APPROVED,
        origin=ActionOrigin.RULE,
    )


def _completed_run(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "study.csv"
    cleaned = tmp_path / "cleaned.csv"
    audit = tmp_path / "cleaned.csv.audit.json"
    recipe = tmp_path / "cleaned.csv.recipe.yaml"

    source.write_text(
        'id,label\n1," alpha "\n2,Beta\n',
        encoding="utf-8",
    )
    plan = CleaningPlan(
        source_path=str(source.resolve()),
        source_sha256=file_sha256(source),
        actions=[
            _action(
                action_id="trim",
                action_type=ActionType.TRIM_WHITESPACE,
                parameters={},
            ),
            _action(
                action_id="canonicalize",
                action_type=ActionType.NORMALIZE_CATEGORIES,
                parameters={"mapping": {"alpha": "Alpha"}},
            ),
        ],
    )

    loaded = load_dataset(source)
    cleaned_tables, changeset = apply_plan(
        source,
        loaded.tables,
        plan,
    )
    write_cleaned_dataset(
        source,
        loaded.tables,
        cleaned_tables,
        cleaned,
    )
    changeset.result_path = str(cleaned.resolve())
    changeset.result_sha256 = file_sha256(cleaned)
    audit.write_text(changeset.model_dump_json(indent=2), encoding="utf-8")
    write_recipe(plan, changeset, recipe)
    return source, cleaned, audit, recipe


def test_verify_run_proves_replay_undo_and_repeated_touch_metrics(
    tmp_path: Path,
) -> None:
    source, cleaned, audit, recipe = _completed_run(tmp_path)

    report = verify_run(source, cleaned, audit, recipe)

    assert report.passed is True
    assert all(check.passed for check in report.checks)
    assert report.metrics.execution_count == 2
    assert report.metrics.recipe_action_count == 2
    assert report.metrics.cell_change_events == 2
    assert report.metrics.unique_touched_cells == 1
    assert report.metrics.repeated_touch_events == 1
    assert report.metrics.row_deletions == 0
    assert report.metrics.net_different_cells == 1
    assert report.metrics.replay_byte_identical is True


def test_verify_run_cli_uses_default_artifacts_and_writes_json(
    tmp_path: Path,
) -> None:
    source, cleaned, _audit, _recipe = _completed_run(tmp_path)
    report_path = tmp_path / "verification.json"

    result = CliRunner().invoke(
        app,
        [
            "verify-run",
            str(source),
            str(cleaned),
            "--json-output",
            str(report_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "VERIFICATION PASS" in result.output
    assert "repeated-touch event(s)" in result.output
    assert '"passed": true' in report_path.read_text(encoding="utf-8")


def test_verify_run_fails_if_cleaned_output_was_modified(
    tmp_path: Path,
) -> None:
    source, cleaned, audit, recipe = _completed_run(tmp_path)
    cleaned.write_text(
        "id,label\n1,Tampered\n2,Beta\n",
        encoding="utf-8",
    )

    report = verify_run(source, cleaned, audit, recipe)

    assert report.passed is False
    checks = {check.name: check for check in report.checks}
    assert checks["cleaned_matches_audit"].passed is False
    assert checks["recipe_replay_equivalent"].passed is False


def test_verify_run_fails_if_recipe_no_longer_reproduces_output(
    tmp_path: Path,
) -> None:
    source, cleaned, audit, recipe = _completed_run(tmp_path)
    payload = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    payload["actions"][1]["parameters"]["mapping"]["alpha"] = "ALPHA"
    recipe.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    report = verify_run(source, cleaned, audit, recipe)

    assert report.passed is False
    checks = {check.name: check for check in report.checks}
    assert checks["recipe_action_sequence_matches_audit"].passed is True
    assert checks["recipe_replay_equivalent"].passed is False
    assert checks["audit_undo_restores_source"].passed is True


def test_verify_run_fails_if_audit_counts_are_internally_inconsistent(
    tmp_path: Path,
) -> None:
    source, cleaned, audit, recipe = _completed_run(tmp_path)
    payload = __import__("json").loads(audit.read_text(encoding="utf-8"))
    payload["executions"][0]["affected_count"] += 1
    audit.write_text(
        __import__("json").dumps(payload, indent=2),
        encoding="utf-8",
    )

    report = verify_run(source, cleaned, audit, recipe)

    assert report.passed is False
    checks = {check.name: check for check in report.checks}
    assert checks["audit_counts_consistent"].passed is False
