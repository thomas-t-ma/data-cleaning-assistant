from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from data_cleaning_assistant.actions import CleaningPlan
from data_cleaning_assistant.cli import app
from data_cleaning_assistant.guided import (
    GuidedStage,
    create_guided_session,
    load_guided_session,
    prepare_guided_plan,
)
from data_cleaning_assistant.io import load_dataset
from data_cleaning_assistant.planner import file_sha256


def _write_dataset(path: Path) -> None:
    path.write_text(
        "SubjectID,Status,Outcome\n"
        "S001,Active,Yes\n"
        "S002,active,NA\n"
        "S003,Active,No\n",
        encoding="utf-8",
    )


def test_guided_session_start_is_read_only_and_refuses_reuse(
    tmp_path: Path,
) -> None:
    source = tmp_path / "study.csv"
    workspace = tmp_path / "run"
    _write_dataset(source)
    before = file_sha256(source)

    session = create_guided_session(source, workspace)

    assert session.stage == GuidedStage.SCHEMA_REVIEW
    assert file_sha256(source) == before
    assert (workspace / "session.json").exists()
    assert (workspace / "profile.json").exists()
    assert (workspace / "schema_draft.yaml").exists()
    assert not Path(session.cleaned_path).exists()

    with pytest.raises(FileExistsError, match="guided-resume"):
        create_guided_session(source, workspace)



def test_guided_clean_defaults_workspace_beside_dataset(
    tmp_path: Path,
) -> None:
    source = tmp_path / "study.csv"
    _write_dataset(source)

    result = CliRunner().invoke(
        app,
        ["guided-clean", str(source)],
        # Pause immediately at the first schema suggestion.
        input="q\n",
    )

    assert result.exit_code == 0, result.output
    workspace = tmp_path / "study_cleaning_run"
    assert workspace.exists()
    session = load_guided_session(workspace)
    assert session.stage == GuidedStage.SCHEMA_REVIEW
    assert session.workspace == str(workspace.resolve())


def test_guided_clean_runs_review_apply_and_verification_end_to_end(
    tmp_path: Path,
) -> None:
    source = tmp_path / "study.csv"
    workspace = tmp_path / "run"
    _write_dataset(source)
    before = file_sha256(source)

    result = CliRunner().invoke(
        app,
        [
            "guided-clean",
            str(source),
            "--workspace",
            str(workspace),
        ],
        input="a\na\na\na\na\n",
    )

    assert result.exit_code == 0, result.output
    assert "GUIDED CLEANING COMPLETE" in result.output
    assert file_sha256(source) == before

    session = load_guided_session(workspace)
    assert session.stage == GuidedStage.VERIFIED
    assert session.approved_action_count == 2
    assert session.executed_action_count == 2
    assert session.audit_path is not None
    assert session.recipe_path is not None
    assert session.verification_path is not None

    cleaned = load_dataset(session.cleaned_path).tables
    frame = next(iter(cleaned.values()))
    assert frame["Status"].tolist() == ["Active", "Active", "Active"]
    assert frame["Outcome"].tolist() == ["Yes", "", "No"]


def test_guided_resume_reviews_only_unresolved_actions_and_preserves_history(
    tmp_path: Path,
) -> None:
    source = tmp_path / "study.csv"
    workspace = tmp_path / "run"
    _write_dataset(source)
    runner = CliRunner()

    first = runner.invoke(
        app,
        [
            "guided-clean",
            str(source),
            "--workspace",
            str(workspace),
        ],
        # Approve all three schema suggestions, approve the first cleaning
        # action, then quit during the second.
        input="a\na\na\na\nq\n",
    )
    assert first.exit_code == 0, first.output
    assert "paused during cleaning review" in first.output

    paused = load_guided_session(workspace)
    assert paused.stage == GuidedStage.CLEANING_REVIEW
    assert paused.approved_action_count == 1

    partial_plan = CleaningPlan.model_validate_json(
        Path(paused.reviewed_plan_path).read_text(encoding="utf-8")
    )
    assert partial_plan.review_session is not None
    assert partial_plan.review_session.outcome == "partial"
    prior_events = len(partial_plan.review_session.events)

    # Only the one unresolved action should be prompted on resume.
    resumed = runner.invoke(
        app,
        ["guided-resume", str(workspace)],
        input="a\n",
    )
    assert resumed.exit_code == 0, resumed.output
    assert "GUIDED CLEANING COMPLETE" in resumed.output

    finished = load_guided_session(workspace)
    assert finished.stage == GuidedStage.VERIFIED
    assert finished.approved_action_count == 2

    completed_plan = CleaningPlan.model_validate_json(
        Path(finished.reviewed_plan_path).read_text(encoding="utf-8")
    )
    assert completed_plan.review_session is not None
    assert completed_plan.review_session.outcome == "completed"
    assert len(completed_plan.review_session.events) > prior_events
    decisions = [
        event.decision
        for event in completed_plan.review_session.events
    ]
    assert decisions.count("approved") == 2
    assert "quit" in decisions


def test_guided_resume_fails_closed_if_source_changes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "study.csv"
    workspace = tmp_path / "run"
    _write_dataset(source)
    session = create_guided_session(source, workspace)

    source.write_text(
        source.read_text(encoding="utf-8") + "S004,Active,Yes\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source dataset changed"):
        prepare_guided_plan(session)


def test_guided_status_reads_manifest_without_modifying_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "study.csv"
    workspace = tmp_path / "run"
    _write_dataset(source)
    before = file_sha256(source)
    create_guided_session(source, workspace)

    result = CliRunner().invoke(
        app,
        ["guided-status", str(workspace)],
    )

    assert result.exit_code == 0, result.output
    assert "schema_review" in result.output
    assert str(workspace.resolve()) in result.output
    assert file_sha256(source) == before
