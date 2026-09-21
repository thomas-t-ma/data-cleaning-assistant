"""Plan triage must preserve missingness QA without creating no-value cleanup queues."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from data_cleaning_assistant.actions import ActionType, CleaningPlan
from data_cleaning_assistant.ai_candidates import build_ai_candidates
from data_cleaning_assistant.ai_planner import propose_ai_plan
from data_cleaning_assistant.cli import app
from data_cleaning_assistant.engine import profile_dataset
from data_cleaning_assistant.io import load_dataset
from data_cleaning_assistant.planner import file_sha256, propose_actions
from data_cleaning_assistant.study_schema import StudySchema
from data_cleaning_assistant.transforms import apply_plan


def _wide_clinical_export(path: Path) -> None:
    rows = 12
    frame = pd.DataFrame({
        "subject_id": [f"P{i:03d}" for i in range(rows)],
        **{
            f"unreported_{index:03d}": ["NA"] * rows
            for index in range(40)
        },
        "mixed_nulls": ["NA"] * 8 + [None] * 4,
        "partial_age": ["NA", "42"] + ["35"] * 10,
        "contextual_status": ["Unknown", "NA"] + ["Known"] * 10,
        "all_contextual": ["Unknown"] * rows,
    })
    frame.to_csv(path, index=False)


def _missing_actions(plan: CleaningPlan):
    return [
        action for action in plan.actions
        if action.type == ActionType.NORMALIZE_MISSING
    ]


def test_structural_only_columns_are_coverage_findings_not_routine_actions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "study.csv"
    _wide_clinical_export(source)
    report = profile_dataset(source)
    all_structural = [
        i for i in report.issues
        if i.type == "hidden_missing_values" and i.column == "unreported_000"
    ]
    assert len(all_structural) == 1
    assert all_structural[0].affected_count == 12

    plan = propose_actions(source)
    assert len(plan.planning_notes) == 41
    assert len(_missing_actions(plan)) == 2
    assert {a.column for a in _missing_actions(plan)} == {
        "partial_age", "contextual_status"
    }
    assert plan.source_sha256 == file_sha256(source)
    assert all(note.reason == "structural_only_column" for note in plan.planning_notes)
    assert "unreported_000" in {note.column for note in plan.planning_notes}
    assert not any(a.column == "all_contextual" for a in plan.actions)


def test_ai_candidates_skip_structural_only_fields_but_keep_partial(
    tmp_path: Path,
) -> None:
    source = tmp_path / "study.csv"
    _wide_clinical_export(source)
    report = profile_dataset(source)
    candidates = build_ai_candidates(report, load_dataset(source).tables)
    missing = [
        candidate for candidate in candidates
        if candidate.action.type == ActionType.NORMALIZE_MISSING
    ]
    assert {candidate.action.column for candidate in missing} == {
        "partial_age", "contextual_status"
    }
    assert all(
        candidate.action.column not in {"mixed_nulls", "unreported_000"}
        for candidate in missing
    )


def test_schema_explicit_token_overrides_suppression(tmp_path: Path) -> None:
    source = tmp_path / "study.csv"
    _wide_clinical_export(source)
    schema = StudySchema.model_validate({
        "tables": {
            "study": {
                "columns": {
                    "unreported_000": {"missing_tokens": ["NA"]}
                }
            }
        }
    })
    plan = propose_actions(source, schema=schema)
    expected = [
        action for action in _missing_actions(plan)
        if action.column == "unreported_000"
    ]
    assert len(expected) == 1
    assert expected[0].parameters == {"values": ["NA"]}
    assert "unreported_000" not in {
        note.column for note in plan.planning_notes
    }
    unchanged = load_dataset(source).tables
    cleaned, audit = apply_plan(source, unchanged, plan)
    assert audit.executions == []
    assert cleaned["study"].equals(unchanged["study"])


class MustNotCallAI:
    name = "fake"

    def generate_structured(self, **kwargs):
        raise AssertionError("No actionable semantic candidates; model must be skipped")


def test_ai_only_plan_preserves_non_executable_notes(tmp_path: Path) -> None:
    source = tmp_path / "study.csv"
    pd.DataFrame({"all_missing": ["NA", "NA"], "status": ["Unknown", "Unknown"]}).to_csv(
        source, index=False
    )
    result = propose_ai_plan(
        source, provider=MustNotCallAI(), model="fake", include_rules=False
    )
    assert result.candidates == []
    assert result.plan.actions == []
    assert result.plan.deferred_reviews == []
    assert len(result.plan.planning_notes) == 1


def test_cli_plan_persists_compact_notes_without_data_edits(tmp_path: Path) -> None:
    source = tmp_path / "study.csv"
    output = tmp_path / "plan.json"
    _wide_clinical_export(source)
    original_bytes = source.read_bytes()
    result = CliRunner().invoke(
        app, ["plan", str(source), "--output", str(output)]
    )
    assert result.exit_code == 0, result.output
    assert "Non-executable coverage QA:" in result.output
    assert "41 structural-only" in result.output
    assert output.exists()
    saved = CleaningPlan.model_validate_json(output.read_text(encoding="utf-8"))
    assert len(saved.planning_notes) == 41
    assert len(saved.actions) == 2
    assert source.read_bytes() == original_bytes
