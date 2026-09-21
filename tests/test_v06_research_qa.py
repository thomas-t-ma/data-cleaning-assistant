from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from data_cleaning_assistant.actions import ActionType, CleaningAction, CleaningPlan
from data_cleaning_assistant.cli import app
from data_cleaning_assistant.engine import profile_dataset
from data_cleaning_assistant.issue_summary import summarize_issues, visible_issues
from data_cleaning_assistant.preview_overlap import overlapping_previews
from data_cleaning_assistant.planner import file_sha256
from data_cleaning_assistant.study_schema import StudySchema


def _small_study(tmp_path: Path) -> Path:
    path = tmp_path / "study.csv"
    path.write_text(
        "subject_id,emptyfield,site,group,score\n"
        "A,NA,Atlanta, Female ,1\n"
        "B,NA,Atlanta,Male,2\n"
        "C,NA,Atlanta,Female,3\n"
        "D,NA,Atlanta,Male,4\n",
        encoding="utf-8",
    )
    return path


def test_lossless_noise_summary_and_cli_switch(tmp_path: Path) -> None:
    source = _small_study(tmp_path)
    report = profile_dataset(source)
    summary = summarize_issues(report)
    assert summary["total_issues"] == len(report.issues)
    assert summary["by_table"]["study"]["coverage_noise_issues"] >= 2
    assert len(visible_issues(report)) < len(report.issues)
    assert any(i.type == "surrounding_whitespace" for i in visible_issues(report))

    output = tmp_path / "profile.json"
    runner = CliRunner()
    brief = runner.invoke(app, ["profile", str(source), "-o", str(output)])
    verbose = runner.invoke(app, ["profile", str(source), "--all-issues"])
    assert brief.exit_code == 0, brief.output
    assert verbose.exit_code == 0, verbose.output
    assert "Coverage-only findings" in brief.output
    assert "Complete-coverage findings are grouped" in brief.output
    assert "All QA findings" in verbose.output
    assert "Coverage-only findings" not in verbose.output
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert len(saved["issues"]) == len(report.issues)


def test_declared_relationships_replace_incidental_shared_identifiers(
    tmp_path: Path,
) -> None:
    source = tmp_path / "study.xlsx"
    with pd.ExcelWriter(source) as writer:
        pd.DataFrame({"case_id": ["1", "2"]}).to_excel(
            writer, sheet_name="Cases", index=False
        )
        pd.DataFrame(
            {"case_id": ["1", "1", "2"], "treatment_id": ["t1", "t2", "t3"]}
        ).to_excel(writer, sheet_name="Treatments", index=False)
        pd.DataFrame(
            {"case_id": ["1", "1", "2"], "followup_id": ["f1", "f2", "f3"]}
        ).to_excel(writer, sheet_name="FollowUps", index=False)
    schema = StudySchema.model_validate({
        "tables": {
            "Cases": {"grain_key": ["case_id"]},
            "Treatments": {"grain_key": ["treatment_id"]},
            "FollowUps": {"grain_key": ["followup_id"]},
        },
        "foreign_keys": [
            {
                "parent_table": "Cases", "parent_column": "case_id",
                "child_table": "Treatments", "child_column": "case_id",
            },
            {
                "parent_table": "Cases", "parent_column": "case_id",
                "child_table": "FollowUps", "child_column": "case_id",
            },
        ],
    })
    unconfigured = profile_dataset(source)
    assert any(r.relationship == "many-to-many" for r in unconfigured.relationships)
    report = profile_dataset(source, schema=schema)
    assert len(report.relationships) == 2
    assert all(r.source == "declared" for r in report.relationships)
    assert all(r.relationship == "one-to-many" for r in report.relationships)
    assert all(t.grain.label.startswith("declared:") for t in report.tables)


def test_grouped_numeric_outliers_only_within_declared_assay_unit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "measurements.csv"
    observations = [
        {"assay": "A", "unit": "mg/L", "value": str(n)}
        for n in [1, 2, 3, 4, 5, 6, 7, 8, 100]
    ] + [
        {"assay": "B", "unit": "g/L", "value": str(n)}
        for n in range(1000, 1009)
    ]
    pd.DataFrame(observations).to_csv(path, index=False)
    schema = StudySchema.model_validate({
        "tables": {"measurements": {
            "numeric_groups": {"value": ["assay", "unit"]}
        }}
    })
    report = profile_dataset(path, schema=schema)
    grouped = [i for i in report.issues if i.type == "statistical_outlier"]
    assert len(grouped) == 1
    assert grouped[0].evidence["group_by"] == ["assay", "unit"]
    assert grouped[0].evidence["group_size"] == 9
    assert grouped[0].affected_count == 1
    assert grouped[0].row_indices_sample == [8]
    assert not any("assay" in str(i.evidence.get("examples", [])) for i in grouped)


def test_schema_numeric_group_must_reference_real_columns(tmp_path: Path) -> None:
    path = _small_study(tmp_path)
    schema = StudySchema.model_validate(
        {"tables": {"study": {"numeric_groups": {"score": ["nonexistent"]}}}}
    )
    with pytest.raises(ValueError, match="unknown columns"):
        profile_dataset(path, schema=schema)


def test_preview_overlap_identifies_shared_row_without_mutation() -> None:
    frame = pd.DataFrame({"Sex": [" Female ", "female", "Male"]})
    actions = [
        CleaningAction(
            id="trim",
            type=ActionType.TRIM_WHITESPACE,
            table="Visits",
            column="Sex",
            title="Trim",
            description="Trim",
            confidence=1.0,
            parameters={},
        ),
        CleaningAction(
            id="dictionary",
            type=ActionType.NORMALIZE_CATEGORIES,
            table="Visits",
            column="Sex",
            title="Canonical",
            description="Canonical",
            confidence=1.0,
            parameters={"mapping": {" Female ": "Female", "female": "Female"}},
        ),
    ]
    before = frame.copy(deep=True)
    overlap = overlapping_previews({"Visits": frame}, actions)
    assert overlap == {
        "trim": [("dictionary", 1)],
        "dictionary": [("trim", 1)],
    }
    pd.testing.assert_frame_equal(frame, before)


def test_unconfigured_workflow_remains_unchanged(tmp_path: Path) -> None:
    path = _small_study(tmp_path)
    report = profile_dataset(path)
    assert all(r.source == "inferred" for r in report.relationships)
    assert any(i.type == "hidden_missing_values" for i in report.issues)


def test_cli_review_warns_when_independent_previews_overlap(tmp_path: Path) -> None:
    source = _small_study(tmp_path)
    plan = CleaningPlan(
        source_path=str(source.resolve()),
        source_sha256=file_sha256(source),
        actions=[
            CleaningAction(
                id="trim",
                type=ActionType.TRIM_WHITESPACE,
                table="study",
                column="group",
                title="Trim",
                description="Trim whitespace",
                confidence=1.0,
                parameters={},
            ),
            CleaningAction(
                id="dictionary",
                type=ActionType.NORMALIZE_CATEGORIES,
                table="study",
                column="group",
                title="Canonical",
                description="Canonical",
                confidence=1.0,
                parameters={"mapping": {" Female ": "Female"}},
            ),
        ],
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.model_dump_json(), encoding="utf-8")
    result = CliRunner().invoke(app, ["preview", str(source), str(plan_path)])
    assert result.exit_code == 0, result.output
    assert "Independent-preview overlap" in result.output
    assert "1 shared row(s)" in result.output
