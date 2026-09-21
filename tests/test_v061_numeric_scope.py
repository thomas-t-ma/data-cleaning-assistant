"""Mixed-assay numeric QA must not advertise pooled test values as outliers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from data_cleaning_assistant.cli import app
from data_cleaning_assistant.engine import profile_dataset
from data_cleaning_assistant.study_schema import StudySchema


def _mixed_assays(path: Path, *, value_column: str = "test_value") -> None:
    first = list(range(1, 9)) + [100]
    second = list(range(1000, 1009))
    rows = [
        {
            "laboratory_test": "Test A",
            "test_units": "mg/L",
            value_column: n,
        }
        for n in first
    ] + [
        {
            "laboratory_test": "Test B",
            "test_units": "g/L",
            value_column: n,
        }
        for n in second
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_mixed_assay_measurements_skip_pooled_outliers_without_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "measurements.csv"
    _mixed_assays(path)
    report = profile_dataset(path)
    assert not [
        issue for issue in report.issues
        if issue.type == "statistical_outlier" and issue.column == "test_value"
    ]
    scoped = [
        issue for issue in report.issues
        if issue.type == "unscoped_numeric_measurement"
    ]
    assert len(scoped) == 1
    issue = scoped[0]
    assert issue.column == "test_value"
    assert issue.auto_fix_safe is False
    assert issue.affected_count == 18
    assert issue.evidence["grouping_metadata_unique_counts"] == {
        "laboratory_test": 2, "test_units": 2,
    }
    assert issue.evidence["suppressed_pooled_outlier_count"] == 0
    assert issue.evidence["grouped_outlier_check_performed"] is False


def test_actual_pooled_outlier_flag_is_suppressed(tmp_path: Path) -> None:
    path = tmp_path / "measurements.csv"
    rows = [
        {"laboratory_test": "A", "test_units": "unit A", "test_value": n}
        for n in list(range(1, 9)) + [100]
    ] + [
        {"laboratory_test": "B", "test_units": "unit B", "test_value": n}
        for n in range(1, 10)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    report = profile_dataset(path)
    scoped = [
        issue for issue in report.issues
        if issue.type == "unscoped_numeric_measurement"
    ]
    assert len(scoped) == 1
    assert scoped[0].evidence["suppressed_pooled_outlier_count"] > 0
    assert not any(
        issue.type == "statistical_outlier" and issue.column == "test_value"
        for issue in report.issues
    )




def test_explicit_assay_and_unit_grouping_restores_within_group_qa(
    tmp_path: Path,
) -> None:
    path = tmp_path / "measurements.csv"
    _mixed_assays(path)
    schema = StudySchema.model_validate({
        "tables": {
            "measurements": {
                "numeric_groups": {
                    "test_value": ["laboratory_test", "test_units"],
                },
            },
        },
    })
    report = profile_dataset(path, schema=schema)
    assert not any(
        issue.type == "unscoped_numeric_measurement" for issue in report.issues
    )
    outliers = [
        issue for issue in report.issues
        if issue.type == "statistical_outlier" and issue.column == "test_value"
    ]
    assert len(outliers) == 1
    assert outliers[0].affected_count == 1
    assert outliers[0].row_indices_sample == [8]
    assert outliers[0].evidence["group_by"] == ["laboratory_test", "test_units"]


def test_non_generic_numeric_column_does_not_inherit_assay_guard(
    tmp_path: Path,
) -> None:
    path = tmp_path / "measurements.csv"
    _mixed_assays(path, value_column="hemoglobin")
    report = profile_dataset(path)
    assert not any(
        issue.type == "unscoped_numeric_measurement" for issue in report.issues
    )


def test_single_assay_keeps_existing_numeric_detector(tmp_path: Path) -> None:
    path = tmp_path / "measurements.csv"
    pd.DataFrame({
        "laboratory_test": ["One assay"] * 9,
        "test_units": ["mg/L"] * 9,
        "test_value": list(range(1, 9)) + [100],
    }).to_csv(path, index=False)
    report = profile_dataset(path)
    assert any(
        issue.type == "statistical_outlier" and issue.column == "test_value"
        for issue in report.issues
    )
    assert not any(
        issue.type == "unscoped_numeric_measurement" for issue in report.issues
    )


def test_unscoped_issue_is_read_only_and_in_full_json(tmp_path: Path) -> None:
    path = tmp_path / "measurements.csv"
    output = tmp_path / "profile.json"
    _mixed_assays(path)
    result = CliRunner().invoke(
        app, ["profile", str(path), "--output", str(output)]
    )
    assert result.exit_code == 0, result.output
    assert "Pooled outlier check skipped" in result.output
    assert "Remaining QA findings" in result.output
    report = json.loads(output.read_text(encoding="utf-8"))
    scope_issues = [
        issue for issue in report["issues"]
        if issue["type"] == "unscoped_numeric_measurement"
    ]
    assert len(scope_issues) == 1
    assert scope_issues[0]["auto_fix_safe"] is False
