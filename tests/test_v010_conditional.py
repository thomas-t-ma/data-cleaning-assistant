from __future__ import annotations

from pathlib import Path

from data_cleaning_assistant.engine import profile_dataset


def _conditional_issues(path: Path):
    return [
        issue
        for issue in profile_dataset(path).issues
        if issue.type == "conditional_dependency_inconsistency"
    ]


def test_if_yes_dependency_flags_contradiction_and_missing_detail(
    tmp_path: Path,
) -> None:
    source = tmp_path / "conditional.csv"
    source.write_text(
        "subject_id,11. Does the data have a license?,12. If yes, what license?\n"
        "S001,Yes,MIT\n"
        "S002,No,\n"
        "S003,No,CC BY\n"
        "S004,Yes,NA\n"
        "S005,Unclear,Unclear\n"
        "S006,No,No\n",
        encoding="utf-8",
    )

    issues = _conditional_issues(source)

    assert len(issues) == 1
    issue = issues[0]
    assert issue.affected_count == 2
    assert issue.evidence["parent_column"] == "11. Does the data have a license?"
    assert issue.evidence["trigger_state"] == "yes"
    assert issue.evidence["contradiction_count"] == 1
    assert issue.evidence["missing_detail_count"] == 1
    assert issue.auto_fix_safe is False


def test_consistent_if_yes_dependency_has_no_issue(tmp_path: Path) -> None:
    source = tmp_path / "consistent.csv"
    source.write_text(
        "subject_id,has_license,If yes, what license?\n"
        "S001,Yes,MIT\n"
        "S002,No,\n"
        "S003,No,No\n"
        "S004,Yes,GPL v3\n"
        "S005,Unclear,Unclear\n",
        encoding="utf-8",
    )

    assert _conditional_issues(source) == []


def test_if_yes_field_does_not_link_to_nonbinary_parent(tmp_path: Path) -> None:
    source = tmp_path / "nonbinary.csv"
    source.write_text(
        "subject_id,status,If yes, explain\n"
        "S001,Alpha,detail one\n"
        "S002,Beta,detail two\n"
        "S003,Gamma,detail three\n"
        "S004,Delta,detail four\n",
        encoding="utf-8",
    )

    assert _conditional_issues(source) == []
