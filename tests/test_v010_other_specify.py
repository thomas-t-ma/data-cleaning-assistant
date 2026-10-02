from __future__ import annotations

from pathlib import Path

from data_cleaning_assistant.engine import profile_dataset


def _other_issues(path: Path):
    return [
        issue
        for issue in profile_dataset(path).issues
        if issue.type == "other_specify_inconsistency"
    ]


def test_other_specify_flags_missing_and_unexpected_detail(tmp_path: Path) -> None:
    source = tmp_path / "other.csv"
    source.write_text(
        'subject_id,repository,"If other repository, specify"\n'
        "S001,Dryad,\n"
        "S002,Other,Institutional archive\n"
        "S003,Other,\n"
        "S004,Zenodo,manual archive\n"
        "S005,Figshare,\n",
        encoding="utf-8",
    )

    issues = _other_issues(source)

    assert len(issues) == 1
    issue = issues[0]
    assert issue.evidence["parent_column"] == "repository"
    assert issue.evidence["missing_detail_count"] == 1
    assert issue.evidence["unexpected_detail_count"] == 1
    assert issue.affected_count == 2
    assert issue.auto_fix_safe is False


def test_consistent_other_specify_has_no_issue(tmp_path: Path) -> None:
    source = tmp_path / "consistent_other.csv"
    source.write_text(
        'subject_id,repository,"If other repository, specify"\n'
        "S001,Dryad,\n"
        "S002,Other,Institutional archive\n"
        "S003,Zenodo,\n"
        "S004,Other,University server\n",
        encoding="utf-8",
    )

    assert _other_issues(source) == []


def test_multiselect_parent_can_select_other(tmp_path: Path) -> None:
    source = tmp_path / "multiselect_other.csv"
    source.write_text(
        'subject_id,software,"If other software, specify"\n'
        'S001,"R;Python",\n'
        'S002,"R;Other",Julia wrapper\n'
        'S003,"Python;Other",\n'
        'S004,"R;Python",custom tool\n',
        encoding="utf-8",
    )

    issues = _other_issues(source)

    assert len(issues) == 1
    assert issues[0].evidence["missing_detail_count"] == 1
    assert issues[0].evidence["unexpected_detail_count"] == 1


def test_nonadjacent_generic_other_field_does_not_guess_parent(tmp_path: Path) -> None:
    source = tmp_path / "nonadjacent_other.csv"
    source.write_text(
        'subject_id,repository,reviewer_note,"If other, specify"\n'
        "S001,Other,checked,Institutional archive\n"
        "S002,Dryad,checked,unexpected detail\n"
        "S003,Zenodo,checked,\n"
        "S004,Other,checked,\n",
        encoding="utf-8",
    )

    assert _other_issues(source) == []


def test_free_text_column_with_word_other_is_not_parent(tmp_path: Path) -> None:
    source = tmp_path / "free_text_parent.csv"
    source.write_text(
        'subject_id,comments,"If other, specify"\n'
        'S001,"Other observations from reviewer",detail\n'
        'S002,"Other issue with repository",detail\n'
        'S003,"General comment about methods",detail\n'
        'S004,"Another long narrative response here",detail\n',
        encoding="utf-8",
    )

    assert _other_issues(source) == []
