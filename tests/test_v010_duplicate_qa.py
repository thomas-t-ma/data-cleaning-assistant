from __future__ import annotations

from pathlib import Path

from data_cleaning_assistant.engine import profile_dataset


def _duplicate_qa(path: Path):
    return [
        issue
        for issue in profile_dataset(path).issues
        if issue.type == "normalization_equivalent_rows"
    ]


def test_normalization_equivalent_rows_are_flagged_without_deletion(
    tmp_path: Path,
) -> None:
    source = tmp_path / "near_duplicates.csv"
    source.write_text(
        "subject_id,label,status\n"
        "S001,Alpha,Active\n"
        "S001, alpha ,active\n"
        "S002,Beta,Active\n",
        encoding="utf-8",
    )

    issues = _duplicate_qa(source)

    assert len(issues) == 1
    issue = issues[0]
    assert issue.affected_count == 2
    assert issue.row_indices_sample == [0, 1]
    assert issue.auto_fix_safe is False
    assert issue.suggested_action is not None
    assert "Do not delete" in issue.suggested_action


def test_exact_duplicates_are_left_to_exact_duplicate_detector(
    tmp_path: Path,
) -> None:
    source = tmp_path / "exact.csv"
    source.write_text(
        "subject_id,label\n"
        "S001,Alpha\n"
        "S001,Alpha\n",
        encoding="utf-8",
    )

    assert _duplicate_qa(source) == []


def test_different_identifiers_do_not_collapse_as_near_duplicates(
    tmp_path: Path,
) -> None:
    source = tmp_path / "different_ids.csv"
    source.write_text(
        "subject_id,label,status\n"
        "S001,Alpha,Active\n"
        "S002, alpha ,active\n",
        encoding="utf-8",
    )

    assert _duplicate_qa(source) == []


def test_structural_missing_sentinels_can_reveal_duplicate_equivalence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "missing_variants.csv"
    source.write_text(
        "subject_id,label,note\n"
        "S001,Alpha,NA\n"
        "S001, alpha ,null\n"
        "S002,Beta,NA\n",
        encoding="utf-8",
    )

    issues = _duplicate_qa(source)

    assert len(issues) == 1
    assert issues[0].affected_count == 2


def test_identifier_case_difference_is_not_normalized_into_duplicate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "case_sensitive_ids.csv"
    source.write_text(
        "subject_id,label,status\n"
        "ABC123,Alpha,Active\n"
        "abc123, alpha ,active\n"
        "XYZ999,Beta,Active\n",
        encoding="utf-8",
    )

    assert _duplicate_qa(source) == []
