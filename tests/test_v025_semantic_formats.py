from __future__ import annotations

from pathlib import Path

from data_cleaning_assistant.actions import ActionType
from data_cleaning_assistant.engine import profile_dataset
from data_cleaning_assistant.golden_validation import run_golden_validation
from data_cleaning_assistant.inference import (
    declared_numeric_units,
    parse_number,
)
from data_cleaning_assistant.planner import propose_actions


def test_boolean_variants_normalize_to_yes_no_golden_output(tmp_path: Path) -> None:
    source = tmp_path / "boolean_dirty.csv"
    golden = tmp_path / "boolean_golden.csv"
    source.write_text(
        "id,Consent_Flag\n"
        "A001,Yes\n"
        "A002,Y\n"
        "A003,TRUE\n"
        "A004,1\n"
        "A005,No\n"
        "A006,N\n"
        "A007,FALSE\n"
        "A008,0\n",
        encoding="utf-8",
    )
    golden.write_text(
        "id,Consent_Flag\n"
        "A001,Yes\n"
        "A002,Yes\n"
        "A003,Yes\n"
        "A004,Yes\n"
        "A005,No\n"
        "A006,No\n"
        "A007,No\n"
        "A008,No\n",
        encoding="utf-8",
    )

    report = run_golden_validation(source, golden)
    plan = propose_actions(source)

    assert report.passed is True
    assert report.metrics.expected_changed_cells == 6
    assert report.metrics.matched_expected_changes == 6
    assert [
        (action.type, action.column)
        for action in plan.actions
        if action.column == "Consent_Flag"
    ] == [(ActionType.NORMALIZE_CATEGORIES, "Consent_Flag")]


def test_declared_units_and_percentages_are_reviewable_numeric_coercions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "units_dirty.csv"
    golden = tmp_path / "units_golden.csv"
    source.write_text(
        "id,Weight_kg,Completion_Percent,Comment\n"
        "A001,72 kg,95%,stable\n"
        "A002,68.5kg,80,stable\n"
        "A003,70,100 %,stable\n"
        "A004,75 kilogram,90%,stable\n",
        encoding="utf-8",
    )
    golden.write_text(
        "id,Weight_kg,Completion_Percent,Comment\n"
        "A001,72,95,stable\n"
        "A002,68.5,80,stable\n"
        "A003,70,100,stable\n"
        "A004,75,90,stable\n",
        encoding="utf-8",
    )

    report = run_golden_validation(source, golden)
    plan = propose_actions(source)
    actions = {
        (action.type, action.column): action
        for action in plan.actions
    }

    assert report.passed is True
    assert (ActionType.COERCE_NUMERIC, "Weight_kg") in actions
    assert (ActionType.COERCE_NUMERIC, "Completion_Percent") in actions
    assert actions[(ActionType.COERCE_NUMERIC, "Weight_kg")].parameters[
        "allowed_units"
    ] == ["kg", "kilogram", "kilograms"]
    assert actions[(ActionType.COERCE_NUMERIC, "Completion_Percent")].parameters[
        "allowed_units"
    ] == ["%", "percent", "pct"]


def test_declared_unit_parser_does_not_strip_undeclared_suffixes() -> None:
    kg_units = declared_numeric_units("Weight_kg")

    assert parse_number("72 kg", allowed_units=kg_units) == 72.0
    assert parse_number("68.5kilograms", allowed_units=kg_units) == 68.5
    assert parse_number("72 lb", allowed_units=kg_units) is None
    assert parse_number("sample72 kg", allowed_units=kg_units) is None
    assert declared_numeric_units("Comment") == ()


def test_locale_numeric_format_is_review_only(tmp_path: Path) -> None:
    source = tmp_path / "locale.csv"
    source.write_text(
        'id,Revenue_EUR\n'
        'A001,"1.234,56"\n'
        'A002,"2.345,67"\n'
        'A003,"987,65"\n'
        'A004,"3.210,00"\n',
        encoding="utf-8",
    )

    profile = profile_dataset(source)
    plan = propose_actions(source)

    assert any(
        issue.type == "locale_numeric_format"
        and issue.column == "Revenue_EUR"
        for issue in profile.issues
    )
    assert not any(
        action.column == "Revenue_EUR"
        for action in plan.actions
    )


def test_mojibake_is_detected_but_not_repaired_automatically(tmp_path: Path) -> None:
    source = tmp_path / "encoding.csv"
    source.write_text(
        "id,City,Note\n"
        "A001,JosÃ©,normal text\n"
        "A002,MÃ¼nchen,normal text\n"
        "A003,ZÃ¼rich,normal text\n"
        "A004,Atlanta,normal text\n",
        encoding="utf-8",
    )

    profile = profile_dataset(source)
    plan = propose_actions(source)

    assert any(
        issue.type == "text_encoding_anomaly"
        and issue.column == "City"
        for issue in profile.issues
    )
    assert not any(action.column == "City" for action in plan.actions)
