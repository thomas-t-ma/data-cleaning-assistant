from __future__ import annotations

from pathlib import Path

from data_cleaning_assistant.actions import ActionType
from data_cleaning_assistant.ai_candidates import build_ai_candidates
from data_cleaning_assistant.benchmark import benchmark_cases
from data_cleaning_assistant.engine import profile_dataset
from data_cleaning_assistant.io import load_dataset
from data_cleaning_assistant.planner import propose_actions
from data_cleaning_assistant.transforms import preview_action


def test_general_suite_spans_eight_nonresearch_domains() -> None:
    cases = benchmark_cases("general")

    assert len(cases) == 8
    assert {case.id for case in cases} == {
        "general_hr_employee_export",
        "general_sales_order_export",
        "general_survey_free_response",
        "general_inventory_export",
        "general_finance_statement_export",
        "general_education_roster",
        "general_crm_contact_export",
        "general_logistics_carrier_typo",
    }


def test_parseable_mixed_date_formats_are_still_reviewable(tmp_path: Path) -> None:
    source = tmp_path / "orders.csv"
    source.write_text(
        "Order_ID,Order_Date,Status\n"
        "ORD001,2026-01-01,Shipped\n"
        "ORD002,01/13/2026,Shipped\n"
        "ORD003,02/14/2026,Pending\n"
        "ORD004,03/18/2026,Delivered\n"
        "ORD005,04/22/2026,Delivered\n",
        encoding="utf-8",
    )

    plan = propose_actions(source)
    action = next(
        item
        for item in plan.actions
        if item.type == ActionType.STANDARDIZE_DATES
        and item.column == "Order_Date"
    )

    assert action.parameters["numeric_order"] == "month_first"
    loaded = load_dataset(source)
    preview = preview_action(loaded.tables, action)
    assert preview.affected_count == 4


def test_small_comment_field_is_not_category_candidate(tmp_path: Path) -> None:
    source = tmp_path / "survey.csv"
    source.write_text(
        "Respondent_ID,Satisfaction,Comment\n"
        "R001,Satisfied,Great service\n"
        "R002,Unknown,Great service\n"
        "R003,Dissatisfied,Great servce\n"
        "R004,Satisfied,Great service\n"
        "R005,Satisfied,Would use again\n",
        encoding="utf-8",
    )

    loaded = load_dataset(source)
    candidates = build_ai_candidates(profile_dataset(source), loaded.tables)

    assert not any(
        candidate.action.type == ActionType.NORMALIZE_CATEGORIES
        and candidate.action.column == "Comment"
        for candidate in candidates
    )
