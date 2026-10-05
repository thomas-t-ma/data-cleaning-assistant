from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from data_cleaning_assistant.actions import ActionDecision, ActionType
from data_cleaning_assistant.engine import profile_dataset
from data_cleaning_assistant.inference import (
    infer_numeric_date_order,
    parse_date,
)
from data_cleaning_assistant.io import load_dataset
from data_cleaning_assistant.planner import propose_actions
from data_cleaning_assistant.transforms import apply_plan, preview_action
from data_cleaning_assistant.web_app import create_app


def _employee_like_csv() -> str:
    return (
        "Employee_ID,Join_Date,Salary,Email,Phone\n"
        "EMP1,4/2/2021,50000,a@example.com,-1651623197\n"
        "EMP2,7/10/2020,60000,b@example.com,-1898471390\n"
        "EMP3,12/7/2023,N/A,c@example.com,-5596363211\n"
        "EMP4,11/27/2021,70000,d@example.com,-3476490784\n"
        "EMP5,1/5/2022,80000,e@example.com,-1586734256\n"
        "EMP6,6/30/2023,90000,f@example.com,-5409003485\n"
        "EMP7,9/21/2022,95000,g@example.com,-4518376063\n"
    )


def test_numeric_date_order_uses_only_directional_evidence() -> None:
    import pandas as pd

    month_first = pd.Series(
        ["4/2/2021", "7/10/2020", "11/27/2021", "6/30/2023"]
    )
    order, evidence = infer_numeric_date_order(month_first)
    assert order is None
    assert evidence["month_first_evidence"] == 2
    assert evidence["ambiguous_values"] == 2

    # Two directional rows are intentionally not enough for column authority.
    assert evidence["directional_evidence"] == 2

    strong_month_first = pd.Series(
        [
            "4/2/2021",
            "7/10/2020",
            "11/27/2021",
            "6/30/2023",
            "9/21/2022",
        ]
    )
    order, evidence = infer_numeric_date_order(strong_month_first)
    assert order == "month_first"
    assert evidence["month_first_evidence"] == 3
    assert evidence["day_first_evidence"] == 0

    conflicting = pd.Series(
        ["11/27/2021", "6/30/2023", "9/21/2022", "29/09/2025"]
    )
    order, evidence = infer_numeric_date_order(conflicting)
    assert order is None
    assert evidence["month_first_evidence"] == 3
    assert evidence["day_first_evidence"] == 1


def test_parse_date_remains_fail_closed_without_column_convention() -> None:
    assert parse_date("4/2/2021") is None
    assert parse_date(
        "4/2/2021",
        numeric_order="month_first",
    ).date().isoformat() == "2021-04-02"
    assert parse_date(
        "4/2/2021",
        numeric_order="day_first",
    ).date().isoformat() == "2021-02-04"


def test_employee_style_date_column_is_cleaned_coherently(tmp_path: Path) -> None:
    source = tmp_path / "employees.csv"
    source.write_text(_employee_like_csv(), encoding="utf-8")
    loaded = load_dataset(source)

    plan = propose_actions(source)
    date_action = next(
        action
        for action in plan.actions
        if action.type == ActionType.STANDARDIZE_DATES
    )
    assert date_action.parameters["numeric_order"] == "month_first"

    preview = preview_action(loaded.tables, date_action)
    assert preview.affected_count == 7

    date_action.decision = ActionDecision.APPROVED
    cleaned, changeset = apply_plan(source, loaded.tables, plan)
    frame = cleaned[source.stem]
    assert frame["Join_Date"].tolist() == [
        "2021-04-02",
        "2020-07-10",
        "2023-12-07",
        "2021-11-27",
        "2022-01-05",
        "2023-06-30",
        "2022-09-21",
    ]
    assert changeset.executions


def test_conflicting_date_conventions_do_not_authorize_ambiguous_rows(
    tmp_path: Path,
) -> None:
    source = tmp_path / "mixed_dates.csv"
    source.write_text(
        "id,visit_date\n"
        "1,11/27/2021\n"
        "2,6/30/2023\n"
        "3,9/21/2022\n"
        "4,29/09/2025\n"
        "5,4/2/2021\n",
        encoding="utf-8",
    )
    plan = propose_actions(source)
    action = next(
        item
        for item in plan.actions
        if item.type == ActionType.STANDARDIZE_DATES
    )
    assert "numeric_order" not in action.parameters

    loaded = load_dataset(source)
    preview = preview_action(loaded.tables, action)
    # The four directional values can be standardized; the ambiguous value stays.
    assert preview.affected_count == 4
    ambiguous_changes = [
        change
        for change in preview.sample_changes
        if change.old_value.value == "4/2/2021"
    ]
    assert ambiguous_changes == []


def test_phone_column_is_semantic_warning_not_cleaning_action(
    tmp_path: Path,
) -> None:
    source = tmp_path / "employees.csv"
    source.write_text(_employee_like_csv(), encoding="utf-8")

    report = profile_dataset(source)
    phone_issue = next(
        issue for issue in report.issues
        if issue.type == "phone_format_anomaly"
    )
    assert phone_issue.column == "Phone"
    assert phone_issue.affected_count == 7
    assert phone_issue.evidence["negative_integer_like_count"] == 7
    assert "stored or exported as a numeric type" in phone_issue.description

    plan = propose_actions(source)
    assert not any(
        action.column == "Phone"
        for action in plan.actions
    )


def test_web_surfaces_phone_warning_with_spreadsheet_context(
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(tmp_path / "web"))
    response = client.post(
        "/api/sessions",
        files={
            "file": (
                "employees.csv",
                _employee_like_csv().encode("utf-8"),
                "text/csv",
            )
        },
    )
    assert response.status_code == 200, response.text
    state = response.json()
    assert state["warning_count"] >= 1

    warnings = client.get(
        f"/api/sessions/{state['id']}/warnings"
    )
    assert warnings.status_code == 200, warnings.text
    phone = next(
        item for item in warnings.json()
        if item["title"] == "Phone values look structurally unusual"
    )
    assert phone["affected_count"] == 7
    assert phone["preview"]["target_columns"] == ["Phone"]
    assert phone["preview"]["rows"]
