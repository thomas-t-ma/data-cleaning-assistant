from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from data_cleaning_assistant.actions import ActionDecision, ActionType, CleaningAction
from data_cleaning_assistant.engine import profile_dataset
from data_cleaning_assistant.io import load_dataset
from data_cleaning_assistant.presentation import (
    action_explanation,
    action_sheet_preview,
    readable_name,
)
from data_cleaning_assistant.semantic_narrator import (
    LocalSemanticNarrator,
    SemanticNarration,
)
from data_cleaning_assistant.web_app import create_app


def _write_context_dataset(path: Path) -> None:
    path.write_text(
        "Employee_ID,Join_Date,Department,Role\n"
        "E001,January 8 2026, Sales ,Analyst\n"
        "E002,2026/01/09,Sales,Manager\n"
        "E003,01-10-2026, Research ,Scientist\n",
        encoding="utf-8",
    )


def test_action_explanation_uses_dataset_language_not_action_codes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "employees.csv"
    _write_context_dataset(source)
    report = profile_dataset(source)
    action = CleaningAction(
        id="date_action",
        type=ActionType.STANDARDIZE_DATES,
        table="employees",
        column="Join_Date",
        title="Standardize parseable date values",
        description="technical description",
        parameters={"format": "%Y-%m-%d", "unparseable": "keep"},
        confidence=0.9,
    )

    explanation = action_explanation(
        action,
        affected_count=3,
        report=report,
    )

    assert explanation.title == "Make Join Date dates consistent"
    assert "3 values in Join Date" in explanation.summary
    assert "YYYY-MM-DD" in explanation.summary
    assert explanation.context is not None
    assert "Join Date looks like" in explanation.context
    assert "standardize_dates" not in explanation.title
    assert "{" not in explanation.summary


def test_action_sheet_preview_looks_like_spreadsheet_context(
    tmp_path: Path,
) -> None:
    source = tmp_path / "employees.csv"
    _write_context_dataset(source)
    tables = load_dataset(source).tables
    action = CleaningAction(
        id="date_action",
        type=ActionType.STANDARDIZE_DATES,
        table="employees",
        column="Join_Date",
        title="dates",
        description="dates",
        parameters={"format": "%Y-%m-%d", "unparseable": "keep"},
        confidence=0.9,
    )

    preview = action_sheet_preview(tables, action, sample_limit=3)

    assert preview.target_columns == ["Join_Date"]
    assert "Employee_ID" in preview.columns
    assert "Join_Date" in preview.columns
    assert "Department" in preview.columns
    assert preview.rows
    first = preview.rows[0]
    assert first.row_number == 2
    target = first.cells["Join_Date"]
    assert target.changed is True
    assert target.before == "January 8 2026"
    assert target.after == "2026-01-08"
    assert first.cells["Employee_ID"].text == "E001"


def test_readable_name_preserves_common_acronyms() -> None:
    assert readable_name("Join_Date") == "Join Date"
    assert readable_name("DOB") == "DOB"
    assert readable_name("employee_id") == "Employee ID"


class RecordingProvider:
    name = "fake"

    def __init__(self) -> None:
        self.system_prompt = ""
        self.user_prompt = ""
        self.calls = 0

    def list_models(self) -> list[str]:
        return ["local-test"]

    def generate_structured(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_model,
    ):
        self.calls += 1
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return response_model(
            title="Make employee join dates consistent",
            explanation=(
                "These values are all being written as the same date style; "
                "the underlying dates are not being changed."
            ),
            data_context=(
                "Join Date appears to record when each employee joined, based on "
                "the field name and nearby Employee ID and Department columns."
            ),
            caution="Check any ambiguous day/month examples before approval.",
        )


def test_local_narrator_is_explanation_only_and_receives_bounded_context(
    tmp_path: Path,
) -> None:
    source = tmp_path / "employees.csv"
    _write_context_dataset(source)
    tables = load_dataset(source).tables
    report = profile_dataset(source)
    action = CleaningAction(
        id="date_action",
        type=ActionType.STANDARDIZE_DATES,
        table="employees",
        column="Join_Date",
        title="dates",
        description="dates",
        parameters={"format": "%Y-%m-%d", "unparseable": "keep"},
        confidence=0.9,
    )
    explanation = action_explanation(action, affected_count=3, report=report)

    from data_cleaning_assistant.presentation import action_context_payload

    payload = action_context_payload(
        dataset_name=source.name,
        tables=tables,
        action=action,
        explanation=explanation,
        report=report,
    )
    provider = RecordingProvider()
    narrator = LocalSemanticNarrator(provider, model="local-test")
    result = narrator.narrate(payload)

    assert result.title == "Make employee join dates consistent"
    assert provider.calls == 1
    assert "ZERO authority" in provider.system_prompt
    assert "Do not introduce new mappings" in provider.system_prompt

    sent = json.loads(provider.user_prompt)
    assert sent["column"] == "Join_Date"
    assert sent["nearby_columns"]
    assert sent["before_after_examples"]
    assert "parameters" not in sent
    assert "full_dataframe" not in sent


def _family_dataset() -> bytes:
    return (
        b"Employee_ID,Department,Role\n"
        b"E001, Sales , Analyst \n"
        b"E002,Research,Manager\n"
        b"E003, Research , Scientist \n"
    )


def _start_and_finish_schema(client: TestClient) -> str:
    response = client.post(
        "/api/sessions",
        files={"file": ("employees.csv", _family_dataset(), "text/csv")},
    )
    assert response.status_code == 200, response.text
    session_id = response.json()["id"]
    schema_items = client.get(f"/api/sessions/{session_id}/schema").json()
    for item in schema_items:
        decided = client.post(
            f"/api/sessions/{session_id}/schema/{item['id']}",
            json={"decision": "rejected"},
        )
        assert decided.status_code == 200, decided.text
    completed = client.post(f"/api/sessions/{session_id}/schema/complete")
    assert completed.status_code == 200, completed.text
    return session_id


def test_web_families_group_same_cleanup_and_show_spreadsheet_examples(
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(tmp_path / "web"))
    session_id = _start_and_finish_schema(client)

    response = client.get(f"/api/sessions/{session_id}/families")
    assert response.status_code == 200, response.text
    families = response.json()

    trim_family = next(
        item for item in families if item["action_type"] == "trim_whitespace"
    )
    assert trim_family["member_count"] == 2
    assert trim_family["affected_count"] == 4
    assert "fields" in trim_family["explanation"]["title"]
    assert len(trim_family["members"]) == 2

    department = next(
        member
        for member in trim_family["members"]
        if member["column"] == "Department"
    )
    assert department["preview"]["rows"]
    changed = department["preview"]["rows"][0]["cells"]["Department"]
    assert changed["before"] == '" Sales "'
    assert changed["after"] == "Sales"

    approved = client.post(
        f"/api/sessions/{session_id}/families/{trim_family['id']}",
        json={"decision": "approved"},
    )
    assert approved.status_code == 200, approved.text
    assert approved.json()["approved_actions"] == 2


class FakeNarrator:
    model = "fake-local"

    def __init__(self) -> None:
        self.calls = 0

    def narrate(self, payload):
        self.calls += 1
        return SemanticNarration(
            title=f"Understand {readable_name(payload.column or payload.table)}",
            explanation="This is an advisory explanation of the existing cleanup.",
            data_context="The field appears to describe employee information.",
            caution="Check the examples before approval.",
        )


def test_web_semantic_narration_is_optional_and_cached(
    tmp_path: Path,
) -> None:
    narrator = FakeNarrator()
    app = create_app(tmp_path / "web", narrator=narrator)
    client = TestClient(app)
    session_id = _start_and_finish_schema(client)

    state = client.get(f"/api/sessions/{session_id}").json()
    assert state["narrator_enabled"] is True

    action = client.get(f"/api/sessions/{session_id}/actions").json()[0]
    first = client.post(
        f"/api/sessions/{session_id}/actions/{action['id']}/narrate"
    )
    second = client.post(
        f"/api/sessions/{session_id}/actions/{action['id']}/narrate"
    )

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first.json() == second.json()
    assert narrator.calls == 1


def test_browser_markup_prioritizes_spreadsheets_over_raw_json(
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(tmp_path / "web"))
    home = client.get("/")

    assert home.status_code == 200
    assert "Review your data the way you would review a spreadsheet" in home.text
    assert 'class="sheet"' in home.text
    assert "before" in home.text
    assert "after" in home.text
    assert "JSON.stringify(item.parameters)" not in home.text
