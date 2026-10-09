from __future__ import annotations

from io import BytesIO
from pathlib import Path

from openpyxl import Workbook

from fastapi.testclient import TestClient

from data_cleaning_assistant.web_app import create_app, validate_bind_host


def _dataset_bytes() -> bytes:
    return (
        b"SubjectID,Status,Outcome\n"
        b"S001,Active,Yes\n"
        b"S002,active,NA\n"
        b"S003,Active,No\n"
    )


def _xlsm_bytes() -> bytes:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Employees"
    sheet.append(["Employee_ID", "Status"])
    sheet.append(["E001", "Active"])
    sheet.append(["E002", "active"])
    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _create_session(client: TestClient) -> dict:
    response = client.post(
        "/api/sessions",
        files={"file": ("study.csv", _dataset_bytes(), "text/csv")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _approve_all_schema(client: TestClient, session_id: str) -> None:
    items = client.get(f"/api/sessions/{session_id}/schema").json()
    for item in items:
        response = client.post(
            f"/api/sessions/{session_id}/schema/{item['id']}",
            json={"decision": "approved"},
        )
        assert response.status_code == 200, response.text
    response = client.post(f"/api/sessions/{session_id}/schema/complete")
    assert response.status_code == 200, response.text


def test_local_web_end_to_end_verified_download(tmp_path: Path) -> None:
    app = create_app(tmp_path / "web")
    client = TestClient(app)

    state = _create_session(client)
    session_id = state["id"]
    assert state["stage"] == "schema_review"
    assert state["schema_suggestions"] > 0

    _approve_all_schema(client, session_id)

    actions_response = client.get(f"/api/sessions/{session_id}/actions")
    assert actions_response.status_code == 200
    actions = actions_response.json()
    assert len(actions) == 2
    assert all(item["affected_count"] >= 1 for item in actions)

    for item in actions:
        response = client.post(
            f"/api/sessions/{session_id}/actions/{item['id']}",
            json={"decision": "approved"},
        )
        assert response.status_code == 200, response.text

    before_complete = client.get(f"/api/sessions/{session_id}").json()
    assert before_complete["action_unresolved"] == 0
    assert before_complete["review_completed"] is False

    complete = client.post(f"/api/sessions/{session_id}/review/complete")
    assert complete.status_code == 200, complete.text
    assert complete.json()["action_unresolved"] == 0
    assert complete.json()["review_completed"] is True

    applied = client.post(f"/api/sessions/{session_id}/apply")
    assert applied.status_code == 200, applied.text
    assert applied.json()["stage"] == "verified"
    assert applied.json()["verification_passed"] is True

    verification = client.get(
        f"/api/sessions/{session_id}/verification"
    )
    assert verification.status_code == 200
    assert verification.json()["passed"] is True
    assert verification.json()["metrics"]["execution_count"] == 2

    download = client.get(f"/api/sessions/{session_id}/download")
    assert download.status_code == 200
    assert b"Active" in download.content
    assert b"active" not in download.content


def test_web_ui_refuses_apply_until_every_action_is_decided(
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(tmp_path / "web"))
    session_id = _create_session(client)["id"]
    _approve_all_schema(client, session_id)

    actions = client.get(f"/api/sessions/{session_id}/actions").json()
    first = actions[0]
    response = client.post(
        f"/api/sessions/{session_id}/actions/{first['id']}",
        json={"decision": "approved"},
    )
    assert response.status_code == 200

    complete = client.post(f"/api/sessions/{session_id}/review/complete")
    assert complete.status_code == 400
    assert "still need a decision" in complete.json()["detail"]

    apply = client.post(f"/api/sessions/{session_id}/apply")
    assert apply.status_code == 400
    assert "review is not complete" in apply.json()["detail"].casefold()


def test_web_session_fails_closed_if_uploaded_source_copy_changes(
    tmp_path: Path,
) -> None:
    app = create_app(tmp_path / "web")
    client = TestClient(app)
    state = _create_session(client)
    session_id = state["id"]
    _approve_all_schema(client, session_id)

    session = app.state.store.load(session_id)
    source = Path(session.source_path)
    source.write_text(
        source.read_text(encoding="utf-8") + "S004,Active,Yes\n",
        encoding="utf-8",
    )

    action = client.get(f"/api/sessions/{session_id}/actions").json()[0]
    response = client.post(
        f"/api/sessions/{session_id}/actions/{action['id']}",
        json={"decision": "approved"},
    )
    assert response.status_code == 400
    assert "source dataset changed" in response.json()["detail"].casefold()


def test_web_upload_validation_and_security_headers(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "web", max_upload_mb=1))

    home = client.get("/")
    assert home.status_code == 200
    assert "Data Cleaning Assistant" in home.text
    assert "Content-Security-Policy" in home.headers
    assert "cdn" not in home.text.casefold()
    assert '.replaceAll("<", "&lt;")' in home.text
    assert 'localStorage.getItem("dataCleanerSession")' in home.text
    assert ".xlsm" in home.text
    assert "Your data stays on this computer" in home.text
    assert "Discard session" in home.text
    assert "Understand fields" in home.text
    assert "Review changes" in home.text

    xlsm = client.post(
        "/api/sessions",
        files={
            "file": (
                "employees.xlsm",
                _xlsm_bytes(),
                "application/vnd.ms-excel.sheet.macroEnabled.12",
            )
        },
    )
    assert xlsm.status_code == 200, xlsm.text

    invalid = client.post(
        "/api/sessions",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert invalid.status_code == 400
    assert "Only CSV, XLSX, XLSM, and Parquet" in invalid.json()["detail"]

    empty = client.post(
        "/api/sessions",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert empty.status_code == 400
    assert "empty" in empty.json()["detail"].casefold()


def test_web_session_delete_removes_app_owned_local_copies(tmp_path: Path) -> None:
    app = create_app(tmp_path / "web")
    client = TestClient(app)
    state = _create_session(client)
    session_id = state["id"]
    session = app.state.store.load(session_id)
    session_root = Path(session.workspace).parent

    assert session_root.exists()

    deleted = client.delete(f"/api/sessions/{session_id}")
    assert deleted.status_code == 200
    assert deleted.json()["status"] == "deleted"
    assert not session_root.exists()

    missing = client.get(f"/api/sessions/{session_id}")
    assert missing.status_code == 404


def test_web_session_delete_failure_is_reported_and_session_remains(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = create_app(tmp_path / "web")
    client = TestClient(app)
    session_id = _create_session(client)["id"]

    def fail_delete(_session_id: str) -> None:
        raise OSError("session files are locked")

    monkeypatch.setattr(app.state.store, "delete", fail_delete)

    deleted = client.delete(f"/api/sessions/{session_id}")
    assert deleted.status_code == 400
    assert "locked" in deleted.json()["detail"]

    still_present = client.get(f"/api/sessions/{session_id}")
    assert still_present.status_code == 200


def test_local_web_rejects_untrusted_host_header(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "web"))

    response = client.get("/", headers={"host": "attacker.example"})

    assert response.status_code == 400
    assert "Invalid host header" in response.text


def test_web_session_identifier_rejects_path_traversal(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "web"))

    response = client.get("/api/sessions/..%2F..%2Fsecret")
    assert response.status_code in {400, 404}


def test_network_binding_requires_explicit_opt_in() -> None:
    validate_bind_host("127.0.0.1", allow_network=False)
    validate_bind_host("localhost", allow_network=False)
    validate_bind_host("::1", allow_network=False)

    try:
        validate_bind_host("0.0.0.0", allow_network=False)
    except ValueError as exc:
        assert "--allow-network" in str(exc)
    else:
        raise AssertionError("Non-loopback binding must fail closed")

    validate_bind_host("0.0.0.0", allow_network=True)
