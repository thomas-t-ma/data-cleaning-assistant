from __future__ import annotations

from pathlib import Path

from data_cleaning_assistant import desktop_app


def test_desktop_port_selection_uses_loopback() -> None:
    port = desktop_app._find_available_port()

    assert 1 <= port <= 65535


def test_desktop_main_uses_user_local_storage(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    def fake_run_local_desktop(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(desktop_app, "run_local_desktop", fake_run_local_desktop)

    desktop_app.main([])

    assert captured["data_root"] == (
        tmp_path / ".data-cleaner" / "desktop" / "sessions"
    )
