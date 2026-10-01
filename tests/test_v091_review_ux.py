from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from data_cleaning_assistant.cli import _friendly_column_name, _safe_console_text, app
from data_cleaning_assistant.planner import propose_actions


def test_grouped_review_uses_plain_language_and_examples(tmp_path: Path) -> None:
    source = tmp_path / "study.csv"
    source.write_text(
        "subject_id,a,b\n"
        "S001,NA,NA\n"
        "S002,1,2\n"
        "S003,3,4\n",
        encoding="utf-8",
    )
    plan = propose_actions(source)
    plan_path = tmp_path / "plan.json"
    reviewed_path = tmp_path / "reviewed.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "review-grouped",
            str(source),
            str(plan_path),
            "--output",
            str(reviewed_path),
        ],
        input="s\n",
    )

    assert result.exit_code == 0, result.output
    assert "Review proposed cleaning" in result.output
    assert "Treat 'NA' as missing" in result.output
    assert "What it does:" in result.output
    assert "What to check:" in result.output
    assert "Where this change would happen" in result.output
    assert "Examples:" in result.output
    assert "'NA' → missing (null)" in result.output
    assert "source dataset was not changed" in result.output

    assert "Origin:" not in result.output
    assert "Parameters:" not in result.output
    assert "Affected by independent previews" not in result.output


def test_unnamed_export_columns_get_readable_labels() -> None:
    assert _friendly_column_name("...63") == "Unnamed column 63"
    assert _friendly_column_name("real_name") == "real_name"


def test_console_text_falls_back_for_cp1252() -> None:
    assert _safe_console_text("→", "->", encoding="cp1252") == "->"
    assert _safe_console_text("→", "->", encoding="utf-8") == "→"
