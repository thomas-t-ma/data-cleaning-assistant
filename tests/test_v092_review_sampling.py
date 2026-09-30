from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from data_cleaning_assistant.cli import app
from data_cleaning_assistant.planner import propose_actions


def test_grouped_review_samples_named_and_unnamed_targets(tmp_path: Path) -> None:
    source = tmp_path / "wide.csv"
    source.write_text(
        "...63,...64,important_question,rare_question\n"
        "NA,NA,NA,NA\n"
        "NA,NA,NA,value\n"
        "NA,NA,value,value\n"
        "NA,NA,value,value\n",
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
    assert "Scope includes 2 named column(s) and 2 unnamed export column(s)." in result.output
    assert "Representative targets across this group" in result.output
    assert "Representative before" in result.output
    assert "important_question" in result.output
    assert "rare_question" in result.output
    assert "Unnamed column 63" in result.output
    assert "Press i to inspect targets one by one" in result.output
    assert "Use  if you are not confident" not in result.output
