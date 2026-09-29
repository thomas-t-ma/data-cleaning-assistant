from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from data_cleaning_assistant.cli import app
from data_cleaning_assistant.recall_eval import build_recall_surface_report


def test_recall_report_summarizes_new_review_surfaces(tmp_path: Path) -> None:
    source = tmp_path / "recall.csv"
    source.write_text(
        'subject_id,language,has_license,"If yes, what license?"\n'
        'S001,"R;BUGS",Yes,MIT\n'
        'S002,"R;Bugs",No,\n'
        'S003,"Python;BUGS",No,CC BY\n'
        'S004,"R;BUGS",Yes,GPL v3\n'
        'S005,"R;BUGS",Yes,MIT\n'
        'S006,"Python;BUGS",Yes,MIT\n'
        'S007,"R;BUGS",No,\n'
        'S008,"R;BUGS",Yes,MIT\n'
        'S009,"Python;BUGS",Yes,GPL v3\n'
        'S010,"R;BUGS",No,\n'
        'S011,"R;BUGS",Yes,MIT\n'
        'S012,"Python;BUGS",Yes,MIT\n'
        'S013,"R;BUGS",No,\n'
        'S014,"R;BUGS",Yes,GPL v3\n'
        'S015,"Python;BUGS",Yes,MIT\n'
        'S016,"R;BUGS",No,\n'
        'S017,"R;BUGS",Yes,MIT\n'
        'S018,"Python;BUGS",Yes,GPL v3\n'
        'S019,"R;BUGS",No,\n'
        'S020,"R;BUGS",Yes,MIT\n',
        encoding="utf-8",
    )

    report = build_recall_surface_report(source)

    assert report.multiselect_action_count == 1
    assert report.conditional_pair_count == 1
    assert report.conditional_affected_count == 1
    assert report.normalized_duplicate_group_issue_count == 0
    assert report.codebook_suggestion_count >= 1
    assert "scientific correctness" in report.interpretation_note


def test_recall_report_cli_is_read_only(tmp_path: Path) -> None:
    source = tmp_path / "simple.csv"
    source.write_text(
        "subject_id,status\n"
        + "\n".join(
            [f"S{index:03d}," + ("Yes" if index % 2 else "No") for index in range(1, 31)]
        )
        + "\n",
        encoding="utf-8",
    )
    original = source.read_bytes()
    output = tmp_path / "recall.json"

    result = CliRunner().invoke(
        app,
        ["recall-report", str(source), "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert source.read_bytes() == original
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["source_sha256"]
    assert "no Ollama calls" in result.output
