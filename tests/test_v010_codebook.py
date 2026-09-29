from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from data_cleaning_assistant.cli import app
from data_cleaning_assistant.codebook import suggest_codebooks
from data_cleaning_assistant.io import load_dataset


def test_codebook_suggestion_separates_core_labels_from_rare_writeins(
    tmp_path: Path,
) -> None:
    source = tmp_path / "study.csv"
    rows = (
        ["Active"] * 30
        + ["active"] * 3
        + ["Inactive"] * 6
        + ["reviewer-specific note"]
    )
    source.write_text(
        "status\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )

    loaded = load_dataset(source)
    report = suggest_codebooks(source, loaded.tables)

    assert len(report.suggestions) == 1
    suggestion = report.suggestions[0]
    assert suggestion.core_values == ["Active", "Inactive"]
    assert suggestion.canonical_map == {"active": "Active"}
    assert suggestion.rare_values == ["reviewer-specific note"]
    assert suggestion.core_coverage >= 0.95
    assert suggestion.review_required is True


def test_codebook_ties_remain_unresolved(tmp_path: Path) -> None:
    source = tmp_path / "study.csv"
    rows = ["Closed"] * 30 + ["Open"] * 5 + ["open"] * 5
    source.write_text(
        "status\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )

    loaded = load_dataset(source)
    report = suggest_codebooks(source, loaded.tables)

    suggestion = report.suggestions[0]
    assert suggestion.canonical_map == {}
    assert suggestion.unresolved_variant_groups == [["Open", "open"]]


def test_codebook_suggestion_skips_free_text(tmp_path: Path) -> None:
    source = tmp_path / "comments.csv"
    source.write_text(
        "comments\n"
        + "\n".join(
            [
                f"Detailed reviewer comment number {index} with several words"
                for index in range(1, 31)
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_dataset(source)
    report = suggest_codebooks(source, loaded.tables)

    assert report.suggestions == []


def test_codebook_cli_is_read_only_and_writes_review_report(
    tmp_path: Path,
) -> None:
    source = tmp_path / "study.csv"
    rows = ["Yes"] * 25 + ["No"] * 10 + ["Unsure"] * 4 + ["manual note"]
    source.write_text(
        "status\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    original = source.read_bytes()
    output = tmp_path / "codebooks.yaml"

    result = CliRunner().invoke(
        app,
        ["codebook-suggest", str(source), "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert source.read_bytes() == original
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert payload["source_sha256"]
    assert payload["suggestions"]
    assert "authorize no transformations" in result.output
