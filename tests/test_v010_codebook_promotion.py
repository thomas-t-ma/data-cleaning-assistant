from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from data_cleaning_assistant.cli import app
from data_cleaning_assistant.codebook import (
    load_codebook_suggestion_report,
    promote_codebook_mappings,
    suggest_codebooks,
)
from data_cleaning_assistant.io import load_dataset
from data_cleaning_assistant.study_schema import load_study_schema


def _source(tmp_path: Path) -> Path:
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
    return source


def test_explicit_codebook_promotion_exports_only_canonical_map(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    tables = load_dataset(source).tables
    report = suggest_codebooks(source, tables)

    schema = promote_codebook_mappings(
        report,
        tables,
        {("study", "status")},
    )

    spec = schema.tables["study"].columns["status"]
    assert spec.canonical_map == {"active": "Active"}
    assert spec.allowed_values is None
    assert spec.missing_tokens == []


def test_codebook_promotion_requires_explicit_selection(tmp_path: Path) -> None:
    source = _source(tmp_path)
    tables = load_dataset(source).tables
    report = suggest_codebooks(source, tables)

    with pytest.raises(ValueError, match="No codebook targets were approved"):
        promote_codebook_mappings(report, tables, set())


def test_codebook_promotion_rejects_unknown_target(tmp_path: Path) -> None:
    source = _source(tmp_path)
    tables = load_dataset(source).tables
    report = suggest_codebooks(source, tables)

    with pytest.raises(ValueError, match="Unknown codebook suggestion"):
        promote_codebook_mappings(
            report,
            tables,
            {("study", "not_a_column")},
        )


def test_codebook_promotion_rejects_tied_variant_group(tmp_path: Path) -> None:
    source = tmp_path / "ties.csv"
    rows = ["Closed"] * 30 + ["Open"] * 5 + ["open"] * 5
    source.write_text("status\n" + "\n".join(rows) + "\n", encoding="utf-8")
    tables = load_dataset(source).tables
    report = suggest_codebooks(source, tables)

    with pytest.raises(ValueError, match="no unambiguous canonical mapping"):
        promote_codebook_mappings(
            report,
            tables,
            {("ties", "status")},
        )


def test_codebook_suggestion_report_round_trips_yaml(tmp_path: Path) -> None:
    source = _source(tmp_path)
    tables = load_dataset(source).tables
    report = suggest_codebooks(source, tables)
    report_path = tmp_path / "codebooks.yaml"
    report_path.write_text(
        yaml.safe_dump(report.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )

    loaded = load_codebook_suggestion_report(report_path)

    assert loaded.source_sha256 == report.source_sha256
    assert loaded.suggestions[0].canonical_map == {"active": "Active"}


def test_codebook_promote_cli_requires_explicit_review_and_preserves_rare_values(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    suggestions = tmp_path / "codebooks.yaml"
    schema_path = tmp_path / "study_schema.yaml"

    suggest = CliRunner().invoke(
        app,
        ["codebook-suggest", str(source), "-o", str(suggestions)],
    )
    assert suggest.exit_code == 0, suggest.output

    promote = CliRunner().invoke(
        app,
        [
            "codebook-promote",
            str(source),
            str(suggestions),
            "--approve",
            "study::status",
            "-o",
            str(schema_path),
        ],
    )
    assert promote.exit_code == 0, promote.output
    normalized_output = " ".join(promote.output.split())
    assert "No dataset edits were applied" in normalized_output

    schema = load_study_schema(schema_path)
    spec = schema.tables["study"].columns["status"]
    assert spec.canonical_map == {"active": "Active"}
    assert spec.allowed_values is None


def test_codebook_promote_cli_rejects_stale_source(tmp_path: Path) -> None:
    source = _source(tmp_path)
    suggestions = tmp_path / "codebooks.yaml"
    tables = load_dataset(source).tables
    report = suggest_codebooks(source, tables)
    suggestions.write_text(
        yaml.safe_dump(report.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    source.write_text(source.read_text(encoding="utf-8") + "Closed\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "codebook-promote",
            str(source),
            str(suggestions),
            "--approve",
            "study::status",
        ],
    )

    assert result.exit_code != 0
    assert "no longer matches" in result.output
