from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from data_cleaning_assistant.cli import app
from data_cleaning_assistant.io import load_dataset
from data_cleaning_assistant.recode import suggest_recodes


def test_recode_suggestion_collapses_case_equivalence_before_typo_review(
    tmp_path: Path,
) -> None:
    source = tmp_path / "status.csv"
    source.write_text(
        "status\n"
        + "Unclear\n" * 20
        + "unclear\n" * 5
        + "Uclear\n"
        + "Known\n" * 10,
        encoding="utf-8",
    )

    report = suggest_recodes(source, load_dataset(source).tables)

    matches = [
        suggestion
        for suggestion in report.suggestions
        if suggestion.column == "status"
    ]
    assert len(matches) == 1
    suggestion = matches[0]
    assert suggestion.suggested_target == "Response Unclear"
    assert suggestion.target_basis == "observed_frequency_dominance"
    assert suggestion.left.total_count + suggestion.right.total_count == 26
    values = set(suggestion.left.values) | set(suggestion.right.values)
    assert values == {"Response Unclear", "response unclear", "Response Uclear"}


def test_recode_suggestion_keeps_equal_frequency_variants_unresolved(
    tmp_path: Path,
) -> None:
    source = tmp_path / "license.csv"
    source.write_text(
        "license\n"
        + "Creative Commons Attribution Internation 4.0\n"
        + "CC BY;Creative Commons Attribution 4.0 International\n"
        + "MIT\n" * 12
        + "GPL\n" * 8,
        encoding="utf-8",
    )

    report = suggest_recodes(source, load_dataset(source).tables)

    matches = [
        suggestion
        for suggestion in report.suggestions
        if "Internation" in " ".join(
            [*suggestion.left.values, *suggestion.right.values]
        )
    ]
    assert len(matches) == 1
    assert matches[0].suggested_target is None
    assert matches[0].component_mode is True


def test_recode_suggestion_never_groups_different_numeric_versions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "license.csv"
    source.write_text(
        "license\n"
        + "GNU General Public License v2.0\n" * 10
        + "GNU General Public License v3\n" * 10,
        encoding="utf-8",
    )

    report = suggest_recodes(source, load_dataset(source).tables)

    assert report.suggestions == []


def test_recode_suggestion_rejects_single_token_semantic_neighbors(
    tmp_path: Path,
) -> None:
    source = tmp_path / "country.csv"
    source.write_text(
        "country\n"
        + "Australia\n" * 30
        + "Austria\n" * 3
        + "Ireland\n" * 10
        + "Iceland\n",
        encoding="utf-8",
    )

    report = suggest_recodes(source, load_dataset(source).tables)

    assert report.suggestions == []


def test_recode_suggestion_skips_free_text_and_multiselect(
    tmp_path: Path,
) -> None:
    source = tmp_path / "mixed.csv"
    source.write_text(
        "comments,language\n"
        + '"Detailed reviewer comment about Internation spelling","R;BUGS"\n' * 6
        + '"Another detailed reviewer comment about International spelling","R;Bugs"\n' * 6,
        encoding="utf-8",
    )

    report = suggest_recodes(source, load_dataset(source).tables)

    assert report.suggestions == []


def test_recode_suggest_cli_is_read_only(tmp_path: Path) -> None:
    source = tmp_path / "status.csv"
    source.write_text(
        "status\n"
        + "Unclear\n" * 20
        + "Uclear\n"
        + "Known\n" * 10,
        encoding="utf-8",
    )
    original = source.read_bytes()
    output = tmp_path / "recode.yaml"

    result = CliRunner().invoke(
        app,
        ["recode-suggest", str(source), "-o", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert source.read_bytes() == original
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert payload["suggestions"]
    normalized_output = " ".join(result.output.split())
    assert "authorize no transformations" in normalized_output
