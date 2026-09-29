from __future__ import annotations

import csv
from pathlib import Path


def _write_raw(path: Path, rows: int = 1862) -> None:
    header = [
        "Timestamp",
        "1. Paper number",
        "2. DOI of paper",
        *[f"Question {index}" for index in range(17)],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index in range(rows):
            writer.writerow(
                [
                    "2025-09-29 09:00:00",
                    str(index),
                    f"10.0000/example.{index}",
                    *["Yes"] * 17,
                ]
            )


def _write_clean(path: Path, rows: int = 1861) -> None:
    header = [
        "paper_number",
        "doi",
        "year_published",
        "journal",
        "article_type",
        "data_used",
        "data_availability",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index in range(rows):
            writer.writerow(
                [
                    index,
                    f"10.0000/example.{index}",
                    2024,
                    "Journal of Ecology",
                    "Research Article",
                    "Yes",
                    "Yes",
                ]
            )


def test_bes_fetcher_validates_expected_raw_and_cleaned_shapes(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from fetch_bes_hackathon import validate_sources

    raw = tmp_path / "raw.csv"
    cleaned = tmp_path / "cleaned.csv"
    _write_raw(raw)
    _write_clean(cleaned)

    result = validate_sources(raw, cleaned)
    assert result["raw_rows"] == 1862
    assert result["raw_columns"] == 20
    assert result["raw_encoding"] == "utf-8-sig"
    assert result["clean_rows"] == 1861
    assert result["clean_columns"] == 7
    assert result["clean_encoding"] == "utf-8-sig"


def test_bes_fetcher_rejects_wrong_cleaned_row_count(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from fetch_bes_hackathon import validate_sources

    raw = tmp_path / "raw.csv"
    cleaned = tmp_path / "cleaned.csv"
    _write_raw(raw)
    _write_clean(cleaned, rows=1800)

    try:
        validate_sources(raw, cleaned)
    except ValueError as exc:
        assert "row count differs" in str(exc)
    else:
        raise AssertionError("Expected invalid cleaned row count to fail.")


def test_bes_fetcher_accepts_explicit_cp1252_reference(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from fetch_bes_hackathon import validate_sources

    raw = tmp_path / "raw.csv"
    cleaned = tmp_path / "cleaned.csv"
    _write_raw(raw)
    _write_clean(cleaned)

    text = cleaned.read_text(encoding="utf-8")
    # Insert a CP1252-only byte (Œ = 0x8C) into a non-key free-text field.
    text = text.replace("Journal of Ecology", "Journal Œcology", 1)
    cleaned.write_bytes(text.encode("cp1252"))

    result = validate_sources(raw, cleaned)
    assert result["raw_encoding"] == "utf-8-sig"
    assert result["clean_encoding"] == "cp1252"
    assert result["clean_rows"] == 1861
