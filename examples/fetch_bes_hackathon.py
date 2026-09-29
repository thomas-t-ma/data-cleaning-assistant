"""Fetch the pinned public BES hackathon raw/cleaned research datasets.

Upstream:
  Natalie Cooper and the BES Data and Code Hackathon Group.
  Data and code for MEE paper.
  https://github.com/nhcooper123/reproduce-reuse-recycle
  Zenodo DOI: 10.5281/zenodo.19737911

The raw file is the anonymized Google Form output used by the authors before
cleaning. The cleaned file is retained only as a reference for the workflow
pilot. No data edits or model calls occur in this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

PINNED_COMMIT = "ef25b60f333452b0baad7a5a4d0bccc6386e67af"
REPO_URL = "https://github.com/nhcooper123/reproduce-reuse-recycle"
ZENODO_DOI = "10.5281/zenodo.19737911"
RAW_NAME = "BES-data-code-hackathon-raw-outputs_ANON_2025-11-16.csv"
CLEAN_NAME = "BES-data-code-hackathon-cleaned_2025-12-01.csv"
RAW_URL = (
    "https://raw.githubusercontent.com/nhcooper123/reproduce-reuse-recycle/"
    f"{PINNED_COMMIT}/raw-data/{RAW_NAME}"
)
CLEAN_URL = (
    "https://raw.githubusercontent.com/nhcooper123/reproduce-reuse-recycle/"
    f"{PINNED_COMMIT}/data/{CLEAN_NAME}"
)
EXPECTED_CLEAN_ROWS = 1861


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    request = Request(
        url,
        headers={"User-Agent": "DataCleaningAssistant/0.9 research-workflow-pilot"},
    )
    with urlopen(request, timeout=120) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def _detect_text_encoding(path: Path) -> str:
    """Prefer UTF-8, with an explicit Windows-1252 fallback for upstream CSVs."""
    payload = path.read_bytes()
    if not payload:
        raise ValueError(f"Downloaded CSV is empty: {path.name}")
    if b"\x00" in payload[:8192]:
        raise ValueError(f"Downloaded file does not look like text CSV: {path.name}")

    try:
        payload.decode("utf-8-sig")
        return "utf-8-sig"
    except UnicodeDecodeError:
        pass

    try:
        payload.decode("cp1252")
        return "cp1252"
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Downloaded CSV is neither valid UTF-8 nor Windows-1252: {path.name}"
        ) from exc


def _csv_shape(path: Path) -> tuple[list[str], int, str]:
    encoding = _detect_text_encoding(path)
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Downloaded CSV is empty: {path.name}") from exc
        rows = sum(1 for row in reader if row)
    return header, rows, encoding


def validate_sources(raw_path: Path, cleaned_path: Path) -> dict[str, object]:
    raw_header, raw_rows, raw_encoding = _csv_shape(raw_path)
    clean_header, clean_rows, clean_encoding = _csv_shape(cleaned_path)

    raw_requirements = {
        "timestamp": "Timestamp" in raw_header,
        "paper_number": any("Paper number" in column for column in raw_header),
        "doi": any("DOI" in column for column in raw_header),
    }
    if not all(raw_requirements.values()) or len(raw_header) < 20:
        raise ValueError(
            "Raw BES file does not match the expected anonymized Google Form export."
        )
    if raw_rows < 1800:
        raise ValueError(
            f"Raw BES row count is unexpectedly small: {raw_rows}."
        )

    required_clean = {
        "paper_number",
        "doi",
        "year_published",
        "journal",
        "article_type",
        "data_used",
        "data_availability",
    }
    missing = sorted(required_clean - set(clean_header))
    if missing:
        raise ValueError(
            "Clean BES reference is missing expected columns: " + ", ".join(missing)
        )
    if clean_rows != EXPECTED_CLEAN_ROWS:
        raise ValueError(
            "Clean BES reference row count differs from the upstream README: "
            f"{clean_rows} != {EXPECTED_CLEAN_ROWS}."
        )
    if raw_rows < clean_rows:
        raise ValueError("Raw BES file has fewer rows than its cleaned reference.")

    return {
        "raw_rows": raw_rows,
        "raw_columns": len(raw_header),
        "raw_encoding": raw_encoding,
        "clean_rows": clean_rows,
        "clean_columns": len(clean_header),
        "clean_encoding": clean_encoding,
    }


def fetch(out: Path) -> dict[str, object]:
    out = out.expanduser().resolve()
    if out.exists():
        raise FileExistsError(
            f"Output directory already exists: {out}. "
            "Choose a fresh path to preserve provenance."
        )
    out.mkdir(parents=True, exist_ok=False)
    raw_path = out / RAW_NAME
    clean_path = out / CLEAN_NAME

    try:
        _download(RAW_URL, raw_path)
        _download(CLEAN_URL, clean_path)
        shape = validate_sources(raw_path, clean_path)
    except Exception:
        for path in (raw_path, clean_path):
            path.unlink(missing_ok=True)
        try:
            out.rmdir()
        except OSError:
            pass
        raise

    manifest = {
        "version": 1,
        "purpose": "Pinned public BES real-workflow pilot source provenance",
        "upstream_repository": REPO_URL,
        "upstream_commit": PINNED_COMMIT,
        "zenodo_doi": ZENODO_DOI,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "raw": {
            "filename": RAW_NAME,
            "url": RAW_URL,
            "sha256": _sha256(raw_path),
            "bytes": raw_path.stat().st_size,
            "rows": shape["raw_rows"],
            "columns": shape["raw_columns"],
            "encoding": shape["raw_encoding"],
        },
        "cleaned_reference": {
            "filename": CLEAN_NAME,
            "url": CLEAN_URL,
            "sha256": _sha256(clean_path),
            "bytes": clean_path.stat().st_size,
            "rows": shape["clean_rows"],
            "columns": shape["clean_columns"],
            "encoding": shape["clean_encoding"],
        },
        "notes": (
            "Upstream describes the raw file as anonymized Google Form research "
            "data and the cleaned file as the processed analysis dataset. The "
            "cleaned file is a qualitative reference only; no answer key is "
            "inferred and no edits or model calls occur during fetch."
        ),
    }
    (out / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print("BES public research sources downloaded and validated.")
    print(
        f"Raw: {shape['raw_rows']} rows x {shape['raw_columns']} columns "
        f"[{shape['raw_encoding']}] ({raw_path})"
    )
    print(
        f"Cleaned reference: {shape['clean_rows']} rows x "
        f"{shape['clean_columns']} columns [{shape['clean_encoding']}] "
        f"({clean_path})"
    )
    print(f"Pinned upstream commit: {PINNED_COMMIT}")
    print(f"Manifest: {out / 'source_manifest.json'}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("pilot_outputs/bes_public"),
    )
    args = parser.parse_args()
    fetch(args.out)


if __name__ == "__main__":
    main()
