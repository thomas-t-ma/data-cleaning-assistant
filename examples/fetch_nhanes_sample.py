"""Download and validate a public NHANES-derived CSV for workflow-pilot use.

Source:
  OpenIntro nhanes.samp
  https://www.openintro.org/data/index.php?data=nhanes.samp
  https://www.openintro.org/data/csv/nhanes.samp.csv

The dataset is a 200-row sample of the NHANES teaching dataset with 76 variables.
It is derived from real US National Health and Nutrition Examination Survey data
and is intended for educational/statistical use, not as an analysis-ready research
database.

The downloader writes only into a fresh local output directory. It never modifies
another dataset and does not send the CSV to any model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

PAGE_URL = "https://www.openintro.org/data/index.php?data=nhanes.samp"
DOWNLOAD_URL = "https://www.openintro.org/data/csv/nhanes.samp.csv"
DATASET = "nhanes_openintro_sample.csv"
MANIFEST = "source_manifest.json"
EXPECTED_ROWS = 200
EXPECTED_COLUMNS = 76


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_csv(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("Downloaded NHANES CSV is empty.") from exc
        rows = sum(1 for row in reader if row)

    if len(header) != EXPECTED_COLUMNS or rows != EXPECTED_ROWS:
        raise ValueError(
            "Unexpected NHANES sample shape: "
            f"rows={rows}, columns={len(header)}; "
            f"expected {EXPECTED_ROWS}x{EXPECTED_COLUMNS}."
        )

    required = {"ID", "SurveyYr", "Gender", "Age", "Race1", "BMI"}
    missing = sorted(required - set(header))
    if missing:
        raise ValueError(
            "Downloaded file does not look like the expected NHANES sample; "
            f"missing columns: {missing}"
        )
    return {"rows": rows, "columns": len(header)}


def fetch(out: Path) -> Path:
    out = out.expanduser().resolve()
    if out.exists():
        raise FileExistsError(
            f"Output directory already exists: {out}. "
            "Choose a fresh path to preserve source provenance."
        )
    out.mkdir(parents=True, exist_ok=False)
    destination = out / DATASET

    request = Request(
        DOWNLOAD_URL,
        headers={"User-Agent": "DataCleaningAssistant/0.8 workflow-pilot"},
    )
    try:
        with urlopen(request, timeout=60) as response, destination.open("wb") as handle:
            content_type = response.headers.get("Content-Type", "")
            if content_type and "csv" not in content_type.casefold() and "text" not in content_type.casefold():
                raise ValueError(
                    f"Unexpected content type from OpenIntro: {content_type!r}"
                )
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        try:
            out.rmdir()
        except OSError:
            pass
        raise

    shape = validate_csv(destination)
    manifest = {
        "version": 1,
        "purpose": "Public NHANES-derived source provenance for local workflow pilot",
        "source_page": PAGE_URL,
        "download_url": DOWNLOAD_URL,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "filename": DATASET,
        "sha256": _sha256(destination),
        "bytes": destination.stat().st_size,
        "validated_shape": shape,
        "notes": (
            "OpenIntro documents nhanes.samp as a 200-observation sample with "
            "76 variables from the NHANES teaching dataset. The NHANES package "
            "data are adapted for educational purposes and should not be treated "
            "as an analysis-ready research database. No model call or data edit occurred."
        ),
    }
    (out / MANIFEST).write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print("NHANES public sample downloaded and validated.")
    print(f"Rows: {shape['rows']}")
    print(f"Columns: {shape['columns']}")
    print(f"SHA-256: {manifest['sha256']}")
    print(f"CSV: {destination}")
    print(f"Manifest: {out / MANIFEST}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("pilot_outputs/nhanes_public"),
        help="Fresh output directory for the public CSV and manifest",
    )
    args = parser.parse_args()
    fetch(args.out)


if __name__ == "__main__":
    main()
