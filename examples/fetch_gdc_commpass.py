"""Fetch *open* NCI GDC MMRF-CoMMpass clinical metadata into linked pilot tables.

Run locally:
    .venv/bin/python examples/fetch_gdc_commpass.py --limit 100

Requires network access to api.gdc.cancer.gov; no GDC credentials or
controlled-access genomic data are requested. Raw responses are preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from openpyxl import Workbook

API = "https://api.gdc.cancer.gov"
PROJECT = "MMRF-COMMPASS"
EXPAND = (
    "demographic,diagnoses,diagnoses.treatments,"
    "follow_ups,follow_ups.molecular_tests"
)
TABLES = ("Cases", "Diagnoses", "Treatments", "FollowUps", "LabTests")


def get_json(url: str) -> dict[str, Any]:
    """Retry only transient failures; never silently return partial data."""
    for attempt in range(4):
        try:
            request = Request(url, headers={"User-Agent": "data-cleaning-assistant-public-pilot/1"})
            with urlopen(request, timeout=90) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise TypeError("Unexpected non-object GDC API response")
            return payload
        except HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 3:
                raise
        except (TimeoutError, URLError):
            if attempt == 3:
                raise
        time.sleep(2**attempt)
    raise AssertionError("Unreachable retry state")


def fetch_cases(limit: int, page_size: int = 10) -> tuple[list[dict[str, Any]], list[str], int]:
    """Fetch a stable, paginated case cohort. Limit=0 requests all available cases."""
    filters = {
        "op": "in",
        "content": {"field": "project.project_id", "value": [PROJECT]},
    }
    hits: list[dict[str, Any]] = []
    urls: list[str] = []
    total: int | None = None
    while total is None or (len(hits) < total and (limit == 0 or len(hits) < limit)):
        remaining = page_size if limit == 0 else min(page_size, limit - len(hits))
        params = {
            "filters": json.dumps(filters, separators=(",", ":")),
            "fields": "case_id,project.project_id,disease_type,primary_site",
            "expand": EXPAND,
            "size": remaining,
            "from": len(hits),
            "sort": "case_id:asc",
        }
        url = f"{API}/cases?{urlencode(params)}"
        response = get_json(url)
        data = response.get("data")
        if not isinstance(data, dict):
            raise TypeError("GDC response missing data object")
        pagination = data.get("pagination", {})
        current_total = pagination.get("total")
        if not isinstance(current_total, int):
            raise TypeError("GDC response missing cohort count")
        if total is not None and current_total != total:
            raise ValueError("GDC cohort changed during pagination; rerun the download")
        total = current_total
        page = data.get("hits")
        if not isinstance(page, list) or not page:
            raise ValueError("GDC returned an empty or malformed page before completion")
        for case in page:
            if not isinstance(case, dict):
                raise TypeError("GDC case record is not an object")
            project_id = (case.get("project") or {}).get("project_id")
            if project_id != PROJECT or not case.get("case_id"):
                raise ValueError("GDC returned a case outside the requested project")
        hits.extend(page)
        urls.append(url)
        print(f"GDC cases: {len(hits)} / {total} available", flush=True)

    case_ids = [case["case_id"] for case in hits]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Duplicate case IDs across GDC pages")
    if limit and len(hits) != min(limit, total):
        raise ValueError("Incomplete GDC case download")
    if limit == 0 and len(hits) != total:
        raise ValueError("Incomplete GDC cohort download")
    return hits, urls, total


def _scalar(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return value


def _flat(entity: dict[str, Any], excluded: set[str]) -> dict[str, Any]:
    return {
        key: _scalar(value)
        for key, value in entity.items()
        if key not in excluded
    }


def linked_tables(cases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Keep repeated diagnoses/follow-ups/labs separate: never cross-join them."""
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in TABLES}
    for case in cases:
        case_id = str(case["case_id"])
        demographic = case.get("demographic") or {}
        if not isinstance(demographic, dict):
            demographic = {}
        rows["Cases"].append({
            "case_id": case_id,
            "project_id": PROJECT,
            "disease_type": case.get("disease_type"),
            "primary_site": case.get("primary_site"),
            **{
                f"demographic_{key}": _scalar(value)
                for key, value in demographic.items()
                if key not in {"submitter_id", "demographic_id"}
            },
        })

        for diag_index, diagnosis in enumerate(case.get("diagnoses") or [], start=1):
            diagnosis_key = f"{case_id}:D{diag_index:04d}"
            rows["Diagnoses"].append({
                "diagnosis_key": diagnosis_key,
                "case_id": case_id,
                **_flat(diagnosis, {"treatments", "submitter_id"}),
            })
            for treatment_index, treatment in enumerate(
                diagnosis.get("treatments") or [], start=1
            ):
                rows["Treatments"].append({
                    "treatment_key": f"{diagnosis_key}:T{treatment_index:04d}",
                    "diagnosis_key": diagnosis_key,
                    "case_id": case_id,
                    **_flat(treatment, {"submitter_id"}),
                })

        for follow_index, follow_up in enumerate(case.get("follow_ups") or [], start=1):
            follow_key = f"{case_id}:F{follow_index:04d}"
            rows["FollowUps"].append({
                "followup_key": follow_key,
                "case_id": case_id,
                **_flat(
                    follow_up,
                    {"molecular_tests", "other_clinical_attributes", "submitter_id"},
                ),
            })
            for test_index, test in enumerate(
                follow_up.get("molecular_tests") or [], start=1
            ):
                rows["LabTests"].append({
                    "labtest_key": f"{follow_key}:L{test_index:04d}",
                    "followup_key": follow_key,
                    "case_id": case_id,
                    **_flat(test, {"submitter_id"}),
                })
    return rows


def _safe_cell(value: Any) -> Any:
    """Store strings as literal workbook text rather than formulas."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return value


def write_workbook(path: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    wb = Workbook()
    for index, name in enumerate(TABLES):
        sheet = wb.active if index == 0 else wb.create_sheet()
        sheet.title = name
        rows = tables[name]
        headers = list(dict.fromkeys(key for row in rows for key in row))
        if not headers:
            # Empty related tables still retain their expected key columns.
            headers = {
                "Diagnoses": ["diagnosis_key", "case_id"],
                "Treatments": ["treatment_key", "diagnosis_key", "case_id"],
                "FollowUps": ["followup_key", "case_id"],
                "LabTests": ["labtest_key", "followup_key", "case_id"],
            }.get(name, ["case_id"])
        sheet.append(headers)
        for row in rows:
            sheet.append([_safe_cell(row.get(header)) for header in headers])
            for cell in sheet[sheet.max_row]:
                if isinstance(cell.value, str):
                    cell.data_type = "s"
        sheet.freeze_panes = "A2"
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def build_schema(tables: dict[str, list[dict[str, Any]]]) -> str:
    """Static keys/relationships only. Never guess clinical category meanings."""
    spec = {
        "version": 1,
        "tables": {
            "Cases": {"grain_key": ["case_id"]},
            "Diagnoses": {"grain_key": ["diagnosis_key"]},
            "Treatments": {"grain_key": ["treatment_key"]},
            "FollowUps": {"grain_key": ["followup_key"]},
            "LabTests": {"grain_key": ["labtest_key"]},
        },
        "foreign_keys": [
            {
                "child_table": "Diagnoses",
                "child_column": "case_id",
                "parent_table": "Cases",
                "parent_column": "case_id",
            },
            {
                "child_table": "Treatments",
                "child_column": "diagnosis_key",
                "parent_table": "Diagnoses",
                "parent_column": "diagnosis_key",
            },
            {
                "child_table": "FollowUps",
                "child_column": "case_id",
                "parent_table": "Cases",
                "parent_column": "case_id",
            },
            {
                "child_table": "LabTests",
                "child_column": "followup_key",
                "parent_table": "FollowUps",
                "parent_column": "followup_key",
            },
        ],
    }
    import yaml

    return yaml.safe_dump(spec, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=100,
        help="Case count; 0 downloads all available cases (default: 100)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("pilot_outputs/commpass"),
        help="Output directory (must not already contain output files)",
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be >= 0")

    target = args.output.expanduser().resolve()
    filenames = [
        "commpass_pilot.xlsx", "source_cases.json",
        "study_schema.yaml", "source_manifest.json",
    ]
    if any((target / filename).exists() for filename in filenames):
        parser.error("Output already exists; choose a new --output directory")

    cases, urls, available = fetch_cases(args.limit)
    if not cases:
        raise ValueError("No cases returned; refusing to create an empty pilot")
    tables = linked_tables(cases)
    target.mkdir(parents=True, exist_ok=True)
    raw_bytes = (json.dumps(cases, indent=2, ensure_ascii=False) + "\n").encode()
    (target / "source_cases.json").write_bytes(raw_bytes)
    (target / "study_schema.yaml").write_text(build_schema(tables), encoding="utf-8")
    write_workbook(target / "commpass_pilot.xlsx", tables)
    manifest = {
        "source": "NCI Genomic Data Commons open case metadata API",
        "project_id": PROJECT,
        "retrieved_utc": datetime.now(UTC).isoformat(),
        "api_base": API,
        "query_urls": urls,
        "available_cases_at_download": available,
        "downloaded_cases": len(cases),
        "source_cases_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "row_counts": {name: len(rows) for name, rows in tables.items()},
        "notes": (
            "Exploratory GDC clinical metadata, not the complete MMRF IA24 data release. "
            "No scientific meaning inferred or raw values normalized."
        ),
    }
    (target / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {target / 'commpass_pilot.xlsx'}")
    print(f"Row counts: {manifest['row_counts']}")


if __name__ == "__main__":
    main()
