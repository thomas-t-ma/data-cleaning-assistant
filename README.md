# Data Cleaning Assistant

A local-first, research-aware data quality and cleaning assistant. The project is being built in two layers:

1. **Generic data cleaning** for CSV/Excel/Parquet files.
2. **Research Mode** for subject/visit/specimen structure, longitudinal data, reproducibility, and conservative scientific QA.

Current version: **v0.2.0**.

The engine is intentionally **read-only** at this stage. It profiles data and proposes issues without modifying the source file.

## What v0.2 does

### Generic profiling

- Reads CSV, XLSX/XLSM, and Parquet.
- Treats each Excel sheet as a separate table.
- Produces per-column profiles and semantic type guesses.
- Detects hidden missing-value tokens (`N/A`, `unknown`, `.`, etc.).
- Detects exact duplicate rows.
- Detects constant columns.
- Detects capitalization/whitespace category inconsistencies.
- Detects mostly-numeric columns containing non-numeric values.
- Detects malformed values in date-like columns.
- Flags likely identifier fields.
- Flags extreme numeric values using a conservative 3×IQR review rule.
- Detects unnamed/blank spreadsheet headers.
- Suggests highly similar category labels using conservative fuzzy matching.

### Research-aware structure

- Infers candidate single-column and composite keys.
- Detects repeated subject/entity identifiers without calling them duplicates.
- Infers initial table grain such as `subject`, `visit`, `specimen`, `procedure`, `medication`, `event`, or generic `observation`.
- Discovers identifier relationships across Excel sheets/tables.
- Estimates one-to-one / one-to-many / many-to-one / many-to-many relationships.

### Safety / reproducibility

- Writes a structured JSON report for future UI and AI layers.
- Never changes the input dataset.
- Treats suspicious values as review items rather than facts that must be "fixed."

## Setup

Python 3.11+ is recommended.

### macOS / Linux

```bash
cd cleaning-assistant
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Windows PowerShell

```powershell
cd cleaning-assistant
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Profile the deliberately messy example

```bash
python examples/make_messy_example.py
data-cleaner profile examples/messy_research_example.xlsx
```

Write the complete machine-readable report:

```bash
data-cleaner profile examples/messy_research_example.xlsx -o examples/example_report.json
```

Profile your own file:

```bash
data-cleaner profile /path/to/data.xlsx -o report.json
```

## Run tests

```bash
pytest -q
```

Optional linting:

```bash
ruff check src tests
```

GitHub Actions runs the suite on Python 3.11, 3.12, and 3.13.

## Architecture

```text
File
  ↓
load_dataset()
  ↓
Pandas DataFrames (one per table/sheet)
  ├── profiler ─────────────────────→ column/table profiles
  ├── deterministic detectors ─────→ Issue[]
  ├── structural inference ─────────→ candidate keys + table grain
  └── cross-table inference ────────→ relationships
                                        ↓
                                  DatasetReport
                                        ↓
                              CLI / JSON / future UI
                                        ↓
                           future structured AI planner
```

The core design rule is that detections are represented as structured objects rather than immediately editing the dataframe. Later versions add proposed `Action` objects, review/approval, reversible `ChangeSet`s, cleaning recipes, and an audit log.

## Next milestone: v0.3

The next release is the first safe editing layer:

- structured `Action` objects
- before/after previews
- accept / reject / modify decisions
- reversible `ChangeSet`s
- deterministic execution
- cleaning recipe export
- audit log export

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the longer plan.

## Scientific safety principle

An extreme value, repeated identifier, unexpected date, missing value, or unusual category can be scientifically legitimate. The assistant distinguishes **suspicious** from **wrong**. Ambiguous findings are flagged for review rather than silently changed or deleted.

## License

No open-source license has been selected yet. Until we decide how the commercial/research product should be licensed, all rights are reserved by the repository owner.
