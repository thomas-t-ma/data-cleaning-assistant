# Data Cleaning Assistant

A local-first, research-aware data quality and cleaning assistant. The project is being built in two layers:

1. **Generic data cleaning** for CSV/Excel/Parquet files.
2. **Research Mode** for subject/visit/specimen structure, longitudinal data, reproducibility, and conservative scientific QA.

Current version: **v0.3.0**.

The system separates **detection** from **transformation**. Profiling remains read-only; cleaning happens only through explicit, reviewable actions applied to a new output file.

## What v0.3 does

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

### Safe transformations

v0.3 can generate structured proposals for:

- category capitalization/whitespace normalization
- textual missing-value normalization
- date standardization
- numeric coercion for non-identifier columns

Every action starts as **proposed**, not approved.

The transformation layer supports:

- before/after cell previews
- explicit per-action approval
- explicit `--approve-all` when desired
- source-file hashing to reject stale cleaning plans
- deep-copy execution so the original input is never modified
- cell-level audit logs
- reproducible YAML recipes
- reversible ChangeSets
- cleaned-file hashing so undo refuses to run against the wrong file

Deletion, outlier removal, fuzzy-category merging, and imputation are **not** automatically proposed in v0.3.

## One-command setup / sync

GitHub is the source of truth.

### First time on macOS / Linux

```bash
git clone https://github.com/thomas-t-ma/data-cleaning-assistant.git && cd data-cleaning-assistant && bash sync.sh
```

### Normal update

```bash
bash sync.sh
```

See [`docs/WORKFLOW.md`](docs/WORKFLOW.md) for Windows and cross-device development details.

## Basic workflow

Create the deliberately messy example:

```bash
.venv/bin/python examples/make_messy_example.py
```

Profile it:

```bash
.venv/bin/data-cleaner profile examples/messy_research_example.xlsx
```

Generate a cleaning plan:

```bash
.venv/bin/data-cleaner plan examples/messy_research_example.xlsx -o cleaning_plan.json
```

Preview every proposed action:

```bash
.venv/bin/data-cleaner preview examples/messy_research_example.xlsx cleaning_plan.json
```

Apply selected actions:

```bash
.venv/bin/data-cleaner apply \
  examples/messy_research_example.xlsx \
  cleaning_plan.json \
  --approve action_ABC123 \
  --approve action_DEF456 \
  -o cleaned.xlsx
```

Or, after reviewing the plan, explicitly approve every non-rejected action:

```bash
.venv/bin/data-cleaner apply \
  examples/messy_research_example.xlsx \
  cleaning_plan.json \
  --approve-all \
  -o cleaned.xlsx
```

This writes:

```text
cleaned.xlsx
cleaned.xlsx.audit.json
cleaned.xlsx.recipe.yaml
```

Undo the ChangeSet into another file:

```bash
.venv/bin/data-cleaner undo \
  cleaned.xlsx \
  cleaned.xlsx.audit.json \
  -o restored.xlsx
```

## Architecture

```text
File
  ↓
load_dataset()
  ↓
profiling + deterministic detectors
  ↓
DatasetReport / Issue[]
  ↓
deterministic planner
  ↓
CleaningPlan / proposed Action[]
  ↓
preview
  ↓
explicit approval
  ↓
deep-copy deterministic execution
  ↓
cleaned dataset + ChangeSet + audit + recipe
  ↓
optional undo
```

The future AI layer will propose the same validated `CleaningAction` objects. It will not get unrestricted dataframe-write access.

## Run tests

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src tests
```

GitHub Actions runs the suite on Python 3.11, 3.12, and 3.13.

## Scientific safety principle

An extreme value, repeated identifier, unexpected date, missing value, or unusual category can be scientifically legitimate. The assistant distinguishes **suspicious** from **wrong**. Ambiguous findings are flagged for review rather than silently changed or deleted.

## License

No open-source license has been selected yet. Until we decide how the commercial/research product should be licensed, all rights are reserved by the repository owner.
