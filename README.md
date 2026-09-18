# Data Cleaning Assistant

A local-first, research-aware data quality and cleaning assistant. It has two layers:

1. **Generic data cleaning** for CSV/Excel/Parquet files.
2. **Research Mode** for subject/visit/specimen structure, longitudinal data, reproducibility, and conservative scientific QA.

Current version: **v0.3.1**.

The core rule is simple: **detection is not transformation**. Profiling is read-only, proposed fixes are reviewable, and cleaning writes a new output file with an audit trail.

## What it does

The profiler currently detects hidden missing-value tokens, exact duplicate rows, constant columns, capitalization/whitespace category inconsistencies, mostly-numeric columns with non-numeric values, malformed date-like values, likely identifier fields, unnamed columns, conservative statistical outliers, candidate keys, repeated entities, dataset grain, and cross-table relationships.

The transformation layer currently supports reviewable category normalization, missing-token normalization, date standardization, and numeric coercion for non-identifier columns.

v0.3.1 adds two major usability features:

- **Interactive review** — approve/reject/defer proposed actions without hand-editing JSON.
- **Recipe replay** — validate and preview a previously approved YAML recipe on another compatible dataset, then explicitly apply it if desired.

Single-table CSV/Parquet recipes can be replayed across differently named files. Multi-sheet workbooks require compatible sheet names and columns.

Deletion, outlier removal, fuzzy-category merging, and imputation are still not automatically proposed.

## One-command setup / sync

GitHub is the source of truth.

First time on macOS/Linux:

```bash
git clone https://github.com/thomas-t-ma/data-cleaning-assistant.git && cd data-cleaning-assistant && bash sync.sh
```

Normal update:

```bash
bash sync.sh
```

The sync script verifies the virtual environment, repairs missing dependencies, recreates a broken venv if pip is missing, and runs tests.

## Basic workflow

Create the example workbook:

```bash
.venv/bin/python examples/make_messy_example.py
```

Generate a plan:

```bash
.venv/bin/data-cleaner plan examples/messy_research_example.xlsx -o cleaning_plan.json
```

Interactively review it:

```bash
.venv/bin/data-cleaner review \
  examples/messy_research_example.xlsx \
  cleaning_plan.json \
  -o reviewed_plan.json
```

Apply only the actions you approved during review:

```bash
.venv/bin/data-cleaner apply \
  examples/messy_research_example.xlsx \
  reviewed_plan.json \
  -o cleaned.xlsx
```

This writes:

```text
cleaned.xlsx
cleaned.xlsx.audit.json
cleaned.xlsx.recipe.yaml
```

Undo:

```bash
.venv/bin/data-cleaner undo \
  cleaned.xlsx \
  cleaned.xlsx.audit.json \
  -o restored.xlsx
```

## Replay a saved recipe

Preview only:

```bash
.venv/bin/data-cleaner replay \
  another_dataset.xlsx \
  cleaned.xlsx.recipe.yaml
```

If the compatibility check and previews look correct, apply explicitly:

```bash
.venv/bin/data-cleaner replay \
  another_dataset.xlsx \
  cleaned.xlsx.recipe.yaml \
  --apply \
  -o another_dataset_cleaned.xlsx
```

A recipe with missing tables or columns is rejected instead of partially applied.

## Safety / reproducibility

- Source files are never edited in place.
- Plans are bound to source SHA-256 hashes.
- Every changed cell stores its old/new value.
- Cleaned outputs are hashed for safe undo.
- Saved recipes contain only actions that were actually executed.
- Recipe replay previews before writing and requires `--apply`.
- Ambiguous scientific findings are review items, not automatic corrections.

## Run tests

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src tests
```

GitHub Actions runs on Python 3.11, 3.12, and 3.13.

## License

No open-source license has been selected yet. Until we decide how the commercial/research product should be licensed, all rights are reserved by the repository owner.
