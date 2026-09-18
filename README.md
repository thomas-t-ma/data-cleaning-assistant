# Data Cleaning Assistant

A local-first, research-aware data quality and cleaning assistant for CSV, Excel, and Parquet datasets.

Current version: **v0.3.2**.

The core rule is **detection is not transformation**. Profiling is read-only. Every proposed fix is reviewable, approved changes are written to a new file, and the tool records an audit trail plus a reusable recipe.

## Current capabilities

The profiler detects hidden missing-value tokens, exact duplicate rows, constant columns, category inconsistencies, surrounding whitespace, mixed numeric values, malformed dates, likely identifiers, unnamed columns, conservative statistical outliers, candidate keys, repeated entities, dataset grain, and cross-table relationships.

Reviewable deterministic transformations currently include:

- category normalization
- textual missing-value normalization
- date standardization
- numeric coercion for non-identifier columns
- leading/trailing whitespace trimming
- exact duplicate-row removal, keeping the first copy

Exact duplicate removal is never auto-approved because repeated identical rows can still be scientifically legitimate.

## v0.3.2 hardening

### Excel preservation

When the source and output are Excel files, the cleaner now copies the original workbook and updates only approved data cells instead of rebuilding a plain workbook from pandas. This preserves untouched workbook structure such as styles, column widths, merged cells, freeze panes, and formulas.

Row deletion or row reinsertion is refused on formula-containing sheets because worksheet row operations can invalidate formula references. Cell-level edits still preserve unrelated formulas.

### Reversible row deletion

Exact duplicate removal stores the complete deleted row plus its original position in the ChangeSet. Undo can restore those rows after the cleaned file has been saved and reloaded.

### Schema-drift reporting

Saved recipes now include a lightweight source-schema snapshot. Recipe replay reports drift such as:

- new or missing tables/columns
- inferred type changes
- new categorical values
- numeric range expansion
- meaningful missingness shifts

Drift warnings do not automatically block replay. Missing tables or columns required by an actual recipe action still do.

### Safer execution

- Source files cannot be overwritten in place.
- No-op approved actions are omitted from the audit log and recipe.
- Cell edits execute before row deletions so row positions remain reversible.
- Formula cells are not overwritten by preservation writes.

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

## Typical workflow

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

Apply approved actions:

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

Replay a saved recipe in preview-only mode:

```bash
.venv/bin/data-cleaner replay \
  another_dataset.xlsx \
  cleaned.xlsx.recipe.yaml
```

After reviewing compatibility, drift, and previews:

```bash
.venv/bin/data-cleaner replay \
  another_dataset.xlsx \
  cleaned.xlsx.recipe.yaml \
  --apply \
  -o another_dataset_cleaned.xlsx
```

## Safety principles

- Source files are never edited in place.
- Plans are bound to source SHA-256 hashes.
- Every changed cell stores its old/new value.
- Deleted rows store their original content and position.
- Cleaned outputs are hashed for safe undo.
- Saved recipes contain only transformations that actually changed data.
- Recipe replay previews before writing and requires `--apply`.
- Suspicious scientific values are review items, not automatic corrections.

## Tests

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src tests
```

GitHub Actions runs on Python 3.11, 3.12, and 3.13.

## Next

The next major milestone is **v0.4: a local AI planner** that can propose the same typed `CleaningAction` objects from compact profiling evidence. AI will not receive unrestricted dataframe-write access; deterministic code remains the execution layer.

## License

No open-source license has been selected yet. Until we decide how the commercial/research product should be licensed, all rights are reserved by the repository owner.
