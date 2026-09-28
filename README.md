# Data Cleaning Assistant

A local-first, research-aware data quality and cleaning assistant for CSV, Excel, and Parquet datasets.

Current version: **v0.8.0**.

The core rule is **detection is not transformation**. Profiling is read-only. Deterministic rules and optional local AI can propose changes, but only deterministic Python previews and executes them after explicit human approval.

## Current capabilities

The profiler detects hidden missing-value tokens, exact duplicate rows, constant columns, category inconsistencies, surrounding whitespace, mixed numeric values, malformed dates, likely identifiers, unnamed columns, conservative statistical outliers, candidate keys, repeated entities, dataset grain, and cross-table relationships.

Reviewable transformations include category normalization, missing-token normalization, date standardization, numeric coercion for non-identifiers, whitespace trimming, and reversible exact-duplicate removal.

## v0.4 local AI planner

v0.4 adds an optional local Ollama planning layer.

The AI **does not receive the dataframe and does not write data**. Starting in v0.4.3, it also does not author cleaning actions. Deterministic code converts profiler findings into a bounded list of candidate actions; the model only returns a short candidate ID plus `accept` or `defer`, rationale, and confidence.

Ollama structured output must still satisfy the Pydantic JSON schema, but table names, column names, transformation parameters, mappings, tokens, and formats all come from local deterministic code rather than model-generated fields.

Examples of enforced restrictions:

- category-merge candidates can only use observed detector-supported label pairs
- missing-value candidates contain only structural null sentinels; contextual labels such as `Unknown`, `Not available`, and `Not applicable` remain QA findings but are not executable null-normalization candidates without an explicit study schema
- identifier-like numeric coercion is never exposed as an AI candidate
- row deletion is never exposed as an AI candidate
- outlier deletion and imputation are never exposed as AI candidates
- fuzzy category candidates test semantic judgment: the model should accept clear typos but defer scientifically distinct labels
- zero-padding variants such as `Visit 01` vs `Visit 1` receive deterministic numeric-equivalence evidence rather than relying on string similarity alone
- likely typos such as `Peripheral Blod` vs `Peripheral Blood` receive conservative single-edit evidence only when exactly one sufficiently long non-numeric token differs by one character and all surrounding tokens match; numeric/short-token differences remain semantic review cases

The default `ai-plan` mode is **hybrid**: deterministic proposals remain in the plan and validated AI suggestions are added when they contribute something new. If deterministic profiling produces no AI-review candidates, the Ollama call is skipped entirely.

## v0.8.0 — adaptive calibrated consensus

v0.8.0 promotes the evidence-calibrated semantic-review policy into the production
AI planner and adds a conservative repeated-review aggregation rule.

For every bounded candidate, the local model is reviewed twice with the same
evidence-calibrated prompt. If both valid decisions agree, that result is used. If
they disagree, one third review is requested and a 2-of-3 majority decides. Any
structured-output/model failure or malformed candidate selection fails closed to a
non-executable human-review deferral.

The policy was selected after a 100-case synthetic architecture comparison and then
tested on a **new frozen 100-case validation corpus**. On that fresh set, all 50
expected-safe corrections reached model review, seven expected-unsafe cases were
screened out deterministically, and adaptive consensus produced **47/50 safe
acceptances with 0/50 unsafe acceptances and zero protocol failures**. These are
synthetic engineering results, not clinical-safety validation.

The production planner still preserves all previous boundaries: deterministic code
owns the candidate action, AI decisions remain proposed rather than approved, and
preview/human approval/source hashing/audit logging remain mandatory.

## v0.7.3 — evidence-calibrated benchmark experiment

The first live 40-case stress run exposed 32 candidates, with 22 correct
model decisions, **zero unsafe merges**, and ten safe typo corrections
deferred. The model frequently speculated about nonexistent distinct
scientific meanings for ordinary misspellings.

v0.7.3 fixes one erroneous detector-coverage gap for a whitespace case
that had both a valid trim and a valid category mapping. It also introduces
`--prompt-variant evidence_calibrated` for a **benchmark-only** paired
comparison against the unchanged production prompt (`baseline`).
It emphasizes concrete evidence over hypothetical objections without
weakening the protection of distinct codes, markers, doses, stages, and
study dictionary distinctions. No production prompt change or data
transformation occurs in the comparison. See the
[stress-test runbook](examples/SEMANTIC_STRESS.md).

## v0.7.2 independent synthetic semantic stress test

The two-pair Ollama smoke test now has a harder companion: **40
independent synthetic cases**, half expected accept and half defer.
The corpus includes typos, formatting variations, numeric-code traps,
medical-label examples, and matched cases in which explicit fictional
study-dictionary context changes the correct decision. Category
prevalences also vary.

Each case uses real profiling and bounded candidate generation, and
gets its **own single-candidate model call**. Detector misses are
counted separately, not credited as correct model deferrals. The
score separates unsafe merges, missed safe corrections, protocol
failures, source-candidate coverage, explanations, and latency. No
public workbook or GDC records are loaded or modified.

Run a no-model exposure audit first, then try the installed local model:

```bash
bash sync.sh
.venv/bin/python examples/semantic_stress_test.py --audit-only \
  --output pilot_outputs/semantic_stress_coverage_v072.json

.venv/bin/python examples/semantic_stress_test.py --model qwen3.5:4b --runs 1 \
  --output pilot_outputs/semantic_stress_qwen35_4b_v072.json
```

The model name is an **example**; use the exact name from
\`ollama list\`. See the [complete stress-test runbook](examples/SEMANTIC_STRESS.md)
for the summary script, failure categories, caveats, and optional
repeated experiments. The original CoMMpass challenge is unchanged.

## v0.7.1 synthetic-label semantic judgment benchmark

The CoMMpass real-data challenge now has a **separate local Ollama
evaluation**. It uses the same bounded deterministic fuzzy candidates
and AI-planner prompt but sends **only the synthetic BenchmarkLabels sheet**
to the model; no public patient records, file paths, answer key, or the
other CoMMpass tables enter model context. The expected decisions remain
outside the prompt:

- Accept the typo-shaped `Peripheral Blod` → `Peripheral Blood`.
- Defer the genuinely distinct `Synthetic assay control 101` versus
  `Synthetic assay control 102` pair.

Each model run records literal decisions, rationales, confidence, protocol
errors, and latency. Missing/duplicate/unknown candidate IDs count as
failures; no workbook modifications are made. The test is intentionally
two decisions and does not measure performance on clinical research
categories generally.

See [the CoMMpass challenge runbook](examples/COMMPASS_CHALLENGE.md) for
the exact `--model`, `--runs`, and `--output` commands.

## v0.7 known-answer challenge on public CoMMpass data

After the unmodified 100-case public pilot correctly produced **zero**
deterministic cleaning actions, the next test is whether the cleaner finds
genuine seeded defects **without** introducing unjustified edits. A local
script creates a separate, reproducible corrupt copy of the public workbook,
records an answer key, injects four bounded cell defects in the existing
sheets, and adds a clearly synthetic sheet for safe-versus-ambiguous fuzzy
category review. A read-only scoring command checks the resulting plan;
an explicit opt-in command exercises fixture-only apply, workbook save/reload,
and undo against the answer key. Original and corrupt-input hashes are checked.

See [the complete Mac runbook](examples/COMMPASS_CHALLENGE.md). Start here
once the original pilot workbook already exists:

```bash
bash sync.sh
.venv/bin/python examples/commpass_challenge.py make \
  --source pilot_outputs/commpass_100/commpass_pilot.xlsx \
  --schema pilot_outputs/commpass_100/study_schema.yaml \
  --out pilot_outputs/commpass_challenge_01
```

Do not claim this validates a particular AI model's decisions: the harness
only verifies that the synthetic fuzzy pairs are exposed for semantic review,
not that a local model accepted or deferred the right merges.

## Research-aware QA and review (v0.6)

The first public-data pilot exposed repetitive complete-coverage missing/constant
findings, misleading inferred joins between repeated-event tables, pooled numeric
outliers across heterogeneous assays, and overlapping change previews. The
following improvements are **general-purpose**, not hard-coded to CoMMpass:

- `profile` groups complete-coverage missing-token and constant-column
  *findings in the terminal only*. `--all-issues` restores the expanded view;
  `--output` always saves **all** original issues, including contextual
  missingness. Do not interpret "complete coverage" as permission to discard a
  clinically meaningful `Unknown` category.
- If a schema declares foreign keys, the relationship table shows **only
  declared parent-child relationships**, with source `declared`. Inferred
  relationships remain available without a declared key map. A common
  `case_id` between two event tables does not constitute a safe row-by-row join.
- Declared table grain is displayed with the key columns and never inferred
  from duplicated subject IDs when a study schema supplies the grain.
- Optional `numeric_groups` scopes **outlier QA only** to separate assay/unit
  strata. The group columns must exist and groups with too few numeric values
  produce no outlier findings. No values are standardized, deleted, or converted
  across units; blank grouping labels are excluded rather than silently pooled.

For example, once you have *verified the actual dataset column names*, extend
the relevant table in its study schema:

```yaml
tables:
  Measurements:
    grain_key: [measurement_id]
    numeric_groups:
      value: [assay, unit]
```

Do not use subject IDs as assay groups or imply that this schema performs unit
conversion. A candidate column must be explicitly listed; the cleaner does
not infer whether two assays are biologically comparable.

`preview` and `review` now display an **independent-preview overlap** warning
when two proposed edits touch the same original cells. The notice is advisory:
previews always use the original source while approved actions execute in
sequence. The audit remains the authority for the **actual** cell-change count.
Row-deletion interactions are not yet included in overlap analysis.

Re-run the existing public-data pilot from the repo root:

```bash
bash sync.sh
.venv/bin/data-cleaner profile pilot_outputs/commpass_100/commpass_pilot.xlsx \\
  --schema pilot_outputs/commpass_100/study_schema.yaml \\
  -o pilot_outputs/commpass_100/profile_v06.json

# Optional expanded issue view, without changing the dataset:
.venv/bin/data-cleaner profile pilot_outputs/commpass_100/commpass_pilot.xlsx \\
  --schema pilot_outputs/commpass_100/study_schema.yaml --all-issues
```

Retain the original v0.5.2 report to compare finding totals. Grouping is a
display-only change: it must not reduce the raw finding count or introduce
new cleaning actions. Add assay/unit groups to the schema only after verifying
those fields and their meanings.

### v0.6.3 — empty-cell semantics

The CoMMpass v0.6.2 pilot produced ten remaining actions, all
`normalize_missing` with token `''`. With `na_filter=False`, pandas can
represent ordinary blank Excel cells as empty strings. A proposed
`'' → None` change may therefore be a reader-representation artifact,
not a meaningful workbook cleanup. By default, **exact empty text is
already empty** for profiling and planning:

- Blank CSV/Excel cells count toward each column's `null_count` and
  `null_percent`, and are omitted from sample values.
- Exact empty strings do not produce `hidden_missing_values` or
  `constant_column` findings. The CLI prints a compact
  `Already-empty fields` count, so column coverage is not lost.
- No routine empty-string-to-null action is generated from these cells.
  Nonempty sentinels such as `NA`, `N/A`, and whitespace-only strings
  remain separately reviewable; meaningful contextual labels such as
  `Unknown` remain preserved.
- A researcher's explicit `missing_tokens: [""]` study rule remains
  permitted and still requires approval. This is not blanket permission
  to convert all missingness labels.

The loader does not modify the source. A profile from this release may
contain *fewer* issues and planning notes than v0.6.2: the previously
spurious empty-token findings are now represented as column coverage,
not silently filtered only in the terminal. Do not compare raw issue
counts across the versions as though the same detectors were unchanged.

To rerun the **same** local pilot without downloading or applying data:

```bash
bash sync.sh
.venv/bin/data-cleaner profile \
  pilot_outputs/commpass_100/commpass_pilot.xlsx \
  --schema pilot_outputs/commpass_100/study_schema.yaml \
  -o pilot_outputs/commpass_100/profile_v063.json
.venv/bin/data-cleaner plan \
  pilot_outputs/commpass_100/commpass_pilot.xlsx \
  --schema pilot_outputs/commpass_100/study_schema.yaml \
  -o pilot_outputs/commpass_100/plan_v063.json
```

### v0.6.2 — actionable missingness triage

The public CoMMpass pilot exposed 104 deterministic cleaning actions, all
`normalize_missing`, mostly for columns without a single meaningful value.
A complete-coverage field is an important QA finding but usually not 100+ cells
of useful researcher-by-researcher cleaning decisions.

**Routine** missing-token actions are now omitted when every entry in a column
is an actual null or a structural missing-value sentinel. These fields remain
in the full `profile` report, and each omission is persisted in the cleaning
plan's non-executable `planning_notes` with the table, column, source issue
ID and `structural_only_column` reason. `plan`, `preview`, `review`, and
`ai-plan` show a compact per-table summary. This does **not** silently remove
columns, convert values, or authorize any automated edit.

Partial missingness still produces normal reviewable proposals. Contextual
labels such as `Unknown` are not treated as structural nulls. **An explicit
study-schema `missing_tokens` declaration overrides the routine omission**
and still creates an action requiring individual researcher approval. AI-only
planning uses the same routine triage, so a local model is not asked to approve
hundreds of coverage-only rewrites.

Re-run the existing public-data pilot without changing the source workbook:

```bash
bash sync.sh
.venv/bin/data-cleaner plan \
  pilot_outputs/commpass_100/commpass_pilot.xlsx \
  --schema pilot_outputs/commpass_100/study_schema.yaml \
  -o pilot_outputs/commpass_100/plan_v062.json
```

Check the proposed-action count and per-table planning-note counts against the
104-action v0.6.1 plan. Do not reuse old action approvals: review a fresh plan
against the current source hash. No GDC records are downloaded by this command.

### v0.6.1 — mixed-measurement outlier scope

When a generic result column (for example, `test_value`) is accompanied by
**multiple observed assay/type or unit labels**, a pooled IQR test is not
interpretable. The profiler replaces any pooled `statistical_outlier` issue
with one read-only `unscoped_numeric_measurement` warning, even if a pooled
test happens to find no outliers. The warning reports metadata column names,
distinct-label counts, and whether pooled flags were suppressed. It never
labels individual measurements erroneous.

This is a conservative name-based safety guard, **not** proof that two assays
are incomparable and not automatic assay matching. It does not apply to
specific independent numeric columns such as `hemoglobin`; those still need
the researcher's judgment. If the study schema explicitly declares
`numeric_groups`, the cleaner runs separate QA only within those declared
assay/unit groups instead. No unit conversion or editing is performed.

```bash
bash sync.sh
.venv/bin/data-cleaner profile \
  pilot_outputs/commpass_100/commpass_pilot.xlsx \
  --schema pilot_outputs/commpass_100/study_schema.yaml \
  -o pilot_outputs/commpass_100/profile_v061.json
```

Compare the `LabTests.test_value` issue with v0.6.0. The source workbook is
unchanged, and the new scope warning remains in the full JSON. Do not add
`numeric_groups` until the actual column names and meanings have been checked.

## NCI GDC CoMMpass public clinical-data pilot (v0.5.2)

The first non-synthetic research pilot uses the **open clinical metadata** from
NCI GDC project `MMRF-COMMPASS`. It does not download controlled-access genomic
data. This script must be run locally with access to `api.gdc.cancer.gov`;
GitHub CI uses offline fixtures and does not download research data.

```bash
bash sync.sh
.venv/bin/python examples/fetch_gdc_commpass.py \
  --limit 100 --output pilot_outputs/commpass_100

.venv/bin/data-cleaner profile pilot_outputs/commpass_100/commpass_pilot.xlsx \
  --schema pilot_outputs/commpass_100/study_schema.yaml \
  -o pilot_outputs/commpass_100/profile.json
```

The downloader keeps the unmodified response in `source_cases.json`, records
the query URLs and SHA-256 in `source_manifest.json`, and generates Cases,
Diagnoses, Treatments, FollowUps, and LabTests sheets with explicit parent
keys. No clinical categories or null semantics are guessed in the generated
schema. Check row counts and the presence of needed variables before forming
a prognostic model. Use `--limit 0` to request the full GDC project **only
after** validating the 100-case pilot. An already-populated output folder
is not overwritten.

See `examples/PILOT.md` for the earlier synthetic end-to-end pilot.

## Synthetic end-to-end pilot (v0.5.1)

Start with [the synthetic pilot runbook](examples/PILOT.md). It covers the full CLI workflow, local Ollama test, explicit review, source-preserving apply, audit/recipe checks, and undo before trying de-identified research data. CI runs the deterministic pilot on Python 3.11, 3.12, and 3.13.

## Study schemas and research QA (v0.5)

Study dictionaries are **opt-in**. The default no-schema workflow stays unchanged. A local
YAML/JSON file may define exact workbook sheet names, grain keys, allowed categories, explicit
canonical mappings, study-specific null tokens, and foreign-key relationships. A subject ID,
visit order, and date enable a read-only chronological check.

Try the bundled **synthetic** workbook and schema:

```bash
.venv/bin/python examples/make_messy_example.py
.venv/bin/data-cleaner profile examples/messy_research_example.xlsx \
  --schema examples/study_schema_example.yaml -o schema_report.json
.venv/bin/data-cleaner plan examples/messy_research_example.xlsx \
  --schema examples/study_schema_example.yaml -o schema_plan.json
.venv/bin/data-cleaner ai-plan examples/messy_research_example.xlsx \
  --schema examples/study_schema_example.yaml --model qwen3.5:4b \
  -o ai_cleaning_plan.json
```

The schema uses exact source values and target values. For example:

```yaml
version: 1
tables:
  Visits:
    grain_key: [PatientID, VisitDate]
    subject_id: PatientID
    columns:
      SpecimenSource:
        allowed_values: [Peripheral Blood, Bone Marrow]
        canonical_map:
          Peripheral Blod: Peripheral Blood
      MissingReason:
        missing_tokens: [Unknown] # Only when this study truly defines it as null
```

**Use the actual sheet and column names for your dataset.** An unknown schema reference
fails validation instead of silently guessing an alias. Canonical targets must be declared
allowed values when `allowed_values` is present; identifier-edit mappings are prohibited.
The schema file's SHA-256 is recorded in the plan and audit log (not the raw schema content).
`unit` is available as descriptive schema metadata but unit conversion is not implemented.

The read-only schema checks currently include declared grain-key missingness/duplicates,
unrecognized categorical values, child-to-parent foreign-key gaps, non-unique parent keys,
and visit dates that reverse declared visit order for the same subject. None of these checks
authorizes record deletion, imputes a value, or reorders visits.

Study-defined canonical mappings and null tokens create **proposed actions only**; a researcher
still needs to inspect and approve them. In schema-managed category columns, the study
dictionary takes precedence over heuristic fuzzy category merges. Undeclared `Unknown` and
`Not applicable` remain untouched.

### Deferred AI reviews

Model deferrals are saved in `CleaningPlan.deferred_reviews` with the bounded proposed
action, exact candidate ID, evidence, question, rationale, and status. They are
**not** in `CleaningPlan.actions`; `apply --approve-all` cannot execute them.
`data-cleaner review INPUT PLAN -o REVIEWED.json` displays each deferred item and
allows explicit approval, rejection, or leaving it pending. Approving one converts that
bounded action to an approved, reviewable plan action; no model-generated parameters
can enter the plan.

## AI privacy / transparency

Run:

```bash
.venv/bin/data-cleaner ai-context your_dataset.xlsx -o ai_context.json
```

to inspect **exactly what would be sent to the local model**.

The default AI context excludes:

- source file path
- raw row samples
- identifier values
- numeric summaries for identifier columns

Detector evidence may still include category labels, missing-value tokens, malformed date examples, or other values needed to evaluate a supported cleaning action. Generated AI context/plan/audit artifacts are ignored by Git by default.

## Local AI setup

Ollama is intentionally not installed or started by this project. Install it separately, then pull whichever local model you want to use.

The current recommended local baseline is `qwen3.5:4b`. On the fixed core suite it completed 24/24 case-runs with 100% required-action recall, 100% accepted-action precision, zero unsafe accepts, and zero validator rejections.

```bash
ollama pull qwen3.5:4b
```

Check the connection and installed models:

```bash
.venv/bin/data-cleaner ai-status
```

If exactly one model is installed, `ai-plan` can select it automatically. If multiple are installed, choose one explicitly with `--model`.

## AI benchmark

v0.4.1 includes a fixed synthetic benchmark with hidden answer keys. The model sees only the same privacy-minimized profiler context used by normal AI planning; it never sees the benchmark description or expected actions.

The original regression suite is now called `core`. Structured planning uses Ollama with thinking disabled by default for lower latency:

```bash
.venv/bin/data-cleaner ai-benchmark \
  --suite core \
  --model qwen3.5:4b \
  --output ai_benchmark_core.json
```

v0.4.5 adds a harder `research` suite with visit/cohort formatting, specimen labels, censored lab values, dose and unit distinctions, timepoints, treatment arms, and meaningful values such as `Unknown` or `Not applicable`.

v0.4.6 hardens the semantics exposed by that suite: contextual missingness values remain visible QA findings but are removed from the executable candidate surface by default, while numeric zero-padding variants get explicit machine-derived equivalence evidence.

v0.4.7 adds generic single-edit typo-shape evidence for long non-numeric tokens while deliberately excluding short or numeric differences such as `Stage II/III`, `Dose 5/50`, or `Arm A/B`.

```bash
.venv/bin/data-cleaner ai-benchmark \
  --suite research \
  --model qwen3.5:4b \
  --output ai_benchmark_research.json
```

Use `--suite all` to run both suites together.

Use `--think` if you explicitly want to compare the model's thinking mode.

For a more stable estimate, repeat every case three times:

```bash
.venv/bin/data-cleaner ai-benchmark \
  --suite research \
  --model qwen3.5:4b \
  --runs 3 \
  --output ai_benchmark_research_qwen35_4b.json
```

Failed cases print the model summary, accepted actions, and any candidate-ID validation problems directly in the terminal. Under v0.4.3, normal failures should mostly reflect semantic accept/defer judgment rather than malformed action fields.

The benchmark reports:

- **required-action recall** — how often the model proposes the cleaning action the case requires
- **accepted-action precision** — how many validator-surviving actions are actually correct
- **forbidden accepted** — semantically unsafe model decisions that survived bounded candidate review
- **unsafe candidates exposed** — actions that should have been blocked by architecture before the model ever saw them
- **validator catch rate** — forbidden model proposals blocked by the validator
- **case-run pass rate** — cases with all required actions found and no unexpected accepted actions
- **latency** — mean local-model time per synthetic case

The safety cases deliberately distinguish two failure modes. Some capabilities, such as row deletion, identifier coercion, or collapsing contextual missingness labels, should not be exposed to the model at all. Others, such as merging lexically similar but scientifically distinct categories like `Stage II` and `Stage III`, are deliberately exposed because they test the model's semantic judgment.

Benchmark JSON is ignored by Git by default so local model-evaluation results are not committed accidentally.

## AI-assisted workflow

Inspect the exact model context:

```bash
.venv/bin/data-cleaner ai-context \
  examples/messy_research_example.xlsx \
  -o ai_context.json
```

Generate a hybrid deterministic + local-AI plan:

```bash
.venv/bin/data-cleaner ai-plan \
  examples/messy_research_example.xlsx \
  --model qwen3.5:4b \
  --context-out ai_context.json \
  -o ai_cleaning_plan.json
```

AI suggestions that fail deterministic safety validation are displayed as rejected and are not placed in the plan.

Review the resulting plan exactly like a normal plan:

```bash
.venv/bin/data-cleaner review \
  examples/messy_research_example.xlsx \
  ai_cleaning_plan.json \
  -o reviewed_plan.json
```

Then apply only approved actions:

```bash
.venv/bin/data-cleaner apply \
  examples/messy_research_example.xlsx \
  reviewed_plan.json \
  -o cleaned.xlsx
```

The AI layer never bypasses preview, approval, source hashing, audit logging, recipe generation, Excel safety rules, or undo.

## Deterministic-only workflow

The original local workflow remains available with no AI runtime:

```bash
.venv/bin/data-cleaner plan examples/messy_research_example.xlsx -o cleaning_plan.json
.venv/bin/data-cleaner review examples/messy_research_example.xlsx cleaning_plan.json -o reviewed_plan.json
.venv/bin/data-cleaner apply examples/messy_research_example.xlsx reviewed_plan.json -o cleaned.xlsx
```

## Recipe replay

Saved recipes contain only transformations that actually changed data. Preview a recipe against another dataset:

```bash
.venv/bin/data-cleaner replay another_dataset.xlsx cleaned.xlsx.recipe.yaml
```

Schema drift is shown before replay. Execution still requires `--apply`.

## One-command project sync

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

## Safety principles

- Source files are never edited in place.
- Plans are bound to source SHA-256 hashes.
- Every changed cell stores its old/new value.
- Deleted rows store their original content and position.
- AI proposals are untrusted until deterministic validation succeeds.
- AI actions begin as proposed, never approved.
- Suspicious scientific values are review items, not automatic corrections.

## Tests

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src tests
```

GitHub Actions runs on Python 3.11, 3.12, and 3.13. AI tests use fake providers; CI does not require Ollama or internet access.

## License

No open-source license has been selected yet. Until we decide how the commercial/research product should be licensed, all rights are reserved by the repository owner.
