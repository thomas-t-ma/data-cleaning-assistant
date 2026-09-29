# BES real-workflow pilot for v0.9

This pilot exercises the v0.9 production candidate on a second, structurally
different public research dataset.

## Why this dataset

The upstream British Ecological Society Data and Code Hackathon project publishes:

- an anonymized raw Google Form research dataset;
- the authors' cleaned analysis dataset;
- the cleaning code and paper/repository provenance.

The upstream README explicitly describes the raw form output as very messy and
states that their cleaning work standardizes write-in answers, fixes typos,
removes duplicates, and resolves flagged paper-level issues. The cleaned file is
therefore useful as a qualitative reference, but it is **not** treated as a
cell-by-cell answer key for this cleaner.

The fetcher pins the upstream repository commit so the source cannot silently
change between runs.

## 1. Fetch the public sources

```bash
.venv/bin/python examples/fetch_bes_hackathon.py \
  --out pilot_outputs/bes_public
```

Use the raw anonymized export as the cleaner input:

```bash
SOURCE="pilot_outputs/bes_public/BES-data-code-hackathon-raw-outputs_ANON_2025-11-16.csv"
OUT="pilot_outputs/bes_v090_01"
mkdir -p "$OUT"
```

Do not edit the source and do not create a study schema for the first pass.

## 2. Profile with the v0.9 candidate

```bash
.venv/bin/data-cleaner profile "$SOURCE" \
  -o "$OUT/profile.json"
```

## 3. Generate the production AI-assisted plan

```bash
.venv/bin/data-cleaner ai-plan "$SOURCE" \
  --model qwen3.5:4b \
  --context-out "$OUT/ai_context.json" \
  -o "$OUT/ai_plan.json"
```

Normal v0.9 routing sends only bounded semantic category-equivalence candidates
to Ollama in focused batches of at most four candidates. Mechanical cleanup stays
deterministic.

Summarize the untouched plan:

```bash
.venv/bin/python examples/summarize_v090_workflow.py \
  --plan "$OUT/ai_plan.json" \
  --context "$OUT/ai_context.json"
```

Stop here initially and inspect the output before making review decisions. In
particular, record:

- semantic AI candidate count;
- deterministic vs AI-origin action counts;
- deferred semantic reviews;
- action-type mix.

## 4. Grouped human review

After inspecting the initial plan:

```bash
.venv/bin/data-cleaner review-grouped "$SOURCE" "$OUT/ai_plan.json" \
  -o "$OUT/reviewed_plan.json"
```

Use family approval only when the exact transformation is appropriate across the
displayed targets. Multi-action approval requires typing `YES`. Drill down when a
family is heterogeneous or scientifically unclear.

This command writes review decisions only. It does not modify the source dataset.

Summarize the reviewed plan:

```bash
.venv/bin/python examples/summarize_v090_workflow.py \
  --plan "$OUT/reviewed_plan.json" \
  --context "$OUT/ai_context.json" \
  --output "$OUT/review_summary.json"
```

The reviewed plan records review duration and family/individual decision
provenance automatically.

## 5. Compare qualitatively with the upstream cleaned reference

The upstream cleaned dataset is:

```bash
REFERENCE="pilot_outputs/bes_public/BES-data-code-hackathon-cleaned_2025-12-01.csv"
```

Do **not** treat differences as automatic cleaner errors. The authors' cleaning
script performs study-specific recoding, variable selection, flagged-record
resolution, and other transformations outside the current product's claimed
scope.

Use the reference to answer narrower questions:

- Did the cleaner surface obvious spelling/case/whitespace variants that the
  authors later standardized?
- Did semantic-only routing avoid wasting model calls on mechanical cleanup?
- Were model-accepted category merges defensible?
- Were model deferrals genuinely ambiguous?
- Did grouped review make the deterministic workload manageable?
- Are there obvious issues the current detectors should reasonably have surfaced
  but missed?

Do not apply the reviewed plan during this validation unless a separate explicit
apply/undo test is later designed.

## Interpretation

This is a product-development pilot on public research data. It does not
establish detector recall, scientific correctness, clinical safety, or
generalization to other studies.
