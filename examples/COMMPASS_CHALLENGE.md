# Public CoMMpass corruption challenge

This is a cleaner benchmark, not a myeloma research analysis.
Use the existing, locally downloaded public GDC workbook. No new data download,
patient identifiers, API credentials, or AI model are required.

The script writes a copy to a NEW directory and stores a local answer key. The
original five sheets remain, with four seeded defects on real data fields:
two nonempty null sentinels in previously blank cells, surrounding whitespace
in a category, and a capitalization variant in another category. A sixth sheet,
BenchmarkLabels, contains *only synthetic* specimen and assay names. Its pair
of synthetic assay labels ending in 101 versus 102 must remain separate;
the typo-shaped Peripheral Blod versus Peripheral Blood pair can be suggested
for semantic review, but is not automatically merged.

All output files live under gitignored pilot_outputs. The pristine workbook is
SHA-256 checked. The original is never edited. The script fails if expected
columns or suitable blank/observed cells are unavailable.

## Run it from the repo root on your Mac

1. Create a new challenge. The directory must NOT exist:

    bash sync.sh

    .venv/bin/python examples/commpass_challenge.py make \
      --source pilot_outputs/commpass_100/commpass_pilot.xlsx \
      --schema pilot_outputs/commpass_100/study_schema.yaml \
      --out pilot_outputs/commpass_challenge_01 \
      --seed 20260921

2. Run the cleaner WITHOUT giving it the answer key:

    .venv/bin/data-cleaner profile \
      pilot_outputs/commpass_challenge_01/commpass_challenge.xlsx \
      --schema pilot_outputs/commpass_challenge_01/study_schema.yaml \
      -o pilot_outputs/commpass_challenge_01/profile.json

    .venv/bin/data-cleaner plan \
      pilot_outputs/commpass_challenge_01/commpass_challenge.xlsx \
      --schema pilot_outputs/commpass_challenge_01/study_schema.yaml \
      -o pilot_outputs/commpass_challenge_01/plan.json

    .venv/bin/data-cleaner preview \
      pilot_outputs/commpass_challenge_01/commpass_challenge.xlsx \
      pilot_outputs/commpass_challenge_01/plan.json

3. Evaluate proposed actions without editing ANY workbook:

    .venv/bin/python examples/commpass_challenge.py verify \
      --challenge pilot_outputs/commpass_challenge_01 \
      --plan pilot_outputs/commpass_challenge_01/plan.json

   This checks all four exact expected injected cells, rejects unrelated
   deterministic actions, and verifies both synthetic fuzzy-label pairs
   surfaced for review. Failure writes benchmark_score.json and applies
   nothing. A fuzzy candidate is NOT authorization to merge labels. This
   command does not test whether a particular Ollama model accepts/defers.

4. OPTIONAL: explicitly exercise only answer-key-validated changes, then
   export the cleaned workbook and undo it:

    .venv/bin/python examples/commpass_challenge.py verify \
      --challenge pilot_outputs/commpass_challenge_01 \
      --plan pilot_outputs/commpass_challenge_01/plan.json \
      --exercise-apply-undo

   The fixture-only evaluator approves exact expected actions, NEVER
   unrecognized actions. It writes challenge_cleaned.xlsx,
   challenge_restored.xlsx, challenge_audit.json, benchmark_score.json;
   checks all sheets after export/reload/undo; and reconfirms both input
   hashes. Original and corrupted challenge source files remain unchanged.
   The exercise does not overwrite pre-existing outputs. To rerun, use
   a fresh directory (for example, _02).

For separate *interactive human review*, record decisions without applying:

    .venv/bin/data-cleaner review \
      pilot_outputs/commpass_challenge_01/commpass_challenge.xlsx \
      pilot_outputs/commpass_challenge_01/plan.json \
      -o pilot_outputs/commpass_challenge_01/human_review.json

The deterministic evaluator uses plan.json with a separate private
answer key, not the manually reviewed plan; these are separate tests.

## What passing establishes

- A fixed input + seed reproduces the same injected defects.
- The planner finds bounded repairs for all four seeded cell defects.
- A single whitespace cell may appear in both an independent trim preview
  and a category preview; sequential audit counts actual edits once.
- Deterministic rules do not merge synthetic assay labels with distinct
  numbers, but both label pairs are surfaced for optional semantic review.
- Explicit fixture-only apply and undo preserve every source sheet/value,
  and neither the original public workbook nor the corrupted test input
  is changed.

A pass does NOT mean the unmodified GDC workbook needs any cleaning or
that synthetic labels are genuine clinical observations.


## 5. Semantic accept/defer test using local Ollama (v0.7.1)

This is a SEPARATE test from the four-defect deterministic benchmark. It runs
only on the synthetic BenchmarkLabels candidates already created in step 1.
It does not rerun the downloader, modify the workbook, apply any model-approved
actions, or reveal the answer key to the model. The model receives no other
CoMMpass sheets, GDC case identifiers, source paths, or patient values.

First inspect the locally installed models:

    ollama list

Start Ollama locally if it is not running. Choose the EXACT model identifier
from the list, for example qwen3.5:4b when that model is installed. Do not
assume an example model is already present.

From the repository root:

    bash sync.sh

    .venv/bin/python examples/score_commpass_semantics.py \
      --challenge pilot_outputs/commpass_challenge_01 \
      --model qwen3.5:4b \
      --runs 3 \
      --output pilot_outputs/commpass_challenge_01/semantic_qwen35_4b.json

Change --model and --output together to compare two installed models. Each
output is new; the scorer refuses to overwrite an earlier experiment.
--think is an optional separate experiment; the default matches the standard
non-thinking structured planner. --url selects another Ollama endpoint if
you are not using the local default. Model calls are local but cannot run in
GitHub CI; the offline CI uses fake providers.

The score contains the expected and actual decision, candidate mapping,
model rationale, confidence, response errors, per-run latency, and pass
counts. Read the aggregate result without publishing source data:

    .venv/bin/python - <<'PY'
    import json
    from pathlib import Path

    p = Path("pilot_outputs/commpass_challenge_01/semantic_qwen35_4b.json")
    data = json.loads(p.read_text())
    print("Model:", data["model"])
    print("Passing runs:", data["passing_runs"], "/", data["runs"])
    print("Correct decisions:", data["correct_decisions"], "/",
          data["total_decisions"])
    print("Mean latency:", round(data["mean_latency_seconds"], 2), "seconds")
    for run in data["results"]:
        print("Run", run["run"], "passed:", run["passed"],
              "error:", run.get("error"))
        for choice in run["decisions"]:
            print(" ", choice["column"], choice["expected"],
                  "=>", choice["actual"], "correct:", choice["correct"])
    PY

The scorer requires BOTH decisions on EVERY run to count that run as passed.
An omitted decision does not receive credit for "defer", and unknown or
duplicate candidate IDs invalidate the run. The typo decision is
ACCEPT for Peripheral Blod -> Peripheral Blood; the numbered assay pair
requires DEFER. A model accepting the numbered assay merge fails even if
it correctly identifies the typo.

**Do not use AI-plan on the entire clinical challenge for this first
semantic test.** AI-plan would include the other tables' non-synthetic QA
contexts. The isolated script uses the same candidate construction and
model prompt while deliberately narrowing the material sent to Ollama.

A correct two-pair result is an engineering smoke test of semantic
judgment, not evidence of clinical semantic accuracy or authorization
to clean research records. More varied seeded cases and repeated runs
will be needed before broad claims.


## 6. Forty-case independent semantic stress test (v0.7.2)

This stand-alone synthetic-only experiment **does not read any CoMMpass workbook**.
Its forty fixed cases each create a separate, temporary one-column CSV. The
original two-pair challenge remains unchanged. The suite deliberately includes
twenty intended safe corrections and twenty distinct-label deferrals. It
scores **detector/candidate coverage separately from model judgments**: a
detector-negative example is NOT counted as a correct model deferral.

An actual baseline run of qwen3.5:4b on the initial forty-case corpus returned
32/40 candidates exposed, 22/32 correct decisions, zero unsafe merges and ten
safe corrections deferred. Its repeated justifications invented theoretical
clinical meanings for obvious misspellings. This is a preliminary engineering
benchmark on fixed synthetic cases, NOT an estimate of clinical accuracy.

The initial suite erroneously counted one whitespace case (s20) as a
candidate-coverage gap because an independent whitespace-trim proposal and a
category-mapping proposal coexisted. The revised scorer recognizes the
bounded trim/mapping overlap and scores the semantic mapping separately.
Compare **case-level results and corpus/prompt hashes**, not only topline
numbers, when using previously saved scores.

First audit production detector exposure without calling the model:

    bash sync.sh
    .venv/bin/python examples/semantic_stress_test.py \
      --audit-only \
      --output pilot_outputs/semantic_stress_v072_coverage.json

Then reproduce the current production-prompt baseline with the same model:

    .venv/bin/python examples/semantic_stress_test.py \
      --model qwen3.5:4b --runs 1 \
      --prompt-variant baseline \
      --output pilot_outputs/semantic_stress_v072_baseline.json

For a **separate experimental prompt** that requests evidence-proportional
decisions rather than speculative deferrals:

    .venv/bin/python examples/semantic_stress_test.py \
      --model qwen3.5:4b --runs 1 \
      --prompt-variant evidence_calibrated \
      --output pilot_outputs/semantic_stress_v072_calibrated.json

The evidence-calibrated addendum is benchmark-only: **the production AI planner
prompt has not changed**. It emphasizes accepting ordinary, clearly
typo-shaped words where there is no concrete contrary evidence, while
preserving numeric codes, stages, doses, biomarkers, and explicit study
dictionary distinctions. Compare unsafe merges, safe deferrals, and
context-contrasting zero-padding cases before changing the product prompt.

The JSON report records per-case expected/actual choice, exposure reason,
candidate mapping, rationale, confidence, latency, unknown IDs, protocol
failures, and grouped statistics. Existing scores are never overwritten.
Model runs may be slow because each exposed candidate gets its own local
Ollama request. No workbook transformations are executed.
