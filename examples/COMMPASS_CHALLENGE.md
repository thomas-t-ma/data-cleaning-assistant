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
