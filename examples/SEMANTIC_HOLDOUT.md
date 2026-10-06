# v0.7.4 experiment — fresh single-edit scientific-neighbor holdout

This holdout exists because repeated prompt tuning on the original 40 semantic
fixtures can overfit that corpus. It specifically probes the highest-risk typo
heuristic boundary:

- 20 obvious misspellings of recognizable clinical/scientific terms (expected ACCEPT)
- 20 pairs of distinct valid scientific/clinical terms separated by one character
  (expected DEFER)
- every pair is intentionally constructed to cross the existing fuzzy detector and
  receive `single_edit_typo_pattern` evidence
- the expected decision and case ID never enter the model prompt
- no source dataset is loaded or edited

Examples of risky neighbors include Ileum/Ilium, Edema/Enema, Abduction/Adduction,
Lactate/Lactase, Afferent/Efferent, and Facial/Fascial. These are designed to catch
an unsafe policy that learns "one edit means typo."

## 1. Offline coverage audit

From the repository root:

```bash
.venv/bin/python examples/semantic_holdout_test.py \
  --audit-only \
  --output pilot_outputs/semantic_holdout_coverage_v074.json
```

Expected architectural result: 40/40 candidate exposure and every case should report
`single_edit_typo_pattern` as its evidence basis.

## 2. Freeze the current best prompt before testing

The original 40-case development comparison favored the benchmark-only
`evidence_calibrated` addendum over both the production baseline and the later
coherent replacement experiment. Do not revise the prompt based on this holdout
until the first holdout result is saved.

Run the baseline and calibrated prompts separately:

```bash
.venv/bin/python examples/semantic_holdout_test.py \
  --model qwen3.5:4b --runs 3 --prompt-variant baseline \
  --output pilot_outputs/semantic_holdout_baseline_3runs.json

.venv/bin/python examples/semantic_holdout_test.py \
  --model qwen3.5:4b --runs 3 --prompt-variant evidence_calibrated \
  --output pilot_outputs/semantic_holdout_calibrated_3runs.json
```

## Interpretation

Prioritize the two error types separately:

- Safe typo accepted / exposed
- Unsafe valid-neighbor accepted / exposed

A prompt that improves typo recall but starts merging valid one-edit terms is not an
acceptable tradeoff. Because this is a held-out synthetic set, preserve the first
results before making any prompt changes.

Do not promote any benchmark prompt to production solely because it performs well
on these synthetic cases. The AI planner remains bounded by deterministic candidate
construction and human review.
