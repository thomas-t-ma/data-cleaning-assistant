# Frozen semantic architecture validation (v0.7.7 experiment)

This suite is a one-shot validation set for architectures already developed on the
earlier semantic stress and neighbor-holdout corpora. It must not become another
prompt-tuning set.

## Corpus

100 new synthetic cases, balanced 50 expected ACCEPT / 50 expected DEFER:

- 15 clinical/scientific single-edit typos
- 10 multi-edit typos
- 10 non-medical scientific typos
- 5 schema-authorized case/whitespace normalizations
- 10 schema-authorized zero-padding normalizations
- 20 valid scientific one-edit neighbors
- 10 meaningful/ambiguous abbreviations
- 10 numeric identifiers, doses, and timepoints
- 5 schema-defined distinct zero-padded codes
- 5 semantically distinct status/missingness labels

The labels and case IDs are disjoint from both prior 40-case corpora. Hidden scoring
metadata never enters a model prompt.

## Rule: freeze before looking

Do not edit prompts, labels, expected decisions, candidate generation, or auditor
instructions after viewing live results from this corpus.

The default live command evaluates all three frozen configurations in ONE process and
writes ONE JSON report only after all requested modes complete:

1. production baseline prompt
2. evidence-calibrated prompt
3. evidence-calibrated prompt + blind single-edit semantic veto

This is intentional. Do not run one mode, inspect it, change code, and then run another
mode on the same corpus.

## 1. Offline coverage audit

Run this first. It does not contact Ollama:

```bash
.venv/bin/python examples/frozen_validation_test.py \
  --audit-only \
  --output pilot_outputs/frozen_validation_coverage_v077.json
```

All 50 expected-safe corrections should reach candidate review. Some unsafe cases may
be screened out by deterministic candidate construction; that is recorded separately
and is safe end-to-end behavior, not credited as a model decision.

## 2. Frozen three-mode comparison

Recommended run:

```bash
.venv/bin/python examples/frozen_validation_test.py \
  --model qwen3.5:4b \
  --runs 3 \
  --output pilot_outputs/frozen_validation_all_3runs_v077.json
```

This is intentionally a large run. Use `--runs 2` only if compute time is a material
constraint, and record that choice before inspecting results.

## Metrics to interpret

Do not optimize for overall accuracy alone. Compare:

- safe_accepts / safe_total
- unsafe_accepts / unsafe_total
- safe_deferred
- safe_not_exposed
- unsafe_deferred
- unsafe_screened
- stable_cases_across_runs
- protocol_failures / model_errors
- model_calls and audit_calls
- per-group performance
- primary and audit latency

For this project, unsafe accepts have a higher cost than safe deferrals. A safe
correction that is deferred costs human review time; an unsafe accept can alter
scientific meaning.

## Decision discipline

After the first live report is saved:

- preserve it unchanged;
- do not change this validation corpus to improve the score;
- if a new architecture is designed from these results, create a NEW validation corpus
  before claiming generalization;
- synthetic results support engineering decisions but do not establish clinical safety.
