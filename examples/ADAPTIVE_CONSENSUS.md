# Adaptive calibrated-review consensus (v0.7.8 experiment)

This experiment was frozen only after the v0.7.7 100-case architecture validation
was saved. The v0.7.7 result suggested that repeated calibrated reviews may recover
the safety benefit of the veto without the blind auditor's recall loss.

## Frozen policy

For each bounded candidate:

1. Run the existing `evidence_calibrated` reviewer twice.
2. If both valid responses agree, stop and use that decision.
3. If they disagree, run the same reviewer once more.
4. Use 2-of-3 majority vote.
5. Any model/protocol error fails closed to DEFER / human review.

No new semantic prompt is introduced. The only change under test is repeated independent
review and a predeclared aggregation rule.

## Fresh corpus

100 new synthetic cases, balanced 50 expected ACCEPT / 50 expected DEFER. Label pairs
are disjoint from:
- the original 40-case semantic stress set;
- the 40-case single-edit neighbor holdout;
- the 100-case v0.7.7 frozen architecture validation.

The new set includes:
- 20 clinical/scientific single-edit typos;
- 10 multi-edit typos;
- 10 non-medical scientific typos;
- 5 schema-authorized formatting cases;
- 5 schema-authorized zero-padding cases;
- 25 valid near-neighbor terms;
- 10 meaningful status distinctions;
- 5 abbreviations;
- 5 numeric identifiers;
- 5 schema-defined distinct padded codes.

## 1. Coverage audit

Run this before any live model call:

```bash
.venv/bin/python examples/adaptive_consensus_validation.py \
  --audit-only \
  --output pilot_outputs/adaptive_consensus_coverage_v078.json
```

Expected: 100/100 candidate exposure.

## 2. One-shot live validation

```bash
.venv/bin/python examples/adaptive_consensus_validation.py \
  --model qwen3.5:4b \
  --output pilot_outputs/adaptive_consensus_frozen_v078.json
```

The report includes the first vote's single-pass performance and the final adaptive
consensus performance using the same underlying calls.

## Interpretation

Primary metrics:
- consensus safe accepts / 50;
- consensus unsafe accepts / 50;
- correct final decisions / 100;
- third calls / 100;
- total model calls;
- protocol/model failures;
- per-group performance.

The intended production tradeoff remains asymmetric: unsafe semantic merges are more
costly than safe deferrals. However, this is still synthetic engineering validation,
not evidence of clinical safety.

## Freeze discipline

After the first live report:
- preserve it unchanged;
- do not edit this corpus to improve the result;
- do not tune the calibrated prompt against this corpus;
- if consensus is changed based on these results, validate the revised rule on another
  untouched corpus before claiming generalization.
