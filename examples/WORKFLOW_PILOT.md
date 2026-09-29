# Real-workflow pilot (v0.9 development)

This pilot measures whether the cleaner is actually useful on a messy research
workbook, not just whether it performs well on synthetic semantic benchmarks.

The workflow is intentionally read-only until the researcher separately decides
to apply an approved plan. The pilot itself never edits the source file.

## What to use

Choose one **realistic research workbook** that:

- you already understand well enough to review its proposed changes;
- contains genuine messiness rather than deliberately seeded benchmark defects;
- is a working copy, export, de-identified dataset, or otherwise appropriate for
  local analysis under your organization's data-handling rules;
- is not currently being modified by another process during the pilot.

A study schema is strongly preferred when you have one. Do not invent a schema
just to improve the score.

The model is local Ollama. The AI context excludes source paths, raw row samples,
and identifier values. In this v0.9 workflow experiment, **AI routing is semantic-only**:
the model reviews bounded category-equivalence/fuzzy-label candidates, while
mechanical missing-token, whitespace, date, numeric, and duplicate proposals
remain deterministic human-review items and are never re-decided by the LLM.

This routing is experimental; v0.8 production behavior is unchanged. The goal is
to test whether removing redundant LLM review reduces burden without weakening
the deterministic safety/human-approval boundary.

Pilot artifacts are local and may still contain sensitive research terminology.
Do not publish or upload them casually.

## Public starter dataset: NHANES sample

For the first public workflow pilot, use OpenIntro's `nhanes.samp`: a direct
CSV containing 200 observations and 76 variables derived from the NHANES teaching
dataset. It is based on real US National Health and Nutrition Examination Survey
data, but adapted for educational use rather than intended as an analysis-ready
research database.

Fetch and validate it:

```bash
.venv/bin/python examples/fetch_nhanes_sample.py \
  --out pilot_outputs/nhanes_public
```

The fetcher verifies the published 200 x 76 shape and several expected fields,
then records source provenance and SHA-256. It does not call a model.

Then use:

```bash
SOURCE="pilot_outputs/nhanes_public/nhanes_openintro_sample.csv"
OUT="pilot_outputs/workflow_nhanes_01"
```

For the first pass, do not invent a study schema. We want to observe the generic
cleaner's behavior on real health-survey structure before adding any source-specific
dictionary knowledge.

## 1. Start a fresh pilot

From the repository root:

```bash
.venv/bin/python examples/workflow_pilot.py start \
  --source /path/to/research_workbook.xlsx \
  --schema /path/to/study_schema.yaml \
  --model qwen3.5:4b \
  --out pilot_outputs/workflow_real_01
```

Omit `--schema` only if no verified study dictionary is available.

The harness writes:

- `profile.json` — complete read-only QA profile;
- `ai_context.json` — exact compact context sent to the local model;
- `ai_plan.json` — hybrid deterministic + AI plan;
- `decision_trace.json` — candidate decision/rationale/coverage telemetry;
- `pilot_manifest.json` — source hash and local provenance.

It applies **zero edits**.

## 2. Triage before reviewing

Before opening dozens of actions one-by-one, summarize the pilot:

```bash
.venv/bin/python examples/workflow_pilot.py summarize \
  --pilot pilot_outputs/workflow_nhanes_01
```

This prints grouped counts for issue types, action types, action origins,
candidate types, evidence bases, decisions, and the highest-volume candidate
targets. It is intended to identify detector/review floods before the human
reviewer spends time on them.

## 3. Preview the plan

```bash
.venv/bin/data-cleaner preview \
  /path/to/research_workbook.xlsx \
  pilot_outputs/workflow_real_01/ai_plan.json \
  --sample 8
```

Look for obviously duplicated proposals, scientifically questionable changes,
large affected-cell counts, or proposals whose meaning cannot be judged from the
available study context.

## 4. Review by transformation family

Do **not** use `--approve-all`. Instead use the pilot's grouped reviewer:

```bash
.venv/bin/python examples/workflow_pilot.py review-grouped \
  --pilot pilot_outputs/workflow_real_01 \
  --output pilot_outputs/workflow_real_01/reviewed_plan.json
```

Actions with the same origin, type, and exact parameters are grouped into a
family. For each family the reviewer shows:

- action count;
- total affected items from independent previews;
- the highest-impact table/column targets;
- the exact transformation parameters.

You may:

- **approve all** — requires a second explicit `YES` confirmation for a
  multi-action family;
- **reject all**;
- **inspect individually** — review each target separately with compact previews;
- **skip/defer** the family;
- **quit** and save partial progress.

Grouped approval changes only the reviewed-plan JSON. It does **not** apply edits.

Semantic AI deferrals, when present, are still reviewed individually because
their meaning is candidate-specific rather than a mechanical transformation
family.

Approve only when you would genuinely want the exact transformation across the
displayed family. Reject when the transformation is wrong, misleading, or
unnecessary. Defer when you genuinely cannot decide.

Human review is workflow feedback, **not scientific ground truth**.

## 5. Record obvious misses

After the review, inspect the workbook/profile with your normal domain knowledge.
Write down any important issue the tool completely failed to surface. Examples:

- a meaningful category inconsistency it did not detect;
- a date/numeric problem it missed;
- a relationship/grain problem that would matter in analysis;
- a misleading proposal class you would want the product to suppress.

Do not count features the project does not currently claim to detect merely to
inflate the miss count.

## 6. Score the workflow

If review took about 12 minutes and you noticed two missed issues:

```bash
.venv/bin/python examples/workflow_pilot.py score \
  --pilot pilot_outputs/workflow_real_01 \
  --reviewed-plan pilot_outputs/workflow_real_01/reviewed_plan.json \
  --review-minutes 12 \
  --missed-issue "Visits.Status: two semantically equivalent labels were not surfaced" \
  --missed-issue "Labs.ResultDate: mixed date representation was overlooked"
```

If there were no obvious misses, omit `--missed-issue`.

The score reports:

- candidate decisions resolved by the human reviewer;
- human/AI agreement among resolved candidates;
- **human rejected AI accept** — possible model overreach;
- **human approved AI defer** — possible model overcaution;
- approved/rejected/unresolved deterministic plan actions;
- deferred-candidate outcomes;
- review time;
- reviewer-reported missed issues.

These are product-development metrics, not clinical validation metrics.

## What would count as a promising pilot?

Do not reduce the result to one accuracy number. For early product development,
the strongest pattern would be:

- zero or very few human-rejected AI accepts;
- AI deferrals concentrated in genuinely ambiguous cases;
- most approved actions save meaningful manual cleanup work;
- little duplicate/repetitive review burden;
- no destructive or semantically unsafe proposal;
- few important reviewer-observed misses;
- a review workflow that feels materially faster than manual inspection.

One workbook is qualitative evidence, not generalization. Repeat the same protocol
on several structurally different research datasets before changing production
policy based on the results.
