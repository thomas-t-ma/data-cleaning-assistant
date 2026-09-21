# Synthetic end-to-end pilot (v0.5.1)

Run from the **repository root** on your Mac. This uses bundled synthetic records only.
Nothing here requires an account or remote dataset. Ollama runs locally.

## A. Deterministic smoke test

```bash
bash sync.sh
.venv/bin/python -m pytest -q tests/test_v051_pilot_smoke.py
```

The test exercises `profile → plan → preview → review → apply → replay (preview)
→ undo`. It checks that the original workbook remains unchanged, only
explicitly approved schema-category mappings execute, duplicate observations are
preserved, the audit records schema provenance, and the restored workbook has
the original cell values. It does **not** benchmark a live Ollama model.

## B. Live local-model test

```bash
mkdir -p pilot_outputs
.venv/bin/python examples/make_messy_example.py

.venv/bin/data-cleaner profile examples/messy_research_example.xlsx \
  --schema examples/study_schema_example.yaml \
  -o pilot_outputs/profile.json

.venv/bin/data-cleaner ai-plan examples/messy_research_example.xlsx \
  --schema examples/study_schema_example.yaml \
  --model qwen3.5:4b \
  -o pilot_outputs/ai_plan.json \
  --context-out pilot_outputs/ai_context.json

.venv/bin/data-cleaner preview examples/messy_research_example.xlsx \
  pilot_outputs/ai_plan.json

.venv/bin/data-cleaner review examples/messy_research_example.xlsx \
  pilot_outputs/ai_plan.json \
  -o pilot_outputs/reviewed_plan.json
```

At the interactive review, approve only mappings you can justify from the
synthetic study dictionary. **Reject duplicate-row deletion**: the example
intentionally contains repeated measurements, and there is no proof they
should be discarded. Skip anything ambiguous or not covered by the dictionary.
A deferred AI item is non-executable until you explicitly approve it.

To see whether reviewed actions exist before applying:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from data_cleaning_assistant.actions import ActionDecision, CleaningPlan

plan = CleaningPlan.model_validate_json(
    Path("pilot_outputs/reviewed_plan.json").read_text()
)
print("approved:", sum(a.decision == ActionDecision.APPROVED for a in plan.actions))
print("pending AI reviews:", sum(r.status == "pending" for r in plan.deferred_reviews))
PY
```

If you approved at least one action:

```bash
.venv/bin/data-cleaner apply examples/messy_research_example.xlsx \
  pilot_outputs/reviewed_plan.json \
  -o pilot_outputs/cleaned.xlsx

.venv/bin/data-cleaner undo pilot_outputs/cleaned.xlsx \
  pilot_outputs/cleaned.xlsx.audit.json \
  -o pilot_outputs/restored.xlsx

.venv/bin/data-cleaner replay examples/messy_research_example.xlsx \
  pilot_outputs/cleaned.xlsx.recipe.yaml
```

The replay command is **preview-only** without `--apply`. Do not add
`--approve-all` to this pilot.

## C. What to examine

The original `examples/messy_research_example.xlsx` must remain unchanged.
The schema report should include unmatched subject keys and repeated declared
grain; neither should trigger automatic deletion. Inspect changes in
`pilot_outputs/cleaned.xlsx`, the audit's `executions` and
`study_schema_sha256`, and the restored values. Inspect
`pilot_outputs/ai_plan.json` for `deferred_reviews`; do not mistake a deferral
for an executable cleaning action.

Share **only the synthetic run's terminal output** or issue types/counts and
sanitized notes if you need help diagnosing a failure. Do not upload raw patient
data, identifiers, or model context generated from real clinical workbooks.

The real-data pilot begins only after the local synthetic run has been verified.
