# v0.7.2 — independent semantic stress test

The first two-pair local Ollama test passed 6/6 decisions across three runs.
This is a **separate, harder benchmark**, not a rerun of the public CoMMpass
workbook. It creates 40 tiny, synthetic, independent category fixtures:

- 20 expected safe corrections (clear typos, case/spacing, documented
  display-only zero padding).
- 20 expected deferrals (numbers, markers, doses, stages, and different
  study-specific codes).
- Three matched pairs of labels have opposite expected answers **because
  the explicitly supplied fictional dictionary context differs**. That
  context is model evidence, not a leaked accept/defer label.
- Label frequencies vary; a more common label is not proof of equivalence.

Each case passes through the **production** detector and bounded candidate
constructor. One case is sent to the model at a time. The model receives
only the two synthetic labels, generated candidate evidence, and applicable
fictional context. The scorer separately retains the hidden expected
accept/defer decision. There are no GDC case IDs, clinical records, file
paths, workbook loading, or data edits in this suite.

Some pairs (especially very short or dissimilar labels such as CD38/CD138
or Kappa/Lambda) may NOT cross the existing fuzzy-detector threshold.
This is recorded as **not exposed**, not incorrectly treated as a correct
model deferral. No model call is made for an unexposed pair. The report
separates detector coverage from model decision accuracy. An unexposed
unsafe merge is not a model success or an unsafe accepted merge.

## First run: detector audit with no model

From the repository root on your Mac:

\`\`\`bash
bash sync.sh

.venv/bin/python examples/semantic_stress_test.py \
  --audit-only \
  --output pilot_outputs/semantic_stress_coverage_v072.json
\`\`\`

Review the compact coverage summary:

\`\`\`bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

p = Path("pilot_outputs/semantic_stress_coverage_v072.json")
d = json.loads(p.read_text())
print("Independent cases:", d["unique_cases"])
print("Candidate exposure:", d["exposed_case_runs"], "/", d["total_case_runs"])
print("Safe correction exposure:", d["safe_coverage"], "/", d["safe_total"])
print("Risky pair exposure:", d["unsafe_exposure"], "/", d["unsafe_total"])
print("Not-exposed reasons:", d["coverage_reasons"])
for row in d["results"]:
    if not row["candidate_exposed"]:
        print("NOT EXPOSED:", row["id"], row["group"], row["coverage_reason"])
PY
\`\`\`

The audit creates no workbook and does not contact Ollama. Each output file
is write-once; choose another filename if you repeat an experiment.

## Next: model judgment, one independent run

Check your installed model name with \`ollama list\`. Substitute the exact
installed name below if it differs from this example:

\`\`\`bash
.venv/bin/python examples/semantic_stress_test.py \
  --model qwen3.5:4b \
  --runs 1 \
  --output pilot_outputs/semantic_stress_qwen35_4b_v072.json
\`\`\`

For repeated independent trials, use \`--runs 3\` with a DIFFERENT output
filename. You can use \`--think\` for a separate thinking-mode experiment,
or \`--url\` to select another Ollama endpoint. Nothing is applied to data
regardless of what the model returns.

Read summary and per-case rationales:

\`\`\`bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

p = Path("pilot_outputs/semantic_stress_qwen35_4b_v072.json")
d = json.loads(p.read_text())
print("Model:", d["model"])
print("Candidate coverage:", d["exposed_case_runs"], "/", d["total_case_runs"])
print("Correct model decisions:", d["correct_model_decisions"], "/",
      d["model_decision_denominator"])
print("Unsafe merges accepted:", d["unsafe_merges_accepted"])
print("Safe corrections deferred:", d["missed_safe_corrections"])
print("Protocol failures:", d["protocol_failures"])
print("Model errors:", d["model_errors"])
print("Mean call latency:", d["mean_call_latency_seconds"])
print("Strict full-suite pass:", d["strict_suite_pass"])
for row in d["results"]:
    print(row["id"], row["group"], row["expected"], "=>",
          row["actual"], "exposed:", row["candidate_exposed"],
          "correct:", row["correct"], "reason:", row["coverage_reason"])
    if row["rationale"]:
        print("  Rationale:", row["rationale"])
PY
\`\`\`

A correct decision on the two-label smoke test does not establish general
medical semantic safety. The corpus is synthetic, and certain accepted
typos depend on artificial fixture context. Review failure patterns and
detector misses; do not optimize only for a high headline percentage.

### Metrics and limitations

\`correct_model_decisions / model_decision_denominator\` counts surfaced
pairs on which the model was called, including response errors in its
denominator. Missing, unknown, or duplicate candidate IDs are NOT
credited as deferrals. \`unsafe_merges_accepted\` is the number of
expected-defer pairs for which the model actually returned ACCEPT, even
if there was also a response protocol error. \`missed_safe_corrections\`
counts expected-accept pairs on which it returned DEFER. \`safe_coverage\`
and \`unsafe_exposure\` include all intended cases, even those screened
out before the model.

\`strict_suite_pass\` requires **every fixture to be exposed and every
answer correct**, so a coverage gap makes it false even if the model
never accepts an unsafe merge. Use the separate counters for diagnosis;
do not infer that a detector miss is necessarily an unsafe outcome.
Model confidence is self-reported and is NOT calibrated probability.

This stress suite exercises semantic candidate review, not the approved
cell-edit/undo code path already tested in v0.7.0. The existing
\`examples/score_commpass_semantics.py\` two-pair smoke test remains intact.
