# Data Cleaning Assistant

A local-first, research-aware data quality and cleaning assistant for CSV, Excel, and Parquet datasets.

Current version: **v0.4.1**.

The core rule is **detection is not transformation**. Profiling is read-only. Deterministic rules and optional local AI can propose changes, but only deterministic Python previews and executes them after explicit human approval.

## Current capabilities

The profiler detects hidden missing-value tokens, exact duplicate rows, constant columns, category inconsistencies, surrounding whitespace, mixed numeric values, malformed dates, likely identifiers, unnamed columns, conservative statistical outliers, candidate keys, repeated entities, dataset grain, and cross-table relationships.

Reviewable transformations include category normalization, missing-token normalization, date standardization, numeric coercion for non-identifiers, whitespace trimming, and reversible exact-duplicate removal.

## v0.4 local AI planner

v0.4 adds an optional local Ollama planning layer.

The AI **does not receive the dataframe and does not write data**. It receives compact profiler context and returns structured proposals. Those proposals must pass two gates:

1. Ollama structured output must satisfy the Pydantic JSON schema.
2. A deterministic local validator must confirm that every proposed mutation is supported by profiler evidence and observed local values.

Examples of enforced restrictions:

- category merges can only use observed labels already surfaced as a detector pair
- missing-value normalization can only use tokens already detected as missing-like
- identifier-like columns cannot be proposed for numeric coercion
- duplicate removal requires an exact-duplicate detector finding
- outlier deletion and imputation are not supported AI actions
- cyclic or chained category mappings are rejected

The default `ai-plan` mode is **hybrid**: deterministic proposals remain in the plan and validated AI suggestions are added when they contribute something new.

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

A modest starting example:

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

Run the benchmark against the installed local model:

```bash
.venv/bin/data-cleaner ai-benchmark \
  --model qwen3.5:4b \
  --output ai_benchmark.json
```

For a more stable estimate, repeat every case three times:

```bash
.venv/bin/data-cleaner ai-benchmark \
  --model qwen3.5:4b \
  --runs 3 \
  --output ai_benchmark_qwen35_4b.json
```

The benchmark reports:

- **required-action recall** — how often the model proposes the cleaning action the case requires
- **accepted-action precision** — how many validator-surviving actions are actually correct
- **forbidden accepted** — semantically unsafe actions that survived deterministic validation
- **validator catch rate** — forbidden model proposals blocked by the validator
- **case-run pass rate** — cases with all required actions found and no unexpected accepted actions
- **latency** — mean local-model time per synthetic case

The safety cases deliberately distinguish two failure modes. Some bad proposals, such as coercing an identifier, should be blocked by deterministic validation. Others, such as merging lexically similar but scientifically distinct categories like `Stage II` and `Stage III`, may be detector-permitted and therefore test the model's actual judgment.

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
