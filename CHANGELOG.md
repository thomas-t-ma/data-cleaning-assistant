# Changelog

## 0.4.3

- Replaced free-form AI action authoring with bounded candidate selection.
- Deterministic code now owns table names, columns, action types, mappings, tokens, formats, and parameters.
- The model only returns short candidate IDs such as `C001`, an `accept` or `defer` decision, rationale, and confidence.
- Added deterministic candidate construction from profiler issues.
- Added frequency-aware fuzzy-category candidates while preserving semantic review by the model.
- Removed row deletion and identifier numeric coercion from the AI candidate surface entirely.
- Added candidates to `ai-context` so exported context still exactly matches what the model receives.
- Updated benchmark scoring to measure candidate acceptance/defer behavior rather than model-authored action formatting.
- Added regression tests for bounded fuzzy merges, semantic deferral, absent unsafe candidate surfaces, unknown candidate IDs, and CLI integration.

## 0.4.2

- Made model-supplied issue IDs advisory provenance rather than a mutation-safety gate.
- Valid actions now attach independently matched local detector issue IDs even when a small model copies an ID incorrectly.
- Removed AI authority to propose row deletion; exact duplicates remain deterministic human-review actions.
- Added rejected-proposal parameters and rationale for audit/debugging.
- Made forbidden-action validator catch-rate accounting exact rather than inferred from counts.
- Added failed-case benchmark diagnostics directly to the CLI.
- Removed redundant response-schema text from the user prompt because Ollama already receives the JSON schema through structured output.
- Added configurable Ollama thinking mode with `--think/--no-think`; structured planning defaults to non-thinking mode.
- Added regression tests for incorrect issue IDs, AI row-deletion rejection, exact benchmark catch accounting, and Ollama thinking configuration.

## 0.4.1

- Added a fixed synthetic AI-cleaning benchmark with hidden answer keys.
- Added `ai-benchmark` with configurable repeated runs and detailed JSON output.
- Added required-action recall and accepted-action precision metrics.
- Added explicit forbidden-action safety cases.
- Added validator catch-rate reporting for unsafe model proposals blocked by deterministic checks.
- Added semantically distinct fuzzy-category stress testing where the validator cannot substitute for model judgment.
- Added local-model latency measurement per benchmark case.
- Benchmark execution isolates AI behavior by disabling deterministic-rule proposals during scoring.
- Added benchmark scoring regression tests with fake providers; CI still does not require Ollama.
- Added benchmark artifacts to Git ignore defaults.
- Updated local examples to use `qwen3.5:4b`.

## 0.4.0

- Added a provider abstraction for structured AI planning.
- Added local Ollama support through the HTTP API with no Ollama Python dependency.
- Added `ai-status` to verify Ollama and list installed local models.
- Added `ai-context` to export exactly what the model would receive.
- Added `ai-plan` for hybrid deterministic + AI cleaning plans.
- Added Pydantic JSON-schema structured-output enforcement.
- Added a second deterministic safety-validation layer for every AI proposal.
- Added evidence-gated fuzzy category merging.
- AI category mappings are restricted to observed detector-supported label pairs.
- AI missing-token proposals are restricted to detector-confirmed missing-like tokens.
- Identifier-like columns cannot be AI-proposed for numeric coercion.
- AI cannot propose unsupported outlier deletion, imputation, or non-exact duplicate removal.
- Added category mapping cycle/chain rejection.
- Added action-origin metadata (`rule`, `ai`, `recipe`).
- Removed raw samples, source paths, identifier values, and identifier numeric summaries from AI context.
- Added Git ignore defaults for local AI contexts, plans, recipes, and audit artifacts.
- Added provider-independent AI safety tests; CI does not require a live model.

## 0.3.2

- Added source-workbook-preserving Excel output for approved cell edits.
- Preserves untouched styles, widths, merged cells, freeze panes, and formulas.
- Added explicit formula safety guards for worksheet row deletion/insertion.
- Added reviewable surrounding-whitespace trimming.
- Added reviewable exact duplicate-row removal.
- Added full deleted-row snapshots and positional undo.
- Added persisted save/reload/undo coverage for CSV and Excel duplicate removal.
- Added recipe schema snapshots and schema-drift reporting.
- Added source overwrite protection and omission of no-op executions from audit/recipe output.

## 0.3.1

- Added interactive CLI review with approve/reject/defer decisions.
- Added safe YAML recipe replay with compatibility validation and preview-by-default behavior.
- Hardened sync scripts to repair incomplete environments and rebuild virtualenvs missing pip.

## 0.3.0

- Added structured cleaning plans, deterministic transformations, previews, explicit approvals, reversible ChangeSets, audit logs, recipes, and undo.

## 0.2.0

- Added research-aware structure inference, cross-table relationships, repeated-entity detection, and conservative fuzzy-category suggestions.

## 0.1.0

- Initial read-only profiling engine.
