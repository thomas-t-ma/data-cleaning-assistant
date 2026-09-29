# Changelog

## 0.9.0

- Restricted normal local-AI planning to bounded semantic category-equivalence candidates; mechanical missing-token, whitespace, date, and numeric cleanup remains deterministic human-review work.
- Kept the historical synthetic benchmark on the broader candidate surface so v0.8 regression results remain comparable.
- Added `review-grouped` to review exact transformation families by origin, type, and parameters instead of forcing repetitive action-by-action approval.
- Multi-action family approval requires an explicit second `YES` confirmation; grouped review never applies edits.
- Added typed review-session provenance with per-family/per-action decisions, affected counts, bulk-confirmation state, and automatic elapsed time.
- Persisted review-session provenance from the reviewed plan into the execution `ChangeSet` audit.
- Aligned `ai-context` with the production semantic-only model surface.
- Added direct regression coverage that malformed Ollama structured output is wrapped as a provider error and therefore fails closed.
- The routing/grouping changes were motivated by a public NHANES-derived workflow pilot in which 71 redundant AI reviews collapsed to zero while all 72 deterministic proposals remained available; grouped review reduced those 72 actions to four exact transformation families. This is product/workflow evidence, not scientific or clinical validation.


## 0.8.0

- Promoted the evidence-calibrated semantic-review prompt from benchmark-only testing into the production local AI planner.
- Added adaptive consensus: two calibrated reviews per candidate set, with a third tie-break review only when candidate decisions disagree.
- Fail closed to non-executable human review on structured-output/model errors, missing selections, or malformed duplicate candidate selections.
- Preserved deterministic ownership of action parameters and mandatory human approval before execution.
- Added CLI reporting for review-call count and candidate disagreements.
- Added offline regressions for 2/2 agreement, 2/3 accept/defer majorities, unknown candidate IDs, and structured-output failures.
- Production selection was informed by a fresh frozen 100-case synthetic validation that produced 47/50 safe accepts and 0/50 unsafe accepts with no protocol failures; this is engineering validation, not clinical validation.

## 0.7.3

- Fixed the s20 benchmark false coverage gap when independent whitespace trimming coexists with a bounded category mapping.
- Added optional baseline versus evidence-calibrated benchmark-only prompt comparison with prompt variant and prompt SHA-256 recorded in scores.
- Retained production AI planner instructions while testing whether real-model typo recall improves without additional unsafe merges.
- Added offline regressions for overlapping proposals and isolated prompt control.

## 0.7.2

- Added forty independently evaluated synthetic semantic-label cases: twenty proposed corrections and twenty expected deferrals, including context-contrast controls and varied frequencies.
- Ran each case through production fuzzy detection/candidate construction and a separate synthetic-only model call; withheld expected decisions from the prompt.
- Reported detector coverage separately from model decisions, with unsafe-merge acceptances, safe-correction deferrals, malformed/missing model selections, rationale and latency.
- Added an offline detector-only audit and non-overwriting, repeatable model-scoring JSON.
- Added fake-provider tests for correct/unsafe decisions, protocol errors, 40-case accounting, and opposite contextual answers.

## 0.7.1

- Added isolated, model-specific local Ollama semantic evaluation on the already generated synthetic BenchmarkLabels sheet.
- Scored literal accept/defer judgments against a withheld answer key, including missing, duplicate, and unknown candidate-ID protocol failures.
- Recorded per-run model rationales, confidence, success, and latency to a non-overwriting local JSON report.
- Kept original public CoMMpass sheet profiles, identifiers, source path, and answer key outside the model prompt.
- Added offline fake-provider tests for both correct and unsafe decisions, missing outputs, reproducible fixture context, and unchanged workbook hashes.

## 0.7.0

- Added a local-only seeded corruption challenge for the already downloaded public CoMMpass clinical workbook; preserves original input and writes a separate test copy and answer key.
- Added four realistic existing-sheet defects and a clearly synthetic sixth sheet for fuzzy category semantic-review safety.
- Added answer-key scoring for deterministic plan targets, bounded effects, unexpected proposals, and fuzzy candidate exposure.
- Added explicit fixture-only apply/Excel export/undo validation with both original and challenge input hash checks.
- Added an offline multi-table CI integration test, reproducibility checks, and the Mac challenge runbook.

## 0.6.3

- Treated literal zero-length cells as already empty for CSV/Excel column coverage rather than hidden missing-value tokens.
- Counted blank cells in null percentages, omitted blank samples, and summarized entirely blank fields in the terminal.
- Prevented routine `'' → None` actions that arose from pandas Excel blank representation; left nonempty sentinels and explicit study rules reviewable.
- Added Excel/CSV regression tests protecting nonempty missingness tokens, whitespace-only strings, schema overrides, and untouched source bytes.

## 0.6.2

- Kept structural-only all-missing columns as QA findings while omitting low-value routine null-normalization actions.
- Persisted non-executable, issue-linked planning notes with compact per-table CLI counts; explicit study dictionary overrides remain reviewable.
- Aligned deterministic, hybrid-AI and AI-only proposal surfaces with the same conservative triage.
- Regressed wide clinical exports, mixed contextual/structural missingness, schema overrides, and unchanged source bytes.

## 0.6.1

- Added a read-only guard for generic numerical result columns mixed across multiple assay/unit metadata categories.
- Replaced potentially misleading pooled outlier findings with an explicit scope warning, even when the pooled detector found no outliers.
- Preserved opt-in within-assay/unit `numeric_groups` checks and left specific independent numeric columns unchanged.
- Clarified inferred candidate keys, declared relationship basis, distinct shared-key counts, and informational QA headings.
- Added regression coverage for heterogeneous, single-assay, explicitly grouped, and no-pooled-outlier cases.

## 0.6.0

- Grouped complete-coverage constant/missingness QA findings in terminal reports; original issue records remain in JSON and `profile --all-issues` restores expanded display.
- Prefer study-declared grain and foreign keys over coincidental shared-identifier inferred relationships.
- Added opt-in `numeric_groups` to scope numeric outlier flags within declared assay/unit strata, without changing any values.
- Added independent-preview overlap warnings in CLI preview/review, retaining accurate sequential execution audit counts.
- Added regression tests for lossless summaries, foreign-key precedence, grouped outliers, schema validation and overlapping edits.

## 0.5.2

- Added a reproducible download of open NCI GDC MMRF-CoMMpass clinical metadata, using fixed project filtering, pagination, and fail-closed completeness checks.
- Preserved the untouched source response plus API URLs, retrieval timestamp, row counts, and source SHA-256.
- Generated a linked multi-sheet exploratory workbook and conservative study schema without guessing clinical categorical or missingness semantics.
- Added offline regression tests for pagination, in-project filtering, repeated visits/labs, workbook structure, and schema QA.
- No live GDC API download in CI; first Mac retrieval still requires validation.

## 0.5.1

- Added an end-to-end synthetic CLI pilot smoke test covering profiling, schema planning, preview, interactive review, approved edits, audit/recipe generation, preview-only replay, and undo.
- Verified source workbook integrity and preserved duplicate observations across the workflow.
- Added a Mac runbook for live Ollama validation on bundled synthetic data before any de-identified pilot.
- Ignored generated pilot outputs and synthetic workbooks in git.

## 0.5.0

- Added strict, optional YAML/JSON study schema validation and a synthetic workbook example.
- Added declared grain-key missingness and uniqueness, allowed-category, cross-table foreign-key,
  non-unique parent-key, and visit-order chronology QA findings (all read-only).
- Added exact study-authorized canonical mappings and study-specific missing tokens as proposed,
  human-reviewed deterministic actions; identifier edits and unknown schema references fail closed.
- Schema-managed category columns no longer receive conflicting AI fuzzy-merge candidates.
- Added source schema SHA-256 provenance to plans and execution audit logs.
- Persisted AI deferrals as non-executable review records with evidence, rationale, and decision.
- Added CLI manual review for deferred candidates; approve-all cannot silently execute deferrals.
- Added regression coverage for semantic missingness, exact schema mappings, chronology,
  foreign keys, deferred persistence, explicit approval, and CLI schema options.

## 0.4.7

- Added conservative single-edit typo-shape evidence for fuzzy category candidates.
- Typo evidence requires exactly one differing token, edit distance 1, identical surrounding tokens, non-numeric differences, and token length of at least four characters.
- Added frequency context showing whether the proposed canonical label is more common than the variant.
- Explicitly excludes numeric and short-token changes so dose, stage, treatment-arm, and similar coded distinctions do not receive typo evidence.
- Updated the AI review prompt to treat single-edit evidence as strong but still deferrable when scientific meaning could differ.
- Added regression tests for `Peripheral Blod → Peripheral Blood` and for negative controls such as `Stage II/III`, `Dose 5/50`, and `Arm A/B`.

## 0.4.6

- Split structural null sentinels from contextual missingness labels.
- `Unknown`, `Unk`, `Not available`, and `Not applicable` remain visible QA findings but are no longer treated as null-like by semantic inference.
- Deterministic and AI planners no longer generate null-normalization actions for contextual missingness without an explicit schema.
- Hidden-missing issues now report structural and contextual token groups separately.
- Added deterministic zero-padding equivalence evidence for fuzzy category candidates such as `Visit 01` vs `Visit 1`.
- Updated the AI review prompt to treat zero-padding equivalence as strong formatting evidence.
- Added benchmark-level `must_not_expose` safety expectations and an `unsafe candidates exposed` metric.
- Reclassified duplicate deletion, identifier coercion, and contextual-null collapse as candidate-surface safety tests.
- Added regression tests covering zero-padding evidence and contextual missingness across both AI and deterministic planning.

## 0.4.5

- Added separate `core`, `research`, and `all` AI benchmark suites.
- Preserved the original 8-case core suite unchanged for stable regression tracking.
- Added a 12-case research-oriented suite focused on semantic accept/defer judgment.
- Added positive cases for visit/cohort formatting variants, specimen typos, censored numeric lab values, and specimen whitespace.
- Added safety cases for clinically distinct dose levels, unit differences, study timepoints, treatment arms, specimen types, explicit `Unknown` outcome categories, and `Not applicable` adverse-event grades.
- Added structural tests ensuring every research answer key corresponds to a real deterministic AI candidate.
- Added `--suite core|research|all` to `ai-benchmark`.
- Documented `qwen3.5:4b` as the current recommended local baseline after a 24/24 perfect core stability run.

## 0.4.4

- Added a no-candidate fast path: Ollama is not called when deterministic profiling produces no AI-review candidates.
- Eliminates spurious unknown-candidate responses on duplicate-only and identifier-only safety cases.
- Reduces unnecessary latency and removes meaningless validator rejections for empty candidate sets.
- Added regression coverage proving the provider is never invoked when the candidate list is empty.

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
