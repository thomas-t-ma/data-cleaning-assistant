# Roadmap

## v0.7.2 — forty-case independent semantic stress test (current)

- Fixed synthetic cleaning corpus with hidden answer keys.
- AI-only scoring so deterministic rules do not inflate model performance.
- Required-action recall.
- Accepted-action precision.
- Forbidden-action acceptance counts.
- Deterministic validator catch-rate measurement.
- Semantically distinct fuzzy-category safety stress tests.
- Repeatable local-model latency measurement.
- Detailed JSON benchmark artifacts for model comparisons.
- Soft issue-ID provenance backed by independent local evidence matching.
- No AI row-deletion authority.
- Exact rejected-proposal diagnostics and validator catch accounting.
- Non-thinking Ollama mode by default for structured planning, with opt-in `--think`.
- Deterministic construction of bounded candidate actions.
- Short session-local candidate IDs for small-model reliability.
- AI limited to semantic accept/defer decisions rather than action authoring.
- Unsafe capabilities such as row deletion and identifier coercion removed from the AI candidate surface.
- Skip the model entirely when no AI-review candidates exist.
- Preserve the original core suite as a fixed regression benchmark.
- Add a harder research suite with realistic semantic ambiguity and meaningful missing-like categories.
- Support `core`, `research`, and combined benchmark execution.
- Distinguish structural null sentinels from contextual missingness labels.
- Remove contextual null-collapse actions from both deterministic and AI candidate surfaces by default.
- Add explicit unsafe-candidate-exposure metrics.
- Add numeric zero-padding equivalence evidence for research category labels.
- Add conservative single-edit typo-shape evidence while excluding numeric and short-token semantic distinctions.
- Add opt-in study dictionaries and read-only longitudinal/cross-table QA.
- Persist non-executable AI deferrals for explicit researcher review.
- Acquire and audit open CoMMpass clinical metadata before real-data modeling.
- Summarize complete-coverage findings without dropping them from JSON.
- Prefer declared foreign-key relationships and grain over inferred shared identifiers.
- Scope outlier QA to declared assay/unit groups.
- Show overlap warnings for independently previewed actions.
- Warn when unscoped generic numeric measurements mix multiple assay/unit categories.
- Suppress routine actions for entirely structural-only columns, retaining non-executable issue-linked planning notes.
- Treat exact blank CSV/Excel cells as existing missingness coverage, not proposed null conversions.
- Seed four bounded test defects in an isolated public-data workbook and score deterministic proposals against a private answer key.
- Exercise fixture-only transformations, persistence, and undo; protect original and challenge input hashes.
- Score local model accept/defer decisions on two isolated synthetic-label pairs, with per-run latency and protocol failures.

### v0.4.x evaluation and provider hardening

- Run the research benchmark repeatedly on qwen3.5:4b and at least one larger local model.
- Add multi-table benchmark fixtures and longitudinal temporal-consistency scenarios.
- Add prompt/provider regression fixtures.
- Add explicit context-size/token budgeting for very wide workbooks.
- Expand study-schema rules with typed field/unit validation and exportable data dictionaries.
- Add user-configurable redaction policies for category/evidence values.
- Add provider metadata and model/version information to plan/audit provenance.
- Add optional hosted-provider adapters without changing execution semantics.

## v0.5 — local visual review UI

- Drag/drop dataset loading.
- Dataset quality dashboard.
- AI/rule origin badges.
- Side-by-side change previews.
- Accept/reject/modify controls.
- Schema-drift views.
- Benchmark/model settings.
- Audit and recipe export from the UI.

## Research-mode expansion

- Temporal consistency rules.
- Unit inference and harmonization suggestions.
- Missingness-pattern analysis.
- Outcome leakage detection.
- Study-grain and longitudinal validation.
- Data-dictionary generation.
- Methods-text generation from the actual audit log.
- Configurable research rule packs.
