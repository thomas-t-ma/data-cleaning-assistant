# Roadmap

## v0.4.2 — AI hardening (current)

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

### v0.4.x evaluation and provider hardening

- Run the benchmark repeatedly on qwen3.5:4b and at least one larger local model.
- Add more research-specific ambiguity cases and multi-table benchmark fixtures.
- Add prompt/provider regression fixtures.
- Add explicit context-size/token budgeting for very wide workbooks.
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
