# Roadmap

## v0.4.0 — local AI planner (current)

- Privacy-minimized structured profiler context.
- Exact AI-context inspection/export.
- Ollama provider using structured JSON-schema output.
- Hybrid deterministic + AI planning.
- Deterministic validation of all AI mutations.
- Evidence-gated fuzzy category merging.
- Action-origin tracking.
- No unrestricted dataframe access for the model.

### v0.4.x evaluation and provider hardening

- Benchmark several small local models on a fixed synthetic cleaning corpus.
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
