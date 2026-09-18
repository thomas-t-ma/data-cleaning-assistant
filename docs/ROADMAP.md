# Roadmap

## v0.3.1 — replay + review (current)

- Interactive plan review without hand-editing JSON.
- Recipe compatibility validation.
- Recipe preview before execution.
- Explicit replay approval with `--apply`.
- Single-table filename remapping for CSV/Parquet.

### Remaining hardening before AI

- Workbook-format/style preservation for Excel.
- Recipe/schema drift reporting rather than only pass/fail compatibility.
- Safer row-level transformations with reversible row snapshots.
- More deterministic transformations and configurable rule packs.

## v0.4 — local AI planner

- Compact structured context only; no unrestricted dataframe writes.
- Local provider first (Ollama), hosted APIs optional.
- Pydantic-validated action proposals.
- Deterministic execution after review.
- Explanations grounded in profiler evidence rather than raw free-form guesses.

## v0.5 — local visual review UI

- Drag/drop dataset loading.
- Dataset quality dashboard.
- Side-by-side change previews.
- Accept/reject/modify controls.
- Audit and recipe export from the UI.

## Research-mode expansion

- Temporal consistency rules.
- Unit inference and harmonization suggestions.
- Missingness-pattern analysis.
- Leakage detection.
- Data-dictionary generation.
- Methods-text generation from the actual audit log.
