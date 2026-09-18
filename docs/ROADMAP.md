# Roadmap

## v0.3 — safe transformations (current)

- Structured `CleaningAction` objects.
- Before/after previews.
- Explicit approval/rejection workflow.
- Source-hash validation.
- Reversible cell-level `ChangeSet` execution.
- Cleaning recipe + audit log export.
- Undo with cleaned-file hash validation.

### Follow-up hardening

- Recipe replay against new compatible datasets.
- Workbook-format/style preservation for Excel.
- Safer row-level transformations with reversible row snapshots.
- Interactive plan editing without hand-editing JSON.

## v0.4 — local AI planner

- Compact structured context only; no unrestricted dataframe writes.
- Local provider first (Ollama), hosted APIs optional.
- Pydantic-validated action proposals.
- Deterministic execution after review.

## v0.5 — local visual review UI

- Drag/drop dataset loading.
- Dataset health dashboard.
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
