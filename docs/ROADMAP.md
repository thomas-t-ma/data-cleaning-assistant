# Roadmap

## v0.3.2 — deterministic hardening (current)

- Preserve original Excel workbook structure during cell-level cleaning.
- Protect formula-bearing sheets from unsafe row operations.
- Report recipe schema drift rather than only pass/fail compatibility.
- Reversibly remove exact duplicate rows with full row snapshots.
- Add surrounding-whitespace trimming.
- Reject in-place source overwrite.
- Keep no-op transformations out of audit logs and recipes.

## v0.4 — local AI planner

- Compact structured profiler context only.
- Local provider first, with Ollama as the initial backend.
- Optional hosted-provider abstraction later.
- Pydantic-validated `CleaningAction` proposals.
- Evidence-grounded rationales and confidence.
- No unrestricted dataframe-write access.
- Deterministic preview, review, execution, audit, recipe, and undo remain unchanged.

## v0.5 — local visual review UI

- Drag/drop dataset loading.
- Dataset quality dashboard.
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
