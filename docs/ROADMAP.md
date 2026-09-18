# Roadmap

## v0.2 — research-aware structure (current)

- Candidate primary/composite keys.
- Repeated-entity detection without mislabeling longitudinal records as duplicates.
- Cross-sheet identifier relationships.
- Conservative fuzzy category suggestions.
- Initial table-grain inference (subject, visit, specimen, procedure, medication, event, observation).

## v0.3 — safe transformations

- Structured `Action` objects.
- Before/after previews.
- Accept / reject / modify workflow.
- Reversible `ChangeSet` execution.
- Cleaning recipe + audit log export.

## v0.4 — local AI planner

- Compact structured context only; no unrestricted dataframe writes.
- Local provider first (Ollama), hosted APIs optional.
- Pydantic-validated action proposals.
- Deterministic execution after review.

## Research-mode expansion

- Temporal consistency rules.
- Unit inference and harmonization suggestions.
- Missingness-pattern analysis.
- Leakage detection.
- Data-dictionary generation.
- Methods-text generation from the actual audit log.
