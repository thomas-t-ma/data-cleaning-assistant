# Changelog

## 0.3.1

- Added interactive CLI review with approve/reject/defer decisions.
- Added YAML recipe loading and compatibility validation.
- Added safe recipe replay with read-only preview by default.
- Added explicit `--apply` gate for recipe execution.
- Added single-table recipe remapping across differently named CSV/Parquet files.
- Added rejection for missing recipe tables/columns instead of partial application.
- Added regression tests for replay safety and output generation.
- Hardened sync scripts to repair incomplete environments and rebuild virtualenvs missing pip.

## 0.3.0

- Added structured cleaning plans and typed `CleaningAction` objects.
- Added deterministic proposals for category normalization, textual missing values, date standardization, and numeric coercion.
- Added before/after cell previews.
- Added explicit approval gates before execution.
- Added source SHA-256 validation to reject stale plans.
- Added deep-copy transformation execution; source files remain untouched.
- Added cell-level reversible `ChangeSet` audit logs.
- Added cleaned-output hashes for safe undo validation.
- Added reproducible YAML recipe export.
- Added CLI commands: `plan`, `preview`, `apply`, and `undo`.

## 0.2.0

- Added candidate key inference.
- Added initial dataset-grain inference.
- Added repeated-entity detection for longitudinal/repeated-measures data.
- Added cross-sheet identifier relationship discovery.
- Added conservative fuzzy-category suggestions.
- Added GitHub Actions test/lint workflow.

## 0.1.0

- Initial read-only profiling engine.
- CSV/XLSX/XLSM/Parquet ingestion.
- Core deterministic data-quality detectors.
- CLI and JSON report output.
