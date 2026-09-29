# Roadmap

## Current production: v0.9.0

v0.9.0 established the conservative execution and review boundary for the
project:

- deterministic preprocessing remains the default source of executable proposals;
- local AI is reserved for bounded semantic category-equivalence decisions;
- semantic review is split into focused batches of at most four candidates;
- multi-select whole-value fuzzy merges are blocked;
- free-response/comment fields are excluded from category normalization and fuzzy
  semantic review;
- conflicting AI-accepted canonical mappings fail closed to human review;
- mechanical actions can be reviewed in exact transformation families;
- review-session provenance is retained in the eventual audit log;
- source datasets are never edited in place;
- human approval remains mandatory before execution.

Public NHANES-derived and BES workflow pilots were used to expose review-burden
and failure modes. They are workflow validation, not evidence of complete issue
recall, scientific correctness, clinical safety, or generalization.

## v0.10 — recall without relaxing safety

The primary v0.10 objective is to identify more of the cleaning work a real
researcher would need while preserving the v0.9 execution boundary.

### 1. Structured multi-select cleaning

Status: **implemented; validation in progress**

- Require repeated delimiter structure before treating a field as multi-select,
  so occasional semicolons in scalar prose do not activate this path.
- Detect semicolon-delimited categorical fields separately from ordinary scalar
  categories.
- Normalize exact capitalization component variants without comparing the whole
  cell as one category.
- Preserve every selected option and its order.
- Do not choose a canonical component casing when observed variants are tied;
  leave the ambiguity visible for schema/human review.
- Keep fuzzy component merging disabled until it has its own evidence and
  validation strategy.
- Support preview, audit, recipe replay, and undo through a dedicated reversible
  action type.

Initial safety invariant:

> Cleaning one component must never remove, add, or infer another selected
> component.

### 2. Conditional and cross-column consistency QA

Status: **initial non-executable detector implemented**

- Detect explicit If yes / If no dependent fields and explicit skip instructions
  such as "SKIP THIS Q IF NO README EXISTS", linking them only to a nearby
  yes/no-like parent field.
- Flag contradictions and trigger-state missing detail as QA findings only.
- Detect repeated conditional structures such as a parent yes/no field followed
  by a dependent detail field.
- Flag contradictions and suspicious populated/empty dependent fields without
  auto-editing them.
- Prefer explicit study-schema relationships when available.
- Keep inferred conditional relationships non-executable until separately
  validated.

Examples:

- “Has README?” versus “README usefulness”.
- “Has license?” versus “license type”.
- “Was cited?” versus “citation count”.

### 3. Duplicate and entity-resolution recall

Status: **initial normalization-equivalent QA implemented**

- Flag non-identical rows that become identical only after safe text normalization
  and structural-missing normalization.
- Keep the finding non-executable: no automatic row deletion or merging.
- Move beyond exact duplicate rows to bounded near-duplicate record suggestions.
- Separate identifier disagreement from ordinary field disagreement.
- Never delete or merge rows automatically.
- Provide row-pair evidence and explicit human review.

### 4. Schema and codebook inference

Status: **initial read-only codebook suggestions implemented**

- Suggest dominant recurring categorical vocabularies with coverage statistics,
  rare write-ins, exact case/whitespace mappings, and unresolved ties.
- Keep suggestions separate from StudySchema authority until explicitly reviewed
  and promoted by the researcher.
- Suggest candidate categorical codebooks from repeated stable labels.
- Distinguish raw observed spellings from a true canonical vocabulary.
- Allow users to promote reviewed codebooks into explicit study schemas.
- Use schema-defined canonical labels to resolve otherwise ambiguous pairwise
  normalization.

This directly addresses the v0.9 BES finding where an AI merge could be
semantically reasonable while still choosing another noisy raw spelling as the
target.

### 5. Cleaning-pattern and recoding suggestions

Planned:

- Surface repeated transformation patterns that are not safely expressible as
  lexical normalization.
- Keep study-specific recoding suggestions separate from generic cleaning.
- Require explicit user-authored or schema-backed rules before execution.
- Preserve raw values and transformation provenance.

### 6. Recall-oriented evaluation

Status: **deterministic recall-surface report implemented**

- `data-cleaner recall-report` summarizes new v0.10 review surfaces without
  Ollama calls or dataset edits.
- `data-cleaner recall-benchmark` runs a frozen synthetic suite with required
  recall and forbidden-exposure controls for the new deterministic features.
- The report separates detector/action exposure from scientific correctness and
  complete issue recall.

Before claiming a v0.10 improvement:

- build synthetic fixtures for each new detector family with hidden negative
  controls;
- measure candidate exposure/recall separately from acceptance precision;
- retain unsafe-exposure and fail-closed metrics;
- rerun at least one untouched public research workflow;
- compare surfaced work with the public study workflow qualitatively rather than
  treating its cleaned output as a universal answer key;
- report what the assistant still misses.

## Later product work

### Local visual review UI

- Drag/drop dataset loading.
- Dataset quality dashboard.
- AI/rule origin badges.
- Side-by-side change previews.
- Accept/reject/modify controls.
- Schema-drift views.
- Benchmark/model settings.
- Audit and recipe export.

### Research-mode expansion

- Temporal consistency rules.
- Unit inference and harmonization suggestions.
- Missingness-pattern analysis.
- Outcome leakage detection.
- Study-grain and longitudinal validation.
- Data-dictionary generation.
- Methods-text generation from the actual audit log.
- Configurable research rule packs.
