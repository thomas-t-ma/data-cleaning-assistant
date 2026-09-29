# BES v0.10 deterministic recall check

This note records a read-only validation of the v0.10 recall foundation against
the public anonymized BES hackathon raw export used during v0.9 validation.

## Source

- Upstream repository: `nhcooper123/reproduce-reuse-recycle`
- Pinned upstream commit:
  `ef25b60f333452b0baad7a5a4d0bccc6386e67af`
- Raw source:
  `raw-data/BES-data-code-hackathon-raw-outputs_ANON_2025-11-16.csv`
- Shape: 2,126 rows × 67 columns

The source is treated as read-only. The study authors' cleaned dataset and
cleaning script are qualitative reference material, not a universal answer key.

## v0.10 review surface

Using the current deterministic v0.10 detector logic:

### Structured multi-select normalization

- 8 fields contain component-level capitalization inconsistencies.
- 7 fields yield an executable `normalize_multiselect` proposal.
- Those proposals affect 37 cells in total.
- 1 field is tie-only and therefore produces no executable normalization.
- Tied component spellings remain unresolved instead of selecting a canonical
  spelling arbitrarily.
- The earlier scalar citation-count false positive is no longer classified as
  multi-select because semicolons are not a repeated field structure there.

Examples of reviewable component variants include:

- `pangaea` → `PANGAEA`
- `MOVEBANK` / `Movebank` → `MoveBank`
- `Jags` → `JAGS`
- `NetLOGO` → `NetLogo`

Examples intentionally left unresolved include frequency ties such as
`BUGS` / `Bugs`.

These are capitalization proposals only. The transformation preserves every
selected component, its order, delimiters, and surrounding delimiter spacing.

### Conditional consistency QA

8 explicit parent/dependent relationships are detected from either an
`If yes` / `If no` dependent field or an explicit skip instruction such as
`SKIP THIS Q IF NO README EXISTS`.

Across those 8 pairs:

- 96 parent/dependent pair-row findings are surfaced.
- 76 unique rows participate in at least one finding.
- Findings include both contradictions and missing detail under an active
  condition.
- Numeric count fields treat values such as `0` and `1` as counts rather than
  Boolean spellings.
- Nearby numeric count columns are not eligible as inferred Boolean parents
  unless the column name itself is Boolean-like.
- Non-adjacent conditional parents require lexical evidence; an adjacent
  semantically matching binary parent takes precedence over older incidental
  lexical matches.

Examples include:

- README absent while a README usefulness score is populated.
- README present while the conditional usefulness field is missing.
- License parent = `No` while a license value is populated.
- Citation parent = `Yes` with a zero or missing conditional citation count.

These findings are QA-only and create no cleaning action.

### Normalization-equivalent duplicate QA

- 0 groups are found in this BES source.

This is a valid negative result. v0.10 does not force a duplicate finding merely
to increase recall. The detector only flags non-identical rows that become
identical after safe text and structural-missing normalization, and it never
authorizes row deletion.

### Read-only codebook suggestions

After excluding identifiers, free text, dates, numeric/count fields, and
structured multi-select fields:

- 25 columns yield dominant observed-vocabulary suggestions.

The report keeps rare/other values visible and proposes canonical mappings only
for exact case/whitespace variants with a unique frequency winner. Ties remain
unresolved.

These suggestions are not automatically promoted to `StudySchema` and
authorize no transformations.

## Synthetic deterministic benchmark

The frozen v0.10 recall suite currently contains 14 positive/negative-control
cases spanning:

- multi-select component normalization;
- selected-option preservation;
- semicolon-in-prose rejection;
- canonical tie deferral;
- conditional parent/dependent QA;
- numeric count semantics;
- normalization-equivalent duplicate QA;
- exact-duplicate separation;
- codebook positive exposure;
- count-field codebook rejection.

At the current checkpoint the suite passes 14/14 with:

- required recall: 100%;
- forbidden exposures: 0.

This is an engineering regression suite, not evidence of complete recall or
scientific correctness.

## Interpretation

Compared with v0.9, v0.10 now surfaces additional structured cleaning and QA
work that v0.9 intentionally left untouched, while preserving the v0.9 safety
boundary:

- new executable authority is limited to component-level capitalization
  normalization in structured multi-select cells;
- conditional, duplicate-equivalence, and codebook features remain read-only or
  human-review-only;
- no AI authority was expanded;
- no source data was modified during validation.

The remaining recall gap still includes study-specific recoding, row-specific
manual corrections, domain-defined canonical vocabularies, and more general
entity resolution.
