from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .actions import ActionDecision, ChangeSet, CleaningPlan
from .ai_candidates import build_ai_candidates, semantic_review_candidates
from .ai_context import build_ai_context
from .ai_planner import build_review_contexts, propose_ai_plan, resolve_model
from .ai_provider import AIProviderError, OllamaProvider
from .benchmark import run_ai_benchmark
from .engine import profile_dataset
from .evaluation import run_offline_evaluation
from .golden_validation import run_golden_validation
from .guided import (
    GuidedSession,
    GuidedStage,
    create_guided_session,
    finalize_guided_session,
    load_guided_session,
    prepare_guided_plan,
    record_confirmed_schema,
    record_reviewed_plan,
    save_guided_session,
)
from .io import load_dataset
from .issue_summary import summarize_issues, visible_issues
from .mapping_safety import (
    approval_mapping_conflicts,
    atomize_category_mappings,
    review_mapping_conflicts,
)
from .planner import file_sha256, propose_actions
from .plan_amendments import (
    amend_action_decision,
    amend_mapping_target,
    promote_deferred_candidate,
    reject_deferred_candidate,
    reset_action_for_backtracking,
    reset_deferred_for_backtracking,
)
from .preview_overlap import overlapping_previews
from .recipe import load_recipe, recipe_to_plan
from .review import (
    action_target,
    begin_review_session,
    finish_review_session,
    group_actions,
    record_review_event,
)
from .schema_draft import (
    SchemaSuggestionDecision,
    SchemaSuggestionKind,
    build_schema_draft,
    load_schema_draft,
    materialize_study_schema,
    write_schema_draft,
    write_study_schema,
)
from .study_schema import load_study_schema
from .verification import verify_run
from .transforms import (
    apply_plan,
    preview_action,
    undo_changeset,
    write_cleaned_dataset,
    write_recipe,
)

app = typer.Typer(
    name="data-cleaner",
    help="Local-first data profiling and cleaning assistant.",
    no_args_is_help=True,
)
console = Console()


def _safe_console_text(
    preferred: str,
    fallback: str,
    *,
    encoding: str | None = None,
) -> str:
    """Return preferred text when the active console encoding can represent it."""
    active_encoding = encoding or getattr(console.file, "encoding", None) or "utf-8"
    try:
        preferred.encode(active_encoding)
    except (LookupError, UnicodeEncodeError):
        return fallback
    return preferred


ARROW = _safe_console_text("→", "->")


@app.callback()
def main() -> None:
    """Data Cleaning Assistant command-line interface."""


def _print_summary(report, *, all_issues: bool = False) -> None:
    console.print(f"\n[bold]Source:[/bold] {report.source_path}")
    console.print(f"[bold]Type:[/bold] {report.source_type}")

    table = Table(title="Dataset profile")
    table.add_column("Table / sheet")
    table.add_column("Rows", justify="right")
    table.add_column("Columns", justify="right")
    table.add_column("Grain")
    table.add_column("Inferred key")
    table.add_column("Issues", justify="right")
    for profile in report.tables:
        issue_count = sum(1 for issue in report.issues if issue.table == profile.name)
        grain = profile.grain.label if profile.grain else "—"
        key = ", ".join(profile.candidate_keys[0].columns) if profile.candidate_keys else "—"
        table.add_row(
            profile.name,
            f"{profile.row_count:,}",
            str(profile.column_count),
            grain,
            key,
            str(issue_count),
        )
    console.print(table)

    blank_fields = {
        profile.name: sum(
            col.row_count > 0 and col.null_count == col.row_count
            for col in profile.columns
        )
        for profile in report.tables
    }
    if any(blank_fields.values()):
        console.print(
            "[bold]Already-empty fields:[/bold] "
            + ", ".join(
                f"{name}: {count}"
                for name, count in blank_fields.items()
                if count
            )
            + ". Counted in column null coverage, not proposed as blank-to-null edits."
        )

    if report.relationships:
        rel_table = Table(title="Cross-table relationships")
        rel_table.add_column("Left")
        rel_table.add_column("Right")
        rel_table.add_column("Relationship")
        rel_table.add_column("Shared keys", justify="right")
        rel_table.add_column("Basis")
        rel_table.add_column("Confidence", justify="right")
        for rel in report.relationships:
            rel_table.add_row(
                f"{rel.left_table}.{rel.left_column}",
                f"{rel.right_table}.{rel.right_column}",
                rel.relationship,
                str(rel.shared_value_count),
                rel.source,
                "—" if rel.source == "declared" else f"{rel.confidence:.0%}",
            )
        console.print(rel_table)

    if report.issues:
        summary = summarize_issues(report)
        console.print(
            f"\\n[bold]QA findings:[/bold] {summary['total_issues']} total; "
            f"{sum(t['coverage_noise_issues'] for t in summary['by_table'].values())} "
            "complete-coverage missing/constant findings grouped below."
        )
        if not all_issues:
            coverage_table = Table(title="Coverage-only findings (details retained in JSON)")
            coverage_table.add_column("Table")
            coverage_table.add_column("Entirely flagged columns")
            coverage_table.add_column("Finding type")
            for table_name, statistics in summary["by_table"].items():
                for kind, columns in statistics["coverage_noise_columns"].items():
                    coverage_table.add_row(table_name, str(len(columns)), kind)
            if coverage_table.row_count:
                console.print(coverage_table)
        unscoped = [
            issue for issue in report.issues
            if issue.type == "unscoped_numeric_measurement"
        ]
        for issue in unscoped:
            console.print(
                "[yellow]Numeric QA scope check skipped:[/yellow] "
                f"{issue.table}.{issue.column}. "
                "Mixed assay/unit metadata; configure schema numeric_groups "
                "before interpreting outliers."
            )
        issue_table = Table(
            title="All QA findings" if all_issues else "Remaining QA findings (incl. informational)"
        )
        issue_table.add_column("Severity")
        issue_table.add_column("Type")
        issue_table.add_column("Table")
        issue_table.add_column("Column")
        issue_table.add_column("Affected", justify="right")
        issue_table.add_column("Description")
        for issue in report.issues if all_issues else visible_issues(report):
            issue_table.add_row(
                issue.severity.value,
                issue.type,
                issue.table,
                issue.column or "—",
                str(issue.affected_count),
                issue.title,
            )
        console.print(issue_table)
        if not all_issues:
            console.print(
                "[dim]Complete-coverage findings are grouped for display only; "
                "the --output JSON retains every original finding. "
                "Use --all-issues to show them here.[/dim]"
            )
    else:
        console.print("[green]No issues detected by the current rule set.[/green]")


def _friendly_column_name(column: str | None) -> str:
    if column is None:
        return "(whole table)"
    if column.startswith("...") and column[3:].isdigit():
        return f"Unnamed column {column[3:]}"
    return column


def _friendly_target(action, *, include_table: bool = False) -> str:
    column = _friendly_column_name(action.column)
    if include_table:
        return action.table if action.column is None else f"{action.table} · {column}"
    return action.table if action.column is None else column


def _compact_text(text: str, limit: int = 360) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _human_action_name(action_type: str, parameters: dict[str, object]) -> str:
    if action_type == "normalize_missing":
        values = [repr(value) for value in parameters.get("values", [])]
        if values:
            return f"Treat {', '.join(values)} as missing"
        return "Convert missing-value placeholders to true missing values"
    if action_type == "trim_whitespace":
        return "Remove spaces at the beginning or end of text"
    if action_type == "standardize_dates":
        return "Standardize recognized dates to YYYY-MM-DD"
    if action_type == "coerce_numeric":
        return "Convert numeric-looking text to numbers"
    if action_type == "normalize_categories":
        mapping = parameters.get("mapping", {})
        if isinstance(mapping, dict) and len(mapping) == 1:
            old, new = next(iter(mapping.items()))
            return f"Standardize {old!r} {ARROW} {new!r}"
        return "Standardize equivalent category labels"
    if action_type == "remove_exact_duplicates":
        return "Remove later copies of fully identical rows"
    return action_type.replace("_", " ").capitalize()


def _review_caution(action_type: str) -> str:
    if action_type == "normalize_missing":
        return (
            "Approve only if the listed text is a missing-value marker in every included "
            "column, not a legitimate category or abbreviation."
        )
    if action_type == "normalize_categories":
        return (
            "Approve only if the source and replacement truly mean the same category. "
            "Do not merge different versions, stages, codes, doses, or scientific labels."
        )
    if action_type == "standardize_dates":
        return (
            "Check the examples for date-order ambiguity. Unparseable values stay unchanged, "
            "but a mixed regional date convention should be reviewed carefully."
        )
    if action_type == "coerce_numeric":
        return (
            "Check that these fields are measurements/counts rather than identifiers or codes. "
            "Values that cannot be parsed as numbers stay unchanged."
        )
    if action_type == "trim_whitespace":
        return (
            "Usually low risk, but inspect if leading or trailing spaces could be meaningful "
            "fixed-width formatting."
        )
    if action_type == "remove_exact_duplicates":
        return (
            "Identical rows can still represent legitimate repeated observations. "
            "Inspect before approving row deletion."
        )
    return "Inspect the examples and make sure the change matches the meaning of the data."


def _snapshot_text(snapshot) -> str:
    if snapshot.kind == "null":
        return "missing (null)"
    return repr(snapshot.value)


def _print_plan(plan: CleaningPlan) -> None:
    if plan.planning_notes:
        omitted: dict[str, set[str]] = {}
        for note in plan.planning_notes:
            if note.reason == "structural_only_column":
                omitted.setdefault(note.table, set()).add(note.column)
        if omitted:
            total = sum(len(columns) for columns in omitted.values())
            console.print(
                f"[bold]Non-executable coverage QA:[/bold] {total} structural-only "
                "column(s) omitted from routine cleaning actions."
            )

    console.print(
        f"\n[bold]Cleaning plan:[/bold] {len(plan.actions)} proposed action(s). "
        "Nothing is changed until you review and later apply approved actions."
    )
    if plan.actions:
        counts: dict[str, int] = {}
        for action in plan.actions:
            label = _human_action_name(action.type.value, action.parameters)
            counts[label] = counts.get(label, 0) + 1

        summary = Table(title="What the plan proposes")
        summary.add_column("Change")
        summary.add_column("Actions", justify="right")
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            summary.add_row(label, str(count))
        console.print(summary)

        sample_table = Table(title="Examples of proposed targets")
        sample_table.add_column("Target")
        sample_table.add_column("Proposed change")
        for action in plan.actions[:12]:
            sample_table.add_row(
                _friendly_target(action, include_table=len({a.table for a in plan.actions}) > 1),
                _human_action_name(action.type.value, action.parameters),
            )
        console.print(sample_table)
        if len(plan.actions) > 12:
            console.print(
                f"[dim]{len(plan.actions) - 12} more action(s) are saved in the plan. "
                "Use review-grouped to review them with examples before deciding.[/dim]"
            )

    if plan.deferred_reviews:
        console.print(
            f"\n[bold]Needs human judgment:[/bold] {len(plan.deferred_reviews)} "
            "AI-reviewed candidate(s) were intentionally left non-executable."
        )
        deferred_table = Table(title="Examples needing judgment")
        deferred_table.add_column("Target")
        deferred_table.add_column("Possible change")
        deferred_table.add_column("Why not automatic")
        for deferred in plan.deferred_reviews[:8]:
            action = deferred.proposed_action
            deferred_table.add_row(
                _friendly_target(action, include_table=False),
                _human_action_name(action.type.value, action.parameters),
                _compact_text(deferred.rationale, 180),
            )
        console.print(deferred_table)
        if len(plan.deferred_reviews) > 8:
            console.print(
                f"[dim]{len(plan.deferred_reviews) - 8} more deferred candidate(s) are "
                "preserved in the plan and will appear during review.[/dim]"
            )

    console.print("\n[dim]No edits have been applied.[/dim]")


def _print_action_preview(tables, action, sample_limit: int = 8) -> None:
    preview = preview_action(tables, action, sample_limit=sample_limit)

    if preview.sample_rows:
        table = Table(
            title=f"Examples of rows that would be removed — "
            f"{preview.affected_count:,} row(s) total"
        )
        table.add_column("Row", justify="right")
        table.add_column("Current row values")
        for snapshot in preview.sample_rows:
            values = {
                _friendly_column_name(column): value.value
                for column, value in snapshot.values.items()
            }
            table.add_row(str(snapshot.row_position), _compact_text(repr(values), 300))
        console.print(table)
        return

    table = Table(
        title=f"Before {ARROW} after examples — {preview.affected_count:,} cell(s) total"
    )
    table.add_column("Row", justify="right")
    table.add_column("Before")
    table.add_column("After")
    for change in preview.sample_changes:
        table.add_row(
            str(change.row_index),
            _snapshot_text(change.old_value),
            _snapshot_text(change.new_value),
        )
    console.print(table)


def _print_overlap_warning(overlaps, action_id: str) -> None:
    related = overlaps.get(action_id, [])
    if not related:
        return
    summary = ", ".join(f"{other} ({count} shared row(s))" for other, count in related)
    console.print(
        "[yellow]Independent-preview overlap:[/yellow] "
        + summary
        + ". These previews all use the original data; sequential execution may "
        "turn a later action into a no-op or change its effective count."
    )


def _print_mapping_conflicts(conflicts, *, title: str = "Canonical mapping conflicts") -> None:
    if not conflicts:
        return
    table = Table(title=title)
    table.add_column("Target")
    table.add_column("Source value")
    table.add_column("Competing proposed targets")
    for conflict in conflicts:
        target = (
            f"{conflict.table}.{_friendly_column_name(conflict.column)}"
            if conflict.column is not None
            else conflict.table
        )
        table.add_row(
            target,
            repr(conflict.source),
            ", ".join(repr(value) for value in conflict.targets),
        )
    console.print(table)


def _print_drift(compatibility) -> None:
    if not compatibility.drift:
        return
    table = Table(title="Schema drift since recipe creation")
    table.add_column("Severity")
    table.add_column("Kind")
    table.add_column("Table")
    table.add_column("Column")
    table.add_column("Details")
    for drift in compatibility.drift:
        table.add_row(
            drift.severity,
            drift.kind,
            drift.table,
            drift.column or "—",
            drift.message,
        )
    console.print(table)


def _load_plan(path: Path) -> CleaningPlan:
    return CleaningPlan.model_validate_json(path.read_text(encoding="utf-8"))


def _review_context_payload(context, review_contexts) -> str:
    return json.dumps(
        {
            "format_version": 1,
            "note": (
                "aggregate_context is the privacy-minimized superset. "
                "review_batches are the exact focused contexts sent independently "
                "to the model; adaptive consensus reuses each batch context."
            ),
            "aggregate_context": context.model_dump(mode="json"),
            "review_batches": [
                item.model_dump(mode="json")
                for item in review_contexts
            ],
        },
        indent=2,
        default=str,
    )


def _write_cleaning_outputs(
    output: Path,
    plan: CleaningPlan,
    changeset: ChangeSet,
    *,
    audit: Path | None = None,
    recipe: Path | None = None,
) -> None:
    changeset.result_path = str(output)
    changeset.result_sha256 = file_sha256(output)

    audit_path = (
        audit.expanduser().resolve()
        if audit
        else output.with_name(f"{output.name}.audit.json")
    )
    recipe_path = (
        recipe.expanduser().resolve()
        if recipe
        else output.with_name(f"{output.name}.recipe.yaml")
    )
    audit_path.write_text(changeset.model_dump_json(indent=2), encoding="utf-8")
    write_recipe(plan, changeset, recipe_path)

    console.print(f"[green]Cleaned dataset:[/green] {output}")
    console.print(f"[green]Audit log:[/green] {audit_path}")
    console.print(f"[green]Recipe:[/green] {recipe_path}")
    console.print(
        f"[bold]{changeset.affected_cell_count:,}[/bold] cell change(s), "
        f"[bold]{changeset.affected_row_count:,}[/bold] row deletion(s), "
        f"across [bold]{len(changeset.executions)}[/bold] action(s)."
    )


@app.command()
def profile(
    input_path: Annotated[Path, typer.Argument(help="CSV, XLSX/XLSM, or Parquet file to profile")],
    all_issues: Annotated[
        bool,
        typer.Option("--all-issues", help="Show every issue instead of grouping coverage noise"),
    ] = False,
    schema_path: Annotated[
        Path | None,
        typer.Option("--schema", help="Study-schema YAML/JSON for read-only validation"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write the full JSON report to this path"),
    ] = None,
) -> None:
    """Profile a dataset without changing the source file."""
    schema = load_study_schema(schema_path) if schema_path else None
    report = profile_dataset(input_path, schema=schema)
    _print_summary(report, all_issues=all_issues)

    if output:
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
        console.print(f"\n[green]Report written to:[/green] {output}")


def _schema_suggestion_text(suggestion) -> str:
    proposed = suggestion.proposed
    if suggestion.kind == SchemaSuggestionKind.GRAIN_KEY:
        return (
            f"Use {proposed.get('columns', [])!r} as the row grain key "
            f"for {suggestion.table!r}"
        )
    if suggestion.kind == SchemaSuggestionKind.CANONICAL_MAPPING:
        return (
            f"Treat {proposed.get('source')!r} as an alias of "
            f"{proposed.get('target')!r} in "
            f"{suggestion.table}.{suggestion.column}"
        )
    if suggestion.kind == SchemaSuggestionKind.MISSING_TOKEN:
        return (
            f"Treat {proposed.get('token')!r} as missing in "
            f"{suggestion.table}.{suggestion.column}"
        )
    if suggestion.kind == SchemaSuggestionKind.FOREIGN_KEY:
        return (
            f"Declare {proposed.get('child_table')}.{proposed.get('child_column')} "
            f"{ARROW} {proposed.get('parent_table')}.{proposed.get('parent_column')}"
        )
    return suggestion.kind.value


@app.command("schema-draft")
def schema_draft_command(
    input_path: Annotated[Path, typer.Argument(help="Dataset to inspect")],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Non-authoritative schema draft YAML",
        ),
    ] = Path("study_schema_draft.yaml"),
) -> None:
    """Propose a non-authoritative study-schema draft from deterministic evidence."""
    draft = build_schema_draft(input_path)
    resolved = write_schema_draft(draft, output)

    counts: dict[str, int] = {}
    for suggestion in draft.suggestions:
        counts[suggestion.kind.value] = counts.get(suggestion.kind.value, 0) + 1

    console.print(
        f"[green]Schema draft written to:[/green] {resolved}"
    )
    console.print(
        f"Proposed [bold]{len(draft.suggestions)}[/bold] item(s). "
        "Nothing in this draft is authoritative until explicitly reviewed."
    )
    if counts:
        table = Table(title="Draft suggestions")
        table.add_column("Kind")
        table.add_column("Count", justify="right")
        for kind, count in sorted(counts.items()):
            table.add_row(kind.replace("_", " "), str(count))
        console.print(table)
    console.print(
        "[dim]Next: schema-review DATASET DRAFT -o study_schema.yaml[/dim]"
    )


@app.command("schema-review")
def schema_review_command(
    input_path: Annotated[Path, typer.Argument(help="Original dataset")],
    draft_path: Annotated[Path, typer.Argument(help="Schema draft YAML")],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Confirmed StudySchema YAML",
        ),
    ] = Path("study_schema.yaml"),
    reviewed_draft: Annotated[
        Path | None,
        typer.Option(
            "--reviewed-draft",
            help="Optional path for the reviewed draft/provenance YAML",
        ),
    ] = None,
    unresolved_only: Annotated[
        bool,
        typer.Option(
            "--unresolved-only",
            help="Review only suggestions still marked proposed",
        ),
    ] = False,
) -> None:
    """Review draft suggestions and compile only approvals into a real schema."""
    draft = load_schema_draft(draft_path)
    current_hash = file_sha256(input_path)
    if current_hash != draft.source_sha256:
        raise typer.BadParameter(
            "The dataset changed after this schema draft was created. "
            "Generate a new schema draft."
        )

    review_items = [
        suggestion
        for suggestion in draft.suggestions
        if not unresolved_only
        or suggestion.decision == SchemaSuggestionDecision.PROPOSED
    ]

    console.print(
        f"[bold]Review schema draft:[/bold] {len(review_items)} suggestion(s)."
    )
    console.print(
        "[dim]Approving a suggestion adds it to the study dictionary; "
        "no dataset values are changed during schema review.[/dim]"
    )

    position = 0
    while position < len(review_items):
        suggestion = review_items[position]
        console.rule(
            f"Schema suggestion {position + 1} of {len(review_items)}"
        )
        console.print(f"[bold]{_schema_suggestion_text(suggestion)}[/bold]")
        console.print(
            f"Confidence: [bold]{suggestion.confidence:.0%}[/bold]"
        )
        console.print(f"Why proposed: {suggestion.rationale}")

        while True:
            decision = typer.prompt(
                "Choose: [a]pprove, [r]eject, [s]decide later, [b]ack, [q]uit",
                default="s",
            ).strip().casefold()

            if decision in {"a", "approve"}:
                suggestion.decision = SchemaSuggestionDecision.APPROVED
                break
            if decision in {"r", "reject"}:
                suggestion.decision = SchemaSuggestionDecision.REJECTED
                break
            if decision in {"s", "skip", "defer"}:
                suggestion.decision = SchemaSuggestionDecision.PROPOSED
                break
            if decision in {"b", "back"}:
                if position == 0:
                    console.print(
                        "[yellow]There is no previous schema suggestion.[/yellow]"
                    )
                    continue
                previous = review_items[position - 1]
                previous.decision = SchemaSuggestionDecision.PROPOSED
                console.print(
                    "[cyan]Previous schema decision reset.[/cyan] "
                    "Returning to that suggestion."
                )
                position -= 1
                break
            if decision in {"q", "quit"}:
                reviewed_path = (
                    reviewed_draft.expanduser().resolve()
                    if reviewed_draft
                    else output.expanduser().resolve().with_name(
                        f"{output.stem}.reviewed-draft.yaml"
                    )
                )
                write_schema_draft(draft, reviewed_path)
                console.print(
                    f"[yellow]Partial schema review saved to:[/yellow] "
                    f"{reviewed_path}"
                )
                return
            console.print("[red]Enter a, r, s, b, or q.[/red]")

        if decision in {"b", "back"}:
            continue
        position += 1

    try:
        schema = materialize_study_schema(
            draft,
            dataset_path=input_path,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    schema_path = write_study_schema(schema, output)
    reviewed_path = (
        reviewed_draft.expanduser().resolve()
        if reviewed_draft
        else schema_path.with_name(f"{schema_path.stem}.reviewed-draft.yaml")
    )
    write_schema_draft(draft, reviewed_path)

    approved = sum(
        item.decision == SchemaSuggestionDecision.APPROVED
        for item in draft.suggestions
    )
    rejected = sum(
        item.decision == SchemaSuggestionDecision.REJECTED
        for item in draft.suggestions
    )
    proposed = len(draft.suggestions) - approved - rejected

    console.print(f"[green]Confirmed study schema:[/green] {schema_path}")
    console.print(f"[green]Reviewed draft:[/green] {reviewed_path}")
    console.print(
        f"Approved: [bold]{approved}[/bold] · "
        f"Rejected: [bold]{rejected}[/bold] · "
        f"Unresolved: [bold]{proposed}[/bold]"
    )


@app.command("validate-golden")
def validate_golden_command(
    source_path: Annotated[
        Path,
        typer.Argument(help="Dirty/source dataset to evaluate"),
    ],
    expected_path: Annotated[
        Path,
        typer.Argument(help="Human-approved expected cleaned dataset"),
    ],
    json_output: Annotated[
        Path | None,
        typer.Option(
            "--json-output",
            help="Optional machine-readable golden validation report",
        ),
    ] = None,
    difference_limit: Annotated[
        int,
        typer.Option(
            "--difference-limit",
            min=1,
            max=500,
            help="Maximum mismatches retained in the report",
        ),
    ] = 50,
) -> None:
    """Compare all deterministic cleaner proposals with a golden cleaned output."""
    try:
        report = run_golden_validation(
            source_path,
            expected_path,
            difference_limit=difference_limit,
        )
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    metrics = report.metrics
    table = Table(title="Golden-pair validation")
    table.add_column("Metric")
    table.add_column("Result", justify="right")
    table.add_row("Planned actions", str(metrics.planned_actions))
    table.add_row("Executed actions", str(metrics.executed_actions))
    table.add_row("Expected changed cells", f"{metrics.expected_changed_cells:,}")
    table.add_row("Actual changed cells", f"{metrics.actual_changed_cells:,}")
    table.add_row(
        "Matched expected changes",
        f"{metrics.matched_expected_changes:,}",
    )
    table.add_row("Missed expected changes", f"{metrics.missed_expected_changes:,}")
    table.add_row("Unexpected changes", f"{metrics.unexpected_changes:,}")
    table.add_row("Wrong replacements", f"{metrics.wrong_replacements:,}")
    table.add_row(
        "Remaining different cells",
        f"{metrics.remaining_different_cells:,}",
    )
    table.add_row("Structural mismatches", str(metrics.structural_mismatches))
    console.print(table)

    if report.planned_action_summaries:
        console.print(
            "[bold]Proposed action surface:[/bold] "
            + ", ".join(report.planned_action_summaries)
        )

    if report.differences:
        diff_table = Table(title="First golden mismatches")
        diff_table.add_column("Kind")
        diff_table.add_column("Location")
        diff_table.add_column("Source")
        diff_table.add_column("Expected")
        diff_table.add_column("Actual")
        for item in report.differences:
            location = item.table
            if item.row_position is not None:
                location += f" row {item.row_position}"
            if item.column is not None:
                location += f" · {item.column}"
            diff_table.add_row(
                item.kind,
                location,
                repr(item.source),
                repr(item.expected),
                repr(item.actual),
            )
        console.print(diff_table)

    if json_output is not None:
        output = json_output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[green]Golden report:[/green] {output}")

    if report.passed:
        console.print("\n[bold green]GOLDEN VALIDATION PASS[/bold green]")
        return

    console.print("\n[bold red]GOLDEN VALIDATION FAIL[/bold red]")
    raise typer.Exit(code=2)


@app.command("evaluate-offline")
def evaluate_offline_command(
    json_output: Annotated[
        Path | None,
        typer.Option(
            "--json-output",
            help="Optional machine-readable evaluation report",
        ),
    ] = None,
) -> None:
    """Run the built-in deterministic/generalization scorecard without Ollama."""
    report = run_offline_evaluation()

    summary = Table(title="Offline generalization scorecard")
    summary.add_column("Metric")
    summary.add_column("Result", justify="right")
    summary.add_row(
        "Benchmark cases",
        f"{report.benchmark_passed_cases}/{report.benchmark_case_count} passed",
    )
    summary.add_row(
        "Required-effect recall",
        f"{report.required_hits}/{report.required_total} "
        f"({report.required_recall:.1%})",
    )
    summary.add_row(
        "Unsafe executable proposals",
        str(report.unsafe_executable_proposals),
    )
    summary.add_row(
        "Prohibited candidate exposures",
        str(report.prohibited_candidate_exposures),
    )
    summary.add_row(
        "Deterministic review actions",
        str(report.deterministic_actions),
    )
    summary.add_row(
        "Semantic review candidates",
        str(report.semantic_review_candidates),
    )
    summary.add_row(
        "Unmatched deterministic review actions",
        str(report.unmatched_deterministic_actions),
    )
    console.print(summary)

    suite_table = Table(title="Coverage by suite")
    suite_table.add_column("Suite")
    suite_table.add_column("Cases", justify="right")
    suite_table.add_column("Required effects", justify="right")
    suite_table.add_column("Unsafe executable", justify="right")
    suite_table.add_column("Prohibited exposure", justify="right")
    for suite_summary in report.suite_summaries:
        suite_table.add_row(
            suite_summary.suite,
            f"{suite_summary.passed_cases}/{suite_summary.case_count}",
            f"{suite_summary.required_hits}/{suite_summary.required_total}",
            str(suite_summary.unsafe_executable_proposals),
            str(suite_summary.prohibited_candidate_exposures),
        )
    console.print(suite_table)

    semantic = report.semantic_coverage
    semantic_table = Table(title="Semantic detector coverage (no model decisions)")
    semantic_table.add_column("Category")
    semantic_table.add_column("Exposed", justify="right")
    semantic_table.add_column("Rate", justify="right")
    semantic_table.add_row(
        "Expected-safe corrections",
        f"{semantic.safe_exposed}/{semantic.safe_cases}",
        f"{semantic.safe_coverage:.1%}",
    )
    semantic_table.add_row(
        "Expected-risky pairs",
        f"{semantic.risky_exposed}/{semantic.risky_cases}",
        f"{semantic.risky_exposure_rate:.1%}",
    )
    console.print(semantic_table)
    console.print(
        "[dim]Risky-pair exposure means the pair reached semantic review; "
        "it does not mean the cleaner merged or approved it.[/dim]"
    )

    failures = [item for item in report.cases if not item.passed]
    if failures:
        failure_table = Table(title="Benchmark cases needing attention")
        failure_table.add_column("Suite")
        failure_table.add_column("Case")
        failure_table.add_column("Reason")
        for item in failures:
            reason = " ".join(item.notes) or "Benchmark gate failed."
            failure_table.add_row(item.suite, item.case_id, reason)
        console.print(failure_table)

    if json_output is not None:
        output = json_output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[green]Evaluation report:[/green] {output}")

    safety = "[green]PASS[/green]" if report.safety_pass else "[red]FAIL[/red]"
    coverage = "[green]PASS[/green]" if report.coverage_pass else "[red]FAIL[/red]"
    console.print(
        f"\nSafety gate: {safety} · Required-coverage gate: {coverage}"
    )
    if report.overall_pass:
        console.print("[bold green]OFFLINE EVALUATION PASS[/bold green]")
        return

    console.print("[bold red]OFFLINE EVALUATION FAIL[/bold red]")
    raise typer.Exit(code=2)


@app.command("ai-status")
def ai_status_command(
    url: Annotated[
        str | None,
        typer.Option("--url", help="Ollama base URL; defaults to OLLAMA_HOST or localhost:11434"),
    ] = None,
) -> None:
    """Check the local Ollama connection and list installed models."""
    provider = OllamaProvider(url)
    try:
        models = provider.list_models()
    except AIProviderError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    console.print(f"[green]Ollama reachable:[/green] {provider.base_url}")
    if not models:
        console.print(
            "[yellow]No local models are installed.[/yellow] "
            "Install/pull a model in Ollama before using ai-plan."
        )
        return

    table = Table(title="Installed Ollama models")
    table.add_column("Model")
    for model in models:
        table.add_row(model)
    console.print(table)


@app.command("ai-context")
def ai_context_command(
    input_path: Annotated[Path, typer.Argument(help="Dataset to profile for AI planning")],
    schema_path: Annotated[
        Path | None,
        typer.Option("--schema", help="Study-schema YAML/JSON"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write the exact AI context JSON to this path"),
    ] = None,
) -> None:
    """Show exactly what compact context would be sent to the AI planner."""
    schema = load_study_schema(schema_path) if schema_path else None
    loaded = load_dataset(input_path)
    report = profile_dataset(input_path, schema=schema)
    candidates = semantic_review_candidates(
        build_ai_candidates(report, loaded.tables, schema=schema)
    )
    context = build_ai_context(report, candidates)
    review_contexts = build_review_contexts(context, candidates)
    payload = _review_context_payload(context, review_contexts)

    if output:
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        console.print(f"[green]AI context written to:[/green] {output}")
    else:
        console.print(payload)


@app.command("ai-plan")
def ai_plan_command(
    input_path: Annotated[Path, typer.Argument(help="Dataset to inspect with local AI")],
    schema_path: Annotated[
        Path | None,
        typer.Option("--schema", help="Study-schema YAML/JSON"),
    ] = None,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Path for the AI-assisted cleaning plan JSON"),
    ] = Path("ai_cleaning_plan.json"),
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Installed Ollama model to use"),
    ] = None,
    url: Annotated[
        str | None,
        typer.Option("--url", help="Ollama base URL; defaults to OLLAMA_HOST or localhost:11434"),
    ] = None,
    think: Annotated[
        bool,
        typer.Option(
            "--think/--no-think",
            help="Enable Ollama thinking mode; disabled by default for structured planning",
        ),
    ] = False,
    rules: Annotated[
        bool,
        typer.Option(
            "--rules/--no-rules",
            help="Include deterministic rule proposals alongside validated AI suggestions",
        ),
    ] = True,
    context_out: Annotated[
        Path | None,
        typer.Option("--context-out", help="Save the exact context sent to the model"),
    ] = None,
) -> None:
    """Generate a reviewable local-AI-assisted plan without modifying data."""
    provider = OllamaProvider(url, think=think)
    try:
        schema = load_study_schema(schema_path) if schema_path else None
        selected_model = resolve_model(provider, model)
        result = propose_ai_plan(
            input_path,
            provider=provider,
            model=selected_model,
            include_rules=rules,
            schema=schema,
        )
    except (AIProviderError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    console.print(
        f"[green]AI planner:[/green] {result.provider} / {result.model}"
    )
    if result.review_call_count:
        console.print(
            "[bold]Adaptive review:[/bold] "
            f"{result.review_call_count} model call(s); "
            f"{result.consensus_disagreement_count} candidate disagreement(s)"
        )
    if result.covered_ai_action_count:
        console.print(
            "[bold]AI actions already covered by deterministic plan:[/bold] "
            f"{result.covered_ai_action_count} omitted from duplicate execution"
        )
    console.print(f"[bold]Model summary:[/bold] {result.response.summary}")

    if result.rejected:
        rejected_table = Table(title="AI proposals rejected by deterministic safety validation")
        rejected_table.add_column("Type")
        rejected_table.add_column("Target")
        rejected_table.add_column("Reason")
        for rejected in result.rejected:
            target = (
                f"{rejected.table}.{rejected.column}"
                if rejected.column is not None
                else rejected.table
            )
            rejected_table.add_row(
                rejected.type.value if rejected.type else "unknown",
                target or rejected.candidate_id,
                rejected.reason,
            )
        console.print(rejected_table)

    _print_plan(result.plan)

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.plan.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"\n[green]AI-assisted plan written to:[/green] {output}")

    if context_out:
        context_out = context_out.expanduser().resolve()
        context_out.parent.mkdir(parents=True, exist_ok=True)
        context_out.write_text(
            _review_context_payload(result.context, result.review_contexts),
            encoding="utf-8",
        )
        console.print(f"[green]AI context written to:[/green] {context_out}")


@app.command("ai-benchmark")
def ai_benchmark_command(
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Installed Ollama model to benchmark"),
    ] = None,
    runs: Annotated[
        int,
        typer.Option(
            "--runs",
            min=1,
            max=10,
            help="Repeat each synthetic case this many times",
        ),
    ] = 1,
    suite: Annotated[
        str,
        typer.Option("--suite", help="Benchmark suite: core, research, general, or all"),
    ] = "core",
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Write detailed benchmark JSON here"),
    ] = Path("ai_benchmark.json"),
    url: Annotated[
        str | None,
        typer.Option("--url", help="Ollama base URL; defaults to OLLAMA_HOST or localhost:11434"),
    ] = None,
    think: Annotated[
        bool,
        typer.Option(
            "--think/--no-think",
            help="Enable Ollama thinking mode during the benchmark",
        ),
    ] = False,
) -> None:
    """Benchmark a local model on synthetic cleaning cases with hidden answer keys."""
    provider = OllamaProvider(url, think=think)
    normalized_suite = suite.strip().casefold()
    if normalized_suite not in {"core", "research", "general", "all"}:
        raise typer.BadParameter(
            "--suite must be one of: core, research, general, all."
        )
    try:
        selected_model = resolve_model(provider, model)
    except (AIProviderError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    console.print(
        f"[green]Benchmarking:[/green] {provider.name} / {selected_model} "
        f"(suite={normalized_suite}; {runs} run(s) per case; "
        f"thinking={'on' if think else 'off'})"
    )
    summary = run_ai_benchmark(
        provider=provider,
        model=selected_model,
        runs_per_case=runs,
        suite=normalized_suite,
    )

    case_table = Table(title=f"AI cleaning benchmark — {summary.suite}")
    case_table.add_column("Case")
    case_table.add_column("Run", justify="right")
    case_table.add_column("Pass")
    case_table.add_column("Required", justify="right")
    case_table.add_column("Accepted", justify="right")
    case_table.add_column("Unsafe accepted", justify="right")
    case_table.add_column("Validator rejects", justify="right")
    case_table.add_column("Seconds", justify="right")
    for result in summary.results:
        case_table.add_row(
            result.case_id,
            str(result.run_index),
            "yes" if result.passed else "NO",
            f"{result.required_hits}/{result.required_total}",
            str(result.accepted_actions),
            str(result.forbidden_accepted),
            str(result.validator_rejections),
            f"{result.latency_seconds:.2f}",
        )
    console.print(case_table)

    failed = [result for result in summary.results if not result.passed]
    if failed:
        detail_table = Table(title="Failed-case diagnostics")
        detail_table.add_column("Case")
        detail_table.add_column("Model summary")
        detail_table.add_column("Accepted")
        detail_table.add_column("Rejected by validator")
        for result in failed:
            accepted_text = "; ".join(
                f"{item.get('type')}:{item.get('column') or 'table'} {item.get('parameters')}"
                for item in result.accepted
            ) or "—"
            rejected_text = "; ".join(
                f"{item.get('type')}:{item.get('column') or 'table'} — {item.get('reason')}"
                for item in result.rejected
            ) or "—"
            detail_table.add_row(
                result.case_id,
                result.model_summary or result.error or "—",
                accepted_text,
                rejected_text,
            )
        console.print(detail_table)

    metrics = Table(title="Benchmark summary")
    metrics.add_column("Metric")
    metrics.add_column("Value", justify="right")
    metrics.add_row("Case-run pass rate", f"{summary.pass_rate:.1%}")
    metrics.add_row("Required-action recall", f"{summary.required_recall:.1%}")
    metrics.add_row("Accepted-action precision", f"{summary.accepted_precision:.1%}")
    metrics.add_row("Forbidden proposals", str(summary.forbidden_raw_proposals))
    metrics.add_row("Forbidden accepted", str(summary.forbidden_accepted))
    metrics.add_row("Unsafe candidates exposed", str(summary.unsafe_candidates_exposed))
    if summary.forbidden_validator_catch_rate is not None:
        metrics.add_row(
            "Validator catch rate (forbidden)",
            f"{summary.forbidden_validator_catch_rate:.1%}",
        )
    metrics.add_row("All validator rejections", str(summary.validator_rejections))
    metrics.add_row("Mean latency / case", f"{summary.mean_latency_seconds:.2f}s")
    console.print(metrics)

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"[green]Detailed benchmark written to:[/green] {output}")


@app.command("plan")
def plan_command(
    input_path: Annotated[Path, typer.Argument(help="Dataset to inspect")],
    schema_path: Annotated[
        Path | None,
        typer.Option("--schema", help="Study-schema YAML/JSON"),
    ] = None,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Path for the proposed cleaning plan JSON"),
    ] = Path("cleaning_plan.json"),
) -> None:
    """Generate reviewable cleaning actions without changing the dataset."""
    schema = load_study_schema(schema_path) if schema_path else None
    plan = propose_actions(input_path, schema=schema)
    _print_plan(plan)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"\n[green]Plan written to:[/green] {output}")


@app.command("preview")
def preview_command(
    input_path: Annotated[Path, typer.Argument(help="Original dataset")],
    plan_path: Annotated[Path, typer.Argument(help="Cleaning plan JSON")],
    sample: Annotated[
        int,
        typer.Option("--sample", min=1, max=50, help="Maximum changes shown per action"),
    ] = 8,
) -> None:
    """Preview proposed changes without modifying the dataset."""
    plan = _load_plan(plan_path)
    if file_sha256(input_path) != plan.source_sha256:
        raise typer.BadParameter(
            "The input file changed after the plan was created. Generate a new plan."
        )
    tables = load_dataset(input_path).tables
    _print_plan(plan)
    overlaps = overlapping_previews(tables, plan.actions)
    for action in plan.actions:
        _print_action_preview(tables, action, sample_limit=sample)
        _print_overlap_warning(overlaps, action.id)


@app.command("review")
def review_command(
    input_path: Annotated[Path, typer.Argument(help="Original dataset")],
    plan_path: Annotated[Path, typer.Argument(help="Cleaning plan JSON")],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Write the reviewed plan JSON here"),
    ] = Path("reviewed_plan.json"),
    sample: Annotated[
        int,
        typer.Option("--sample", min=1, max=50, help="Maximum changes shown per action"),
    ] = 5,
) -> None:
    """Interactively approve, reject, or defer each proposed cleaning action."""
    plan = _load_plan(plan_path)
    if file_sha256(input_path) != plan.source_sha256:
        raise typer.BadParameter(
            "The input file changed after the plan was created. Generate a new plan."
        )

    tables = load_dataset(input_path).tables
    split_count, additional_actions = atomize_category_mappings(plan)
    if split_count:
        console.print(
            f"[cyan]Mapping-level review:[/cyan] split [bold]{split_count}[/bold] "
            f"multi-mapping action(s) into [bold]{additional_actions + split_count}[/bold] "
            "independently reviewable mappings."
        )
    overlaps = overlapping_previews(tables, plan.actions)
    for index, action in enumerate(plan.actions, start=1):
        console.rule(f"Action {index}/{len(plan.actions)}")
        target = (
            f"{action.table}.{action.column}"
            if action.column is not None
            else action.table
        )
        console.print(
            f"[bold]{action.type.value}[/bold] — "
            f"{target} ({action.confidence:.0%} confidence)"
        )
        console.print(action.description)
        _print_action_preview(tables, action, sample_limit=sample)
        _print_overlap_warning(overlaps, action.id)

        while True:
            decision = typer.prompt(
                "Decision: [a]pprove, [r]eject, [s]kip/defer, [q]uit",
                default="s",
            ).strip().casefold()
            if decision in {"a", "approve"}:
                conflicts = approval_mapping_conflicts(plan, [action])
                if conflicts:
                    console.print(
                        "[red]Approval blocked:[/red] this mapping conflicts with "
                        "an already approved canonical target."
                    )
                    _print_mapping_conflicts(conflicts, title="Approval conflict")
                    continue
                action.decision = ActionDecision.APPROVED
                break
            if decision in {"r", "reject"}:
                action.decision = ActionDecision.REJECTED
                break
            if decision in {"s", "skip", "defer"}:
                action.decision = ActionDecision.PROPOSED
                break
            if decision in {"q", "quit"}:
                output = output.expanduser().resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
                console.print(f"[yellow]Review stopped. Partial plan saved to:[/yellow] {output}")
                return
            console.print("[red]Enter a, r, s, or q.[/red]")

    pending_deferred = [
        item for item in plan.deferred_reviews if item.status == "pending"
    ]
    deferred_position = 0
    while deferred_position < len(pending_deferred):
        deferred = pending_deferred[deferred_position]
        action = deferred.proposed_action
        console.rule(
            f"Deferred AI review {deferred_position + 1}/{len(pending_deferred)} "
            f"({deferred.candidate_id})"
        )
        console.print(f"[bold]{action.table}.{action.column or '(table)'}[/bold]")
        console.print(deferred.question)
        console.print(f"[yellow]AI deferred:[/yellow] {deferred.rationale}")
        console.print(f"Proposed parameters: {action.parameters!r}")
        _print_action_preview(tables, action, sample_limit=sample)
        backtracked = False

        while True:
            decision = typer.prompt(
                "Manual decision: [a]pprove candidate, [r]eject, "
                "[s]kip, [b]ack, [q]uit",
                default="s",
            ).strip().casefold()
            if decision in {"a", "approve"}:
                conflicts = approval_mapping_conflicts(plan, [action])
                if conflicts:
                    console.print(
                        "[red]Approval blocked:[/red] this candidate conflicts with "
                        "an already approved canonical target."
                    )
                    _print_mapping_conflicts(conflicts, title="Approval conflict")
                    continue
                promote_deferred_candidate(
                    plan,
                    candidate_id=deferred.candidate_id,
                )
                break
            if decision in {"r", "reject"}:
                reject_deferred_candidate(
                    plan,
                    candidate_id=deferred.candidate_id,
                )
                break
            if decision in {"s", "skip", "defer"}:
                break
            if decision in {"b", "back"}:
                if deferred_position == 0:
                    console.print(
                        "[yellow]There is no previous deferred judgment.[/yellow]"
                    )
                    continue
                previous = pending_deferred[deferred_position - 1]
                reset_deferred_for_backtracking(
                    plan,
                    candidate_id=previous.candidate_id,
                )
                console.print(
                    "[cyan]Previous deferred judgment undone.[/cyan] "
                    "Returning to that candidate."
                )
                deferred_position -= 1
                backtracked = True
                break
            if decision in {"q", "quit"}:
                output = output.expanduser().resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
                console.print(f"[yellow]Partial review saved to:[/yellow] {output}")
                return
            console.print("[red]Enter a, r, s, b, or q.[/red]")

        if backtracked:
            continue
        deferred_position += 1

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    approved = sum(a.decision == ActionDecision.APPROVED for a in plan.actions)
    rejected = sum(a.decision == ActionDecision.REJECTED for a in plan.actions)
    deferred = len(plan.actions) - approved - rejected
    console.print(f"[green]Reviewed plan written to:[/green] {output}")
    console.print(
        f"Approved: [bold]{approved}[/bold] · Rejected: [bold]{rejected}[/bold] · "
        f"Deferred: [bold]{deferred}[/bold]"
    )


@app.command("review-grouped")
def review_grouped_command(
    input_path: Annotated[Path, typer.Argument(help="Original dataset")],
    plan_path: Annotated[Path, typer.Argument(help="Cleaning plan JSON")],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Write the reviewed plan JSON here"),
    ] = Path("reviewed_plan.json"),
    sample: Annotated[
        int,
        typer.Option("--sample", min=1, max=50, help="Maximum examples shown per action"),
    ] = 5,
    unresolved_only: Annotated[
        bool,
        typer.Option(
            "--unresolved-only",
            help="Review only actions still marked proposed plus pending deferred judgments",
        ),
    ] = False,
) -> None:
    """Review proposed changes in plain-language groups before anything is applied."""
    plan = _load_plan(plan_path)
    if file_sha256(input_path) != plan.source_sha256:
        raise typer.BadParameter(
            "The input file changed after the plan was created. Generate a new plan."
        )

    tables = load_dataset(input_path).tables
    split_count, additional_actions = atomize_category_mappings(plan)
    families = group_actions(
        tables,
        plan,
        include_decisions=(
            {ActionDecision.PROPOSED}
            if unresolved_only
            else None
        ),
    )
    session_started = begin_review_session(plan, resume=unresolved_only)
    multiple_tables = len(tables) > 1

    def save_and_report(*, partial: bool) -> None:
        finish_review_session(
            plan,
            session_started,
            outcome="partial" if partial else "completed",
        )
        resolved_output = output.expanduser().resolve()
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        resolved_output.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        approved = sum(a.decision == ActionDecision.APPROVED for a in plan.actions)
        rejected = sum(a.decision == ActionDecision.REJECTED for a in plan.actions)
        unresolved = sum(a.decision == ActionDecision.PROPOSED for a in plan.actions)
        label = "Partial review saved to" if partial else "Reviewed plan written to"
        console.print(f"[green]{label}:[/green] {resolved_output}")
        console.print(
            f"Approved: [bold]{approved}[/bold] · "
            f"Rejected: [bold]{rejected}[/bold] · "
            f"Decide later: [bold]{unresolved}[/bold]"
        )
        if plan.review_session and plan.review_session.duration_seconds is not None:
            console.print(
                "Review time: "
                f"[bold]{plan.review_session.duration_seconds:.1f}s[/bold] · "
                f"Recorded decisions: [bold]{len(plan.review_session.events)}[/bold]"
            )
        console.print("[dim]The source dataset was not changed.[/dim]")

    console.print(
        f"[bold]Review proposed cleaning:[/bold] {len(plan.actions)} action(s) grouped into "
        f"{len(families)} set(s) of identical changes."
    )
    if split_count:
        console.print(
            f"[cyan]Mapping-level review:[/cyan] split [bold]{split_count}[/bold] "
            f"multi-mapping action(s) into [bold]{additional_actions + split_count}[/bold] "
            "independently reviewable mappings."
        )
    unresolved_mapping_conflicts = review_mapping_conflicts(plan)
    if unresolved_mapping_conflicts:
        console.print(
            "[yellow]Conflicting canonical targets detected.[/yellow] "
            "These mappings must not both be approved."
        )
        _print_mapping_conflicts(unresolved_mapping_conflicts)
    console.print(
        "[dim]You are only deciding what should be allowed into the reviewed plan. "
        "Nothing is applied during this review.[/dim]"
    )

    family_position = 0
    while family_position < len(families):
        family_index = family_position + 1
        family = families[family_position]
        family_started = time.perf_counter()
        backtracked = False
        actions = [member.action for member in family.members]
        representative = actions[0]
        is_row_action = family.action_type == "remove_exact_duplicates"
        affected_label = "rows" if is_row_action else "cells"

        console.rule(f"Change {family_index} of {len(families)}")
        console.print(
            f"[bold cyan]{_human_action_name(family.action_type, family.parameters)}[/bold cyan]"
        )
        console.print(
            f"This same change is proposed in [bold]{len(family.members)}[/bold] "
            f"{'table(s)' if representative.column is None else 'column(s)'}, affecting "
            f"[bold]{family.affected_count:,}[/bold] {affected_label} in total."
        )

        if representative.description:
            console.print(f"[bold]What it does:[/bold] {_compact_text(representative.description)}")
        console.print(f"[bold]What to check:[/bold] {_review_caution(family.action_type)}")

        impact = Table(title="Where this change would happen")
        impact.add_column("Column / target")
        impact.add_column(f"{affected_label.capitalize()} changed", justify="right")
        for member in family.members[:8]:
            impact.add_row(
                _friendly_target(member.action, include_table=multiple_tables),
                f"{member.affected_count:,}",
            )
        console.print(impact)
        if len(family.members) > 8:
            console.print(
                f"[dim]{len(family.members) - 8} more target(s) are included in this group.[/dim]"
            )

        console.print("[bold]Examples:[/bold]")
        examples_shown = 0
        for member in family.members:
            if member.affected_count == 0:
                continue
            action = member.action
            preview = preview_action(tables, action, sample_limit=min(sample, 3))
            console.print(
                f"[dim]{_friendly_target(action, include_table=multiple_tables)} "
                f"({member.affected_count:,} affected)[/dim]"
            )
            if preview.sample_rows:
                for snapshot in preview.sample_rows[:2]:
                    values = {
                        _friendly_column_name(column): value.value
                        for column, value in snapshot.values.items()
                    }
                    console.print(
                        f"  row {snapshot.row_position}: {_compact_text(repr(values), 220)}"
                    )
            else:
                for change in preview.sample_changes[:3]:
                    console.print(
                        f"  row {change.row_index}: "
                        f"{_snapshot_text(change.old_value)} {ARROW} "
                        f"{_snapshot_text(change.new_value)}"
                    )
            examples_shown += 1
            if examples_shown >= 3:
                break

        if len(family.members) >= 20 or family.affected_count >= 10_000:
            console.print(
                "[yellow]Large change:[/yellow] this decision covers many cells/columns. "
                "Use [i] if you are not confident the same meaning applies everywhere."
            )

        while True:
            decision = typer.prompt(
                "Choose: [a]pprove this group, [i]nspect one by one, "
                "[r]eject this group, [s]decide later, [b]ack, [q]uit",
                default="s",
            ).strip().casefold()

            if decision in {"a", "approve"}:
                conflicts = approval_mapping_conflicts(plan, actions)
                if conflicts:
                    console.print(
                        "[red]Approval blocked:[/red] this would create competing "
                        "canonical targets for the same source value."
                    )
                    _print_mapping_conflicts(conflicts, title="Approval conflict")
                    continue
                bulk_confirmed = False
                if len(family.members) > 1:
                    confirmation = typer.prompt(
                        "Type YES to confirm approval of "
                        f"{len(family.members)} action(s) affecting "
                        f"{family.affected_count:,} {affected_label}",
                        default="",
                        show_default=False,
                    ).strip()
                    if confirmation != "YES":
                        console.print(
                            "[yellow]Approval not confirmed; nothing was approved.[/yellow]"
                        )
                        continue
                    bulk_confirmed = True
                for action in actions:
                    action.decision = ActionDecision.APPROVED
                record_review_event(
                    plan,
                    scope="family",
                    actions=actions,
                    affected_count=family.affected_count,
                    decision="approved",
                    duration_seconds=time.perf_counter() - family_started,
                    bulk_confirmed=bulk_confirmed,
                    targets_preview=family.targets,
                )
                break

            if decision in {"r", "reject"}:
                for action in actions:
                    action.decision = ActionDecision.REJECTED
                record_review_event(
                    plan,
                    scope="family",
                    actions=actions,
                    affected_count=family.affected_count,
                    decision="rejected",
                    duration_seconds=time.perf_counter() - family_started,
                    targets_preview=family.targets,
                )
                break

            if decision in {"s", "skip", "defer"}:
                for action in actions:
                    action.decision = ActionDecision.PROPOSED
                record_review_event(
                    plan,
                    scope="family",
                    actions=actions,
                    affected_count=family.affected_count,
                    decision="deferred",
                    duration_seconds=time.perf_counter() - family_started,
                    targets_preview=family.targets,
                )
                break

            if decision in {"i", "inspect"}:
                record_review_event(
                    plan,
                    scope="family",
                    actions=actions,
                    affected_count=family.affected_count,
                    decision="inspect_individually",
                    duration_seconds=time.perf_counter() - family_started,
                    targets_preview=family.targets,
                )
                for member_index, member in enumerate(family.members, start=1):
                    action = member.action
                    console.rule(
                        f"Change {family_index} · target "
                        f"{member_index}/{len(family.members)}"
                    )
                    console.print(
                        f"[bold]{_friendly_target(action, include_table=multiple_tables)}[/bold]"
                    )
                    console.print(
                        f"{_human_action_name(action.type.value, action.parameters)} · "
                        f"{member.affected_count:,} "
                        f"{'rows' if action.column is None else 'cells'} affected"
                    )
                    console.print(
                        f"[bold]What to check:[/bold] {_review_caution(action.type.value)}"
                    )
                    _print_action_preview(tables, action, sample_limit=sample)
                    item_started = time.perf_counter()
                    while True:
                        item_decision = typer.prompt(
                            "Choose: [a]pprove, [r]eject, [s]decide later, [q]uit",
                            default="s",
                        ).strip().casefold()
                        if item_decision in {"a", "approve"}:
                            conflicts = approval_mapping_conflicts(plan, [action])
                            if conflicts:
                                console.print(
                                    "[red]Approval blocked:[/red] this mapping conflicts "
                                    "with an already approved canonical target."
                                )
                                _print_mapping_conflicts(conflicts, title="Approval conflict")
                                continue
                            action.decision = ActionDecision.APPROVED
                            event_decision = "approved"
                            break
                        if item_decision in {"r", "reject"}:
                            action.decision = ActionDecision.REJECTED
                            event_decision = "rejected"
                            break
                        if item_decision in {"s", "skip", "defer"}:
                            action.decision = ActionDecision.PROPOSED
                            event_decision = "deferred"
                            break
                        if item_decision in {"q", "quit"}:
                            record_review_event(
                                plan,
                                scope="individual",
                                actions=[action],
                                affected_count=member.affected_count,
                                decision="quit",
                                duration_seconds=time.perf_counter() - item_started,
                                targets_preview=[action_target(action)],
                            )
                            save_and_report(partial=True)
                            return
                        console.print("[red]Enter a, r, s, or q.[/red]")
                    record_review_event(
                        plan,
                        scope="individual",
                        actions=[action],
                        affected_count=member.affected_count,
                        decision=event_decision,
                        duration_seconds=time.perf_counter() - item_started,
                        targets_preview=[action_target(action)],
                    )
                break

            if decision in {"b", "back"}:
                if family_position == 0:
                    console.print("[yellow]There is no previous reviewed group.[/yellow]")
                    continue
                previous = families[family_position - 1]
                previous_actions = [member.action for member in previous.members]
                for previous_action in previous_actions:
                    reset_action_for_backtracking(
                        plan,
                        action_id=previous_action.id,
                    )
                record_review_event(
                    plan,
                    scope="backtrack",
                    actions=previous_actions,
                    affected_count=previous.affected_count,
                    decision="reset_to_proposed",
                    duration_seconds=time.perf_counter() - family_started,
                    targets_preview=previous.targets,
                )
                console.print(
                    "[cyan]Previous decision undone.[/cyan] "
                    "Returning to that group."
                )
                family_position -= 1
                backtracked = True
                break

            if decision in {"q", "quit"}:
                record_review_event(
                    plan,
                    scope="family",
                    actions=actions,
                    affected_count=family.affected_count,
                    decision="quit",
                    duration_seconds=time.perf_counter() - family_started,
                    targets_preview=family.targets,
                )
                save_and_report(partial=True)
                return

            console.print("[red]Enter a, i, r, s, b, or q.[/red]")

        if backtracked:
            continue
        family_position += 1

    pending_deferred = [item for item in plan.deferred_reviews if item.status == "pending"]
    if pending_deferred:
        console.rule("Changes that need human judgment")
        console.print(
            f"The AI left [bold]{len(pending_deferred)}[/bold] possible semantic "
            "correction(s) unresolved. These are not executable unless you explicitly approve them."
        )

    deferred_position = 0
    while deferred_position < len(pending_deferred):
        deferred = pending_deferred[deferred_position]
        action = deferred.proposed_action
        preview = preview_action(tables, action, sample_limit=sample)
        console.rule(
            f"Judgment {deferred_position + 1} of {len(pending_deferred)}"
        )
        console.print(f"[bold]{_friendly_target(action, include_table=multiple_tables)}[/bold]")
        console.print(
            f"[bold]Possible change:[/bold] "
            f"{_human_action_name(action.type.value, action.parameters)}"
        )
        console.print(f"[bold]Why it is uncertain:[/bold] {_compact_text(deferred.rationale, 500)}")
        console.print(f"[bold]What to check:[/bold] {_review_caution(action.type.value)}")
        _print_action_preview(tables, action, sample_limit=sample)
        item_started = time.perf_counter()
        backtracked = False

        while True:
            decision = typer.prompt(
                "Choose: [a]pprove this correction, [r]eject it, "
                "[s]decide later, [b]ack, [q]uit",
                default="s",
            ).strip().casefold()
            if decision in {"a", "approve"}:
                conflicts = approval_mapping_conflicts(plan, [action])
                if conflicts:
                    console.print(
                        "[red]Approval blocked:[/red] this candidate conflicts with "
                        "an already approved canonical target."
                    )
                    _print_mapping_conflicts(conflicts, title="Approval conflict")
                    continue
                recorded_action = promote_deferred_candidate(
                    plan,
                    candidate_id=deferred.candidate_id,
                )
                event_decision = "approved"
                break
            if decision in {"r", "reject"}:
                reject_deferred_candidate(
                    plan,
                    candidate_id=deferred.candidate_id,
                )
                recorded_action = action
                event_decision = "rejected"
                break
            if decision in {"s", "skip", "defer"}:
                recorded_action = action
                event_decision = "deferred"
                break
            if decision in {"b", "back"}:
                if deferred_position == 0:
                    console.print(
                        "[yellow]There is no previous deferred judgment.[/yellow]"
                    )
                    continue
                previous = pending_deferred[deferred_position - 1]
                previous_action = previous.proposed_action
                previous_preview = preview_action(
                    tables,
                    previous_action,
                    sample_limit=sample,
                )
                reset_deferred_for_backtracking(
                    plan,
                    candidate_id=previous.candidate_id,
                )
                record_review_event(
                    plan,
                    scope="deferred_ai_backtrack",
                    actions=[previous_action],
                    affected_count=previous_preview.affected_count,
                    decision="reset_to_pending",
                    duration_seconds=time.perf_counter() - item_started,
                    targets_preview=[action_target(previous_action)],
                )
                console.print(
                    "[cyan]Previous deferred judgment undone.[/cyan] "
                    "Returning to that candidate."
                )
                deferred_position -= 1
                backtracked = True
                break
            if decision in {"q", "quit"}:
                record_review_event(
                    plan,
                    scope="deferred_ai",
                    actions=[action],
                    affected_count=preview.affected_count,
                    decision="quit",
                    duration_seconds=time.perf_counter() - item_started,
                    targets_preview=[action_target(action)],
                )
                save_and_report(partial=True)
                return
            console.print("[red]Enter a, r, s, b, or q.[/red]")

        if backtracked:
            continue

        record_review_event(
            plan,
            scope="deferred_ai",
            actions=[recorded_action],
            affected_count=preview.affected_count,
            decision=event_decision,
            duration_seconds=time.perf_counter() - item_started,
            targets_preview=[action_target(action)],
        )
        deferred_position += 1

    save_and_report(partial=False)


@app.command("plan-actions")
def plan_actions_command(
    plan_path: Annotated[Path, typer.Argument(help="Cleaning plan JSON to inspect")],
) -> None:
    """List atomic action ids and current decisions for supported amendments."""
    plan = _load_plan(plan_path)
    atomize_category_mappings(plan)

    table = Table(title="Plan actions")
    table.add_column("Action ID")
    table.add_column("Decision")
    table.add_column("Target")
    table.add_column("Change")

    multiple_tables = len({action.table for action in plan.actions}) > 1
    for action in plan.actions:
        table.add_row(
            action.id,
            action.decision.value,
            _friendly_target(action, include_table=multiple_tables),
            _human_action_name(action.type.value, action.parameters),
        )

    console.print(table)

    console.print("\n[bold]Exact action IDs (copyable):[/bold]")
    for action in plan.actions:
        console.print(f"  {action.id}", markup=False)

    if plan.amendments:
        console.print(
            f"[dim]Recorded amendments: {len(plan.amendments)}[/dim]"
        )


@app.command("amend-plan")
def amend_plan_command(
    plan_path: Annotated[Path, typer.Argument(help="Existing cleaning plan JSON")],
    action_id: Annotated[
        str,
        typer.Option("--action-id", help="Atomic action id to amend"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Write the amended plan JSON here"),
    ] = Path("amended_plan.json"),
    decision: Annotated[
        str | None,
        typer.Option(
            "--decision",
            help="Set decision to proposed, approved, or rejected",
        ),
    ] = None,
    target: Annotated[
        str | None,
        typer.Option(
            "--target",
            help="Retarget one atomic category mapping to this canonical value",
        ),
    ] = None,
    reason: Annotated[
        str,
        typer.Option("--reason", help="Human-readable reason recorded in provenance"),
    ] = "Manual plan amendment",
) -> None:
    """Safely amend an atomic reviewed action without hand-editing plan JSON."""
    if decision is None and target is None:
        raise typer.BadParameter("Provide --decision and/or --target.")

    plan = _load_plan(plan_path)
    atomize_category_mappings(plan)

    if target is not None:
        try:
            amend_mapping_target(
                plan,
                action_id=action_id,
                target=target,
                reason=reason,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    if decision is not None:
        try:
            parsed_decision = ActionDecision(decision.casefold())
        except ValueError as exc:
            raise typer.BadParameter(
                "--decision must be proposed, approved, or rejected."
            ) from exc
        try:
            amend_action_decision(
                plan,
                action_id=action_id,
                decision=parsed_decision,
                reason=reason,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    resolved_output = output.expanduser().resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(plan.model_dump_json(indent=2), encoding="utf-8")

    amended_action = next(
        action for action in plan.actions if action.id == action_id
    )
    console.print(f"[green]Amended plan written to:[/green] {resolved_output}")
    console.print(
        f"Action [bold]{action_id}[/bold] is now "
        f"[bold]{amended_action.decision.value}[/bold]."
    )
    if amended_action.type.value == "normalize_categories":
        console.print(f"Mapping: {amended_action.parameters.get('mapping')!r}")
    console.print(
        f"Recorded plan amendments: [bold]{len(plan.amendments)}[/bold]"
    )


@app.command("apply")
def apply_command(
    input_path: Annotated[Path, typer.Argument(help="Original dataset")],
    plan_path: Annotated[Path, typer.Argument(help="Cleaning plan JSON")],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="New cleaned dataset path"),
    ],
    approve: Annotated[
        list[str] | None,
        typer.Option("--approve", help="Approve one action ID; repeat for multiple actions"),
    ] = None,
    approve_all: Annotated[
        bool,
        typer.Option("--approve-all", help="Explicitly approve every non-rejected action"),
    ] = False,
    audit: Annotated[
        Path | None,
        typer.Option("--audit", help="Audit-log JSON path"),
    ] = None,
    recipe: Annotated[
        Path | None,
        typer.Option("--recipe", help="Reproducible recipe YAML path"),
    ] = None,
) -> None:
    """Apply explicitly approved actions to a new output file."""
    plan = _load_plan(plan_path)
    preapproved = any(
        action.decision == ActionDecision.APPROVED for action in plan.actions
    )
    if not approve_all and not approve and not preapproved:
        raise typer.BadParameter(
            "No actions are approved. Use --approve ACTION_ID, --approve-all, "
            "or review the plan with data-cleaner review first."
        )

    loaded = load_dataset(input_path)
    cleaned, changeset = apply_plan(
        input_path,
        loaded.tables,
        plan,
        approved_ids=set(approve or []),
        approve_all=approve_all,
    )
    if not changeset.executions:
        raise typer.BadParameter("The selected approvals produced no executable actions.")

    output = write_cleaned_dataset(
        input_path,
        loaded.tables,
        cleaned,
        output,
    )
    _write_cleaning_outputs(
        output,
        plan,
        changeset,
        audit=audit,
        recipe=recipe,
    )


@app.command("verify-run")
def verify_run_command(
    source_path: Annotated[Path, typer.Argument(help="Original source dataset")],
    cleaned_path: Annotated[Path, typer.Argument(help="Cleaned output dataset")],
    audit_path: Annotated[
        Path | None,
        typer.Option(
            "--audit",
            help="Audit JSON; defaults to CLEANED.audit.json",
        ),
    ] = None,
    recipe_path: Annotated[
        Path | None,
        typer.Option(
            "--recipe",
            help="Recipe YAML; defaults to CLEANED.recipe.yaml",
        ),
    ] = None,
    json_output: Annotated[
        Path | None,
        typer.Option(
            "--json-output",
            help="Optional machine-readable verification report",
        ),
    ] = None,
) -> None:
    """Verify provenance, replay reproducibility, and reversibility of a cleaning run."""
    source = source_path.expanduser().resolve()
    cleaned = cleaned_path.expanduser().resolve()
    audit = (
        audit_path.expanduser().resolve()
        if audit_path
        else cleaned.with_name(f"{cleaned.name}.audit.json")
    )
    recipe = (
        recipe_path.expanduser().resolve()
        if recipe_path
        else cleaned.with_name(f"{cleaned.name}.recipe.yaml")
    )

    try:
        report = verify_run(source, cleaned, audit, recipe)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    table = Table(title="Cleaning run verification")
    table.add_column("Check")
    table.add_column("Result")
    table.add_column("Details")
    for check in report.checks:
        table.add_row(
            check.name,
            "[green]PASS[/green]" if check.passed else "[red]FAIL[/red]",
            check.detail,
        )
    console.print(table)

    metrics = report.metrics
    console.print(
        "\n[bold]Run metrics:[/bold] "
        f"{metrics.execution_count:,} executed action(s) · "
        f"{metrics.cell_change_events:,} cell-change event(s) · "
        f"{metrics.unique_touched_cells:,} unique touched cell(s) · "
        f"{metrics.repeated_touch_events:,} repeated-touch event(s) · "
        f"{metrics.row_deletions:,} row deletion(s)"
    )
    if metrics.net_different_cells is not None:
        console.print(
            f"Net cells different from source: "
            f"[bold]{metrics.net_different_cells:,}[/bold]"
        )
    if metrics.replay_byte_identical is not None:
        label = "yes" if metrics.replay_byte_identical else "no"
        console.print(
            f"Recipe replay byte-identical to cleaned output: [bold]{label}[/bold] "
            "[dim](logical equivalence is the required check)[/dim]"
        )

    if json_output is not None:
        resolved_json = json_output.expanduser().resolve()
        resolved_json.parent.mkdir(parents=True, exist_ok=True)
        resolved_json.write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        console.print(f"[green]Verification report:[/green] {resolved_json}")

    if report.passed:
        console.print("\n[bold green]VERIFICATION PASS[/bold green]")
        return

    console.print("\n[bold red]VERIFICATION FAIL[/bold red]")
    raise typer.Exit(code=2)


@app.command("replay")
def replay_command(
    input_path: Annotated[Path, typer.Argument(help="Dataset to receive a saved recipe")],
    recipe_path: Annotated[Path, typer.Argument(help="Cleaning recipe YAML")],
    execute: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="Apply the compatible recipe after previewing it. Without this flag, replay is read-only.",
        ),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Cleaned output path; required with --apply"),
    ] = None,
    sample: Annotated[
        int,
        typer.Option("--sample", min=1, max=50, help="Maximum changes shown per action"),
    ] = 5,
) -> None:
    """Validate, preview, and optionally replay a saved recipe on another dataset."""
    loaded = load_dataset(input_path)
    recipe = load_recipe(recipe_path)
    plan, compatibility = recipe_to_plan(input_path, recipe, loaded.tables)

    _print_drift(compatibility)

    if not compatibility.compatible:
        table = Table(title="Recipe compatibility problems")
        table.add_column("Recipe table")
        table.add_column("Column")
        table.add_column("Problem")
        for issue in compatibility.issues:
            table.add_row(issue.table, issue.column or "—", issue.message)
        console.print(table)
        raise typer.Exit(code=2)

    remapped = {
        source: target
        for source, target in compatibility.table_mapping.items()
        if source != target
    }
    if remapped:
        console.print("[dim]Single-table recipe mapping:[/dim]")
        for source, target in remapped.items():
            console.print(f"  {source} {ARROW} {target}")

    _print_plan(plan)
    for action in plan.actions:
        _print_action_preview(loaded.tables, action, sample_limit=sample)

    if not execute:
        console.print(
            "\n[yellow]Preview only.[/yellow] Re-run with --apply -o OUTPUT "
            "to execute this recipe."
        )
        return

    if output is None:
        raise typer.BadParameter("--output/-o is required when using --apply.")

    cleaned, changeset = apply_plan(
        input_path,
        loaded.tables,
        plan,
        approve_all=True,
    )
    if not changeset.executions:
        raise typer.BadParameter("The recipe produced no executable actions.")

    output = write_cleaned_dataset(
        input_path,
        loaded.tables,
        cleaned,
        output,
    )
    _write_cleaning_outputs(output, plan, changeset)


@app.command("undo")
def undo_command(
    cleaned_path: Annotated[Path, typer.Argument(help="Cleaned dataset")],
    audit_path: Annotated[Path, typer.Argument(help="Audit-log JSON")],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Path for the restored dataset"),
    ],
) -> None:
    """Reverse an applied ChangeSet into a new file."""
    changeset = ChangeSet.model_validate_json(audit_path.read_text(encoding="utf-8"))
    if changeset.result_sha256 and file_sha256(cleaned_path) != changeset.result_sha256:
        raise typer.BadParameter(
            "The cleaned dataset no longer matches this audit log; refusing to undo."
        )
    loaded = load_dataset(cleaned_path)
    restored = undo_changeset(loaded.tables, changeset)
    inserted_rows: dict[str, list] = {}
    for execution in changeset.executions:
        if execution.deleted_rows:
            inserted_rows.setdefault(execution.table, []).extend(execution.deleted_rows)

    restored_path = write_cleaned_dataset(
        cleaned_path,
        loaded.tables,
        restored,
        output,
        inserted_rows=inserted_rows,
    )
    console.print(f"[green]Restored dataset:[/green] {restored_path}")


def _guided_status_table(session: GuidedSession) -> Table:
    table = Table(title="Guided cleaning session")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Stage", session.stage.value)
    table.add_row("Source", session.source_path)
    table.add_row("Workspace", session.workspace)
    table.add_row("Approved actions", str(session.approved_action_count))
    table.add_row("Executed actions", str(session.executed_action_count))
    table.add_row("Cleaned output", session.cleaned_path)
    if session.schema_path:
        table.add_row("Study schema", session.schema_path)
    if session.plan_path:
        table.add_row("Plan", session.plan_path)
    if session.reviewed_plan_path:
        table.add_row("Reviewed plan", session.reviewed_plan_path)
    if session.verification_path:
        table.add_row("Verification", session.verification_path)
    return table


def _print_guided_next_step(session: GuidedSession) -> None:
    console.print(_guided_status_table(session))
    if session.stage in {
        GuidedStage.SCHEMA_REVIEW,
        GuidedStage.CLEANING_REVIEW,
        GuidedStage.PLANNING,
    }:
        console.print(
            f"[dim]Continue later with: data-cleaner guided-resume "
            f"{session.workspace!r}[/dim]"
        )


def _continue_guided_session(
    session: GuidedSession,
    *,
    sample: int,
) -> None:
    source = Path(session.source_path)
    workspace = Path(session.workspace)

    if session.stage == GuidedStage.SCHEMA_REVIEW:
        draft_path = Path(
            session.reviewed_schema_draft_path
            or session.schema_draft_path
        )
        schema_path = workspace / "study_schema.yaml"
        reviewed_draft = workspace / "schema_draft.reviewed.yaml"
        resume_schema = session.reviewed_schema_draft_path is not None

        schema_review_command(
            input_path=source,
            draft_path=draft_path,
            output=schema_path,
            reviewed_draft=reviewed_draft,
            unresolved_only=resume_schema,
        )

        if not schema_path.exists():
            if reviewed_draft.exists():
                session.reviewed_schema_draft_path = str(reviewed_draft)
            session.stage = GuidedStage.SCHEMA_REVIEW
            save_guided_session(session)
            console.print(
                "[yellow]Guided session paused during schema review.[/yellow]"
            )
            _print_guided_next_step(session)
            return

        record_confirmed_schema(
            session,
            schema_path=schema_path,
            reviewed_draft_path=reviewed_draft,
        )

    if session.stage == GuidedStage.PLANNING:
        plan = prepare_guided_plan(session)
        if session.stage == GuidedStage.NO_CHANGES:
            console.print(
                "[green]No reviewable cleaning changes were found.[/green] "
                "The source dataset was not modified."
            )
            _print_guided_next_step(session)
            return
        console.print(
            f"[green]Cleaning plan prepared:[/green] "
            f"{len(plan.actions)} action(s)."
        )

    if session.stage == GuidedStage.CLEANING_REVIEW:
        plan_input = Path(
            session.reviewed_plan_path
            or session.plan_path
            or ""
        )
        if not plan_input.exists():
            raise typer.BadParameter(
                "Guided session is missing its cleaning plan artifact."
            )
        reviewed_path = workspace / "reviewed_plan.json"
        resume_review = session.reviewed_plan_path is not None

        review_grouped_command(
            input_path=source,
            plan_path=plan_input,
            output=reviewed_path,
            sample=sample,
            unresolved_only=resume_review,
        )

        if not reviewed_path.exists():
            raise typer.BadParameter(
                "Cleaning review did not produce a reviewed plan."
            )
        reviewed = record_reviewed_plan(session, reviewed_path)
        if (
            reviewed.review_session is None
            or reviewed.review_session.outcome != "completed"
        ):
            console.print(
                "[yellow]Guided session paused during cleaning review.[/yellow]"
            )
            _print_guided_next_step(session)
            return

        verification = finalize_guided_session(session)
        if session.stage == GuidedStage.NO_APPROVED_CHANGES:
            console.print(
                "[green]Review completed with no approved changes.[/green] "
                "The source dataset was not modified."
            )
            _print_guided_next_step(session)
            return
        if session.stage == GuidedStage.NO_EXECUTABLE_CHANGES:
            console.print(
                "[green]Approved changes produced no remaining executable edits.[/green] "
                "The source dataset was not modified."
            )
            _print_guided_next_step(session)
            return
        if verification is None:
            _print_guided_next_step(session)
            return
        if verification.passed:
            console.print(
                "[bold green]GUIDED CLEANING COMPLETE — VERIFIED[/bold green]"
            )
            console.print(
                f"[green]Cleaned dataset:[/green] {session.cleaned_path}"
            )
            console.print(
                f"[green]Verification report:[/green] "
                f"{session.verification_path}"
            )
            _print_guided_next_step(session)
            return

        console.print(
            "[bold red]Guided cleaning finished but verification failed.[/bold red]"
        )
        _print_guided_next_step(session)
        raise typer.Exit(code=2)

    _print_guided_next_step(session)


@app.command("guided-clean")
def guided_clean_command(
    input_path: Annotated[Path, typer.Argument(help="Dataset to clean")],
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            "-w",
            help=(
                "New directory for run artifacts; defaults beside the dataset "
                "as <name>_cleaning_run"
            ),
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Optional cleaned output path; defaults inside the workspace",
        ),
    ] = None,
    schema_path: Annotated[
        Path | None,
        typer.Option(
            "--schema",
            help="Optional already-confirmed StudySchema; otherwise a draft is reviewed",
        ),
    ] = None,
    sample: Annotated[
        int,
        typer.Option(
            "--sample",
            min=1,
            max=50,
            help="Maximum examples shown during cleaning review",
        ),
    ] = 5,
) -> None:
    """Run the safe deterministic cleaning workflow as one guided session."""
    source = input_path.expanduser().resolve()
    selected_workspace = (
        workspace
        if workspace is not None
        else source.with_name(f"{source.stem}_cleaning_run")
    )
    try:
        session = create_guided_session(
            source,
            selected_workspace,
            output_path=output,
            schema_path=schema_path,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    console.print(
        f"[green]Guided workspace created:[/green] {session.workspace}"
    )
    console.print(
        "[dim]The source dataset is read-only throughout review. "
        "A cleaned copy is written only after completed approval.[/dim]"
    )
    try:
        _continue_guided_session(session, sample=sample)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("guided-resume")
def guided_resume_command(
    workspace: Annotated[
        Path,
        typer.Argument(help="Existing guided workspace or session.json"),
    ],
    sample: Annotated[
        int,
        typer.Option(
            "--sample",
            min=1,
            max=50,
            help="Maximum examples shown during cleaning review",
        ),
    ] = 5,
) -> None:
    """Resume a paused guided cleaning session without repeating decided items."""
    try:
        session = load_guided_session(workspace)
        _continue_guided_session(session, sample=sample)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("guided-status")
def guided_status_command(
    workspace: Annotated[
        Path,
        typer.Argument(help="Existing guided workspace or session.json"),
    ],
) -> None:
    """Show the current stage and artifacts for a guided cleaning session."""
    try:
        session = load_guided_session(workspace)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(_guided_status_table(session))
    typer.echo(f"Exact workspace: {session.workspace}")


@app.command("web")
def web_command(
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            help="Local directory for browser-session uploads and run artifacts",
        ),
    ] = None,
    host: Annotated[
        str,
        typer.Option(
            "--host",
            help="Bind host; loopback only unless --allow-network is explicit",
        ),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", min=1, max=65535),
    ] = 8765,
    open_browser: Annotated[
        bool,
        typer.Option(
            "--open-browser/--no-open-browser",
            help="Open the local UI in the default browser",
        ),
    ] = True,
    allow_network: Annotated[
        bool,
        typer.Option(
            "--allow-network",
            help="Explicitly permit binding to a non-loopback network interface",
        ),
    ] = False,
    max_upload_mb: Annotated[
        int,
        typer.Option(
            "--max-upload-mb",
            min=1,
            help="Maximum dataset upload size accepted by the local UI",
        ),
    ] = 1024,
    narrator_model: Annotated[
        str | None,
        typer.Option(
            "--narrator-model",
            help=(
                "Optional local Ollama model used only to rewrite explanations "
                "in dataset-aware language; it cannot change the cleaning plan"
            ),
        ),
    ] = None,
) -> None:
    """Launch the local-only browser interface."""
    try:
        from .web_app import run_local_web
    except ModuleNotFoundError as exc:
        console.print(
            "[red]The browser UI dependencies are not installed.[/red] "
            'Install with: pip install -e ".[web]"'
        )
        raise typer.Exit(code=2) from exc

    try:
        selected_data_dir = (
            data_dir
            if data_dir is not None
            else Path.home() / ".data-cleaner" / "web"
        )
        run_local_web(
            data_root=selected_data_dir,
            host=host,
            port=port,
            open_browser=open_browser,
            allow_network=allow_network,
            max_upload_mb=max_upload_mb,
            narrator_model=narrator_model,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


if __name__ == "__main__":
    app()
