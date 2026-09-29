from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.table import Table

from .actions import ActionDecision, ChangeSet, CleaningPlan
from .ai_candidates import build_ai_candidates, semantic_review_candidates
from .ai_context import build_ai_context
from .ai_planner import build_review_contexts, propose_ai_plan, resolve_model
from .ai_provider import AIProviderError, OllamaProvider
from .benchmark import run_ai_benchmark
from .codebook import suggest_codebooks
from .engine import profile_dataset
from .io import load_dataset
from .issue_summary import summarize_issues, visible_issues
from .planner import file_sha256, propose_actions
from .preview_overlap import overlapping_previews
from .recipe import load_recipe, recipe_to_plan
from .recall_benchmark import run_v010_recall_benchmark
from .recall_eval import build_recall_surface_report
from .review import (
    action_target,
    begin_review_session,
    finish_review_session,
    group_actions,
    record_review_event,
)
from .study_schema import load_study_schema
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
                "column(s) omitted from routine missing-token proposals. "
                "Original findings remain in the profile; plan notes retain "
                "their targets and issue IDs. Explicit study-schema rules can "
                "still propose a change."
            )
            breakdown = Table(title="Structural-only fields (no routine action)")
            breakdown.add_column("Table")
            breakdown.add_column("Columns", justify="right")
            for name, columns in sorted(omitted.items()):
                breakdown.add_row(name, str(len(columns)))
            console.print(breakdown)

    table = Table(title="Proposed cleaning actions")
    table.add_column("Action ID")
    table.add_column("Type")
    table.add_column("Table")
    table.add_column("Column")
    table.add_column("Confidence", justify="right")
    table.add_column("Origin")
    table.add_column("Decision")
    table.add_column("Description")
    for action in plan.actions:
        table.add_row(
            action.id,
            action.type.value,
            action.table,
            action.column or "—",
            f"{action.confidence:.0%}",
            action.origin.value,
            action.decision.value,
            action.title,
        )
    console.print(table)
    if plan.deferred_reviews:
        deferred_table = Table(title="Deferred AI reviews — not executable")
        deferred_table.add_column("Candidate")
        deferred_table.add_column("Target")
        deferred_table.add_column("Proposed change")
        deferred_table.add_column("Status")
        deferred_table.add_column("Reason")
        for deferred in plan.deferred_reviews:
            action = deferred.proposed_action
            deferred_table.add_row(
                deferred.candidate_id,
                f"{action.table}.{action.column or '(table)'}",
                repr(action.parameters),
                deferred.status,
                deferred.rationale,
            )
        console.print(deferred_table)
    console.print(
        "\n[dim]No action is applied until it is explicitly approved.[/dim]"
    )


def _print_action_preview(tables, action, sample_limit: int = 8) -> None:
    preview = preview_action(tables, action, sample_limit=sample_limit)

    if preview.sample_rows:
        table = Table(
            title=f"{action.id}: {action.title} — {preview.affected_count} row(s)"
        )
        table.add_column("Original row", justify="right")
        table.add_column("Row values")
        for snapshot in preview.sample_rows:
            values = {
                column: value.value
                for column, value in snapshot.values.items()
            }
            table.add_row(str(snapshot.row_position), repr(values))
        console.print(table)
        return

    table = Table(
        title=f"{action.id}: {action.title} — {preview.affected_count} cell(s)"
    )
    table.add_column("Row", justify="right")
    table.add_column("Before")
    table.add_column("After")
    for change in preview.sample_changes:
        table.add_row(
            str(change.row_index),
            repr(change.old_value.value),
            repr(change.new_value.value),
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


@app.command("codebook-suggest")
def codebook_suggest_command(
    input_path: Annotated[Path, typer.Argument(help="Dataset to inspect for recurring categorical vocabularies")],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Write review-only codebook suggestions as YAML"),
    ] = Path("codebook_suggestions.yaml"),
) -> None:
    """Suggest recurring categorical vocabularies without changing data."""
    loaded = load_dataset(input_path)
    report = suggest_codebooks(input_path, loaded.tables)

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(
            report.model_dump(mode="json"),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    table = Table(title="Candidate codebooks — review required")
    table.add_column("Table")
    table.add_column("Column")
    table.add_column("Core labels", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Rare/other values", justify="right")
    table.add_column("Exact variant mappings", justify="right")
    for suggestion in report.suggestions:
        table.add_row(
            suggestion.table,
            suggestion.column,
            str(len(suggestion.core_values)),
            f"{suggestion.core_coverage:.1%}",
            str(len(suggestion.rare_values)),
            str(len(suggestion.canonical_map)),
        )
    console.print(table)
    console.print(
        "[dim]These are observed vocabulary suggestions only. "
        "They are not a study schema and authorize no transformations.[/dim]"
    )
    console.print(f"[green]Codebook suggestions written to:[/green] {output}")


@app.command("recall-benchmark")
def recall_benchmark_command(
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Write deterministic recall benchmark JSON"),
    ] = Path("recall_benchmark.json"),
) -> None:
    """Run the v0.10 deterministic recall benchmark without Ollama."""
    summary = run_v010_recall_benchmark()
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    console.print(
        f"[bold]Recall benchmark:[/bold] {summary.passed_cases}/{summary.case_count} "
        f"cases passed ({summary.pass_rate:.1%})"
    )
    console.print(
        f"Required recall: [bold]{summary.required_hits}/{summary.required_total} "
        f"({summary.required_recall:.1%})[/bold] · "
        f"Forbidden exposures: [bold]{summary.forbidden_hits}/{summary.forbidden_total}[/bold]"
    )
    console.print("[dim]No Ollama calls and no external dataset edits.[/dim]")
    console.print(f"[green]Benchmark written to:[/green] {output}")


@app.command("recall-report")
def recall_report_command(
    input_path: Annotated[Path, typer.Argument(help="Dataset to inspect with deterministic v0.10 recall checks")],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Write the recall-surface JSON report"),
    ] = Path("recall_report.json"),
) -> None:
    """Summarize the deterministic v0.10 review surface without AI or edits."""
    report = build_recall_surface_report(input_path)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    table = Table(title="v0.10 deterministic recall surface")
    table.add_column("Signal")
    table.add_column("Count", justify="right")
    table.add_column("Affected", justify="right")
    table.add_row(
        "Multi-select normalization actions",
        str(report.multiselect_action_count),
        str(report.multiselect_action_affected_count),
    )
    table.add_row(
        "Conditional QA pairs",
        str(report.conditional_pair_count),
        str(report.conditional_affected_count),
    )
    table.add_row(
        "Other/specify QA pairs",
        str(report.other_specify_pair_count),
        str(report.other_specify_affected_count),
    )
    table.add_row(
        "Normalization-equivalent duplicate QA",
        str(report.normalized_duplicate_group_issue_count),
        str(report.normalized_duplicate_affected_count),
    )
    table.add_row(
        "Read-only codebook suggestions",
        str(report.codebook_suggestion_count),
        "—",
    )
    console.print(table)
    console.print(
        "[dim]Read-only evaluation: no Ollama calls and no dataset edits.[/dim]"
    )
    console.print(f"[green]Recall report written to:[/green] {output}")


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
        typer.Option("--suite", help="Benchmark suite: core, research, or all"),
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
    if normalized_suite not in {"core", "research", "all"}:
        raise typer.BadParameter("--suite must be one of: core, research, all.")
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

    for deferred in plan.deferred_reviews:
        if deferred.status != "pending":
            continue
        action = deferred.proposed_action
        console.rule(f"Deferred AI review {deferred.candidate_id}")
        console.print(f"[bold]{action.table}.{action.column or '(table)'}[/bold]")
        console.print(deferred.question)
        console.print(f"[yellow]AI deferred:[/yellow] {deferred.rationale}")
        console.print(f"Proposed parameters: {action.parameters!r}")
        _print_action_preview(tables, action, sample_limit=sample)
        while True:
            decision = typer.prompt(
                "Manual decision: [a]pprove candidate, [r]eject, [s]kip, [q]uit",
                default="s",
            ).strip().casefold()
            if decision in {"a", "approve"}:
                matching = next(
                    (
                        existing
                        for existing in plan.actions
                        if existing.type == action.type
                        and existing.table == action.table
                        and existing.column == action.column
                        and existing.parameters == action.parameters
                    ),
                    None,
                )
                if matching is not None:
                    matching.decision = ActionDecision.APPROVED
                else:
                    approved_action = action.model_copy(deep=True)
                    approved_action.decision = ActionDecision.APPROVED
                    approved_action.description = (
                        "Manually approved deferred AI candidate: " + deferred.rationale
                    )
                    plan.actions.append(approved_action)
                deferred.status = "approved"
                break
            if decision in {"r", "reject"}:
                deferred.status = "rejected"
                break
            if decision in {"s", "skip", "defer"}:
                break
            if decision in {"q", "quit"}:
                output = output.expanduser().resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
                console.print(f"[yellow]Partial review saved to:[/yellow] {output}")
                return
            console.print("[red]Enter a, r, s, or q.[/red]")

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
        typer.Option("--sample", min=1, max=50, help="Maximum changes shown when drilling down"),
    ] = 5,
) -> None:
    """Review exact transformation families with explicit human approval."""
    plan = _load_plan(plan_path)
    if file_sha256(input_path) != plan.source_sha256:
        raise typer.BadParameter(
            "The input file changed after the plan was created. Generate a new plan."
        )

    tables = load_dataset(input_path).tables
    families = group_actions(tables, plan)
    session_started = begin_review_session(plan)

    def save_and_report(*, partial: bool) -> None:
        finish_review_session(plan, session_started)
        resolved_output = output.expanduser().resolve()
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        resolved_output.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        approved = sum(a.decision == ActionDecision.APPROVED for a in plan.actions)
        rejected = sum(a.decision == ActionDecision.REJECTED for a in plan.actions)
        unresolved = sum(a.decision == ActionDecision.PROPOSED for a in plan.actions)
        label = "Partial grouped review saved to" if partial else "Reviewed plan written to"
        console.print(f"[green]{label}:[/green] {resolved_output}")
        console.print(
            f"Approved: [bold]{approved}[/bold] · "
            f"Rejected: [bold]{rejected}[/bold] · "
            f"Deferred: [bold]{unresolved}[/bold]"
        )
        if plan.review_session and plan.review_session.duration_seconds is not None:
            console.print(
                "Review time: "
                f"[bold]{plan.review_session.duration_seconds:.1f}s[/bold] · "
                f"Decision events: [bold]{len(plan.review_session.events)}[/bold]"
            )
        console.print("[dim]No edits were applied.[/dim]")

    console.print(
        f"[bold]Grouped review:[/bold] {len(plan.actions)} action(s) in "
        f"{len(families)} exact transformation family/families."
    )
    console.print(
        "[dim]Bulk approval changes only the reviewed plan. "
        "Multi-action approval requires typing YES exactly.[/dim]"
    )

    for family_index, family in enumerate(families, start=1):
        family_started = time.perf_counter()
        console.rule(f"Family {family_index}/{len(families)}")
        console.print(f"[bold]Type:[/bold] {family.action_type}")
        console.print(f"[bold]Origin:[/bold] {family.origin}")
        console.print(f"[bold]Parameters:[/bold] {family.parameters!r}")
        console.print(
            f"[bold]Actions:[/bold] {len(family.members)} · "
            f"[bold]Affected by independent previews:[/bold] {family.affected_count:,}"
        )

        impact = Table(title="Highest-impact targets")
        impact.add_column("Target")
        impact.add_column("Affected", justify="right")
        for member in family.members[:12]:
            impact.add_row(action_target(member.action), f"{member.affected_count:,}")
        console.print(impact)
        if len(family.members) > 12:
            console.print(
                f"[dim]{len(family.members) - 12} additional target(s) in this family.[/dim]"
            )

        while True:
            decision = typer.prompt(
                "Family decision: [a]pprove all, [r]eject all, "
                "[i]nspect individually, [s]kip/defer, [q]uit",
                default="s",
            ).strip().casefold()

            if decision in {"a", "approve"}:
                bulk_confirmed = False
                if len(family.members) > 1:
                    confirmation = typer.prompt(
                        "Type YES to approve the entire family "
                        f"({len(family.members)} actions; "
                        f"{family.affected_count:,} affected items)",
                        default="",
                        show_default=False,
                    ).strip()
                    if confirmation != "YES":
                        console.print("[yellow]Family approval not confirmed.[/yellow]")
                        continue
                    bulk_confirmed = True
                actions = [member.action for member in family.members]
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
                actions = [member.action for member in family.members]
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
                actions = [member.action for member in family.members]
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
                actions = [member.action for member in family.members]
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
                        f"Family {family_index} · action "
                        f"{member_index}/{len(family.members)}"
                    )
                    console.print(
                        f"[bold]{action_target(action)}[/bold] · "
                        f"{member.affected_count:,} affected"
                    )
                    console.print(action.description)
                    _print_action_preview(tables, action, sample_limit=sample)
                    item_started = time.perf_counter()
                    while True:
                        item_decision = typer.prompt(
                            "Decision: [a]pprove, [r]eject, [s]kip/defer, [q]uit",
                            default="s",
                        ).strip().casefold()
                        if item_decision in {"a", "approve"}:
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

            if decision in {"q", "quit"}:
                actions = [member.action for member in family.members]
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

            console.print("[red]Enter a, r, i, s, or q.[/red]")

    for deferred in plan.deferred_reviews:
        if deferred.status != "pending":
            continue
        action = deferred.proposed_action
        preview = preview_action(tables, action, sample_limit=sample)
        console.rule(f"Deferred AI review {deferred.candidate_id}")
        console.print(f"[bold]{action_target(action)}[/bold]")
        console.print(deferred.question)
        console.print(f"[yellow]AI deferred:[/yellow] {deferred.rationale}")
        console.print(f"Proposed parameters: {action.parameters!r}")
        _print_action_preview(tables, action, sample_limit=sample)
        item_started = time.perf_counter()
        while True:
            decision = typer.prompt(
                "Manual decision: [a]pprove candidate, [r]eject, "
                "[s]kip/defer, [q]uit",
                default="s",
            ).strip().casefold()
            if decision in {"a", "approve"}:
                matching = next(
                    (
                        existing
                        for existing in plan.actions
                        if existing.type == action.type
                        and existing.table == action.table
                        and existing.column == action.column
                        and existing.parameters == action.parameters
                    ),
                    None,
                )
                if matching is not None:
                    matching.decision = ActionDecision.APPROVED
                    recorded_action = matching
                else:
                    approved_action = action.model_copy(deep=True)
                    approved_action.decision = ActionDecision.APPROVED
                    approved_action.description = (
                        "Manually approved deferred AI candidate: " + deferred.rationale
                    )
                    plan.actions.append(approved_action)
                    recorded_action = approved_action
                deferred.status = "approved"
                event_decision = "approved"
                break
            if decision in {"r", "reject"}:
                deferred.status = "rejected"
                recorded_action = action
                event_decision = "rejected"
                break
            if decision in {"s", "skip", "defer"}:
                recorded_action = action
                event_decision = "deferred"
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
            console.print("[red]Enter a, r, s, or q.[/red]")
        record_review_event(
            plan,
            scope="deferred_ai",
            actions=[recorded_action],
            affected_count=preview.affected_count,
            decision=event_decision,
            duration_seconds=time.perf_counter() - item_started,
            targets_preview=[action_target(action)],
        )

    save_and_report(partial=False)


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
            console.print(f"  {source} → {target}")

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


if __name__ == "__main__":
    app()
