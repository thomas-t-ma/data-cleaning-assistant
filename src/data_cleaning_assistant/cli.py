from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .actions import ActionDecision, ChangeSet, CleaningPlan
from .ai_context import build_ai_context
from .ai_planner import propose_ai_plan, resolve_model
from .ai_provider import AIProviderError, OllamaProvider
from .engine import profile_dataset
from .io import load_dataset
from .planner import file_sha256, propose_actions
from .recipe import load_recipe, recipe_to_plan
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


def _print_summary(report) -> None:
    console.print(f"\n[bold]Source:[/bold] {report.source_path}")
    console.print(f"[bold]Type:[/bold] {report.source_type}")

    table = Table(title="Dataset profile")
    table.add_column("Table / sheet")
    table.add_column("Rows", justify="right")
    table.add_column("Columns", justify="right")
    table.add_column("Grain")
    table.add_column("Candidate key")
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

    if report.relationships:
        rel_table = Table(title="Cross-table relationships")
        rel_table.add_column("Left")
        rel_table.add_column("Right")
        rel_table.add_column("Relationship")
        rel_table.add_column("Shared", justify="right")
        rel_table.add_column("Confidence", justify="right")
        for rel in report.relationships:
            rel_table.add_row(
                f"{rel.left_table}.{rel.left_column}",
                f"{rel.right_table}.{rel.right_column}",
                rel.relationship,
                str(rel.shared_value_count),
                f"{rel.confidence:.0%}",
            )
        console.print(rel_table)

    if report.issues:
        issue_table = Table(title="Issues")
        issue_table.add_column("Severity")
        issue_table.add_column("Type")
        issue_table.add_column("Table")
        issue_table.add_column("Column")
        issue_table.add_column("Affected", justify="right")
        issue_table.add_column("Description")
        for issue in report.issues:
            issue_table.add_row(
                issue.severity.value,
                issue.type,
                issue.table,
                issue.column or "—",
                str(issue.affected_count),
                issue.title,
            )
        console.print(issue_table)
    else:
        console.print("[green]No issues detected by the current rule set.[/green]")


def _print_plan(plan: CleaningPlan) -> None:
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
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write the full JSON report to this path"),
    ] = None,
) -> None:
    """Profile a dataset without changing the source file."""
    report = profile_dataset(input_path)
    _print_summary(report)

    if output:
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
        console.print(f"\n[green]Report written to:[/green] {output}")


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
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write the exact AI context JSON to this path"),
    ] = None,
) -> None:
    """Show exactly what compact context would be sent to the AI planner."""
    report = profile_dataset(input_path)
    context = build_ai_context(report)
    payload = context.model_dump_json(indent=2)

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
    provider = OllamaProvider(url)
    try:
        selected_model = resolve_model(provider, model)
        result = propose_ai_plan(
            input_path,
            provider=provider,
            model=selected_model,
            include_rules=rules,
        )
    except (AIProviderError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    console.print(
        f"[green]AI planner:[/green] {result.provider} / {result.model}"
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
            rejected_table.add_row(rejected.type.value, target, rejected.reason)
        console.print(rejected_table)

    _print_plan(result.plan)

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.plan.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"\n[green]AI-assisted plan written to:[/green] {output}")

    if context_out:
        context_out = context_out.expanduser().resolve()
        context_out.parent.mkdir(parents=True, exist_ok=True)
        context_out.write_text(result.context.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[green]AI context written to:[/green] {context_out}")


@app.command("plan")
def plan_command(
    input_path: Annotated[Path, typer.Argument(help="Dataset to inspect")],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Path for the proposed cleaning plan JSON"),
    ] = Path("cleaning_plan.json"),
) -> None:
    """Generate reviewable cleaning actions without changing the dataset."""
    plan = propose_actions(input_path)
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
    for action in plan.actions:
        _print_action_preview(tables, action, sample_limit=sample)


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
                output.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
                console.print(f"[yellow]Review stopped. Partial plan saved to:[/yellow] {output}")
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
