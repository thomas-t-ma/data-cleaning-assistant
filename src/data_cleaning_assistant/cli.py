from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .engine import profile_dataset

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


if __name__ == "__main__":
    app()
