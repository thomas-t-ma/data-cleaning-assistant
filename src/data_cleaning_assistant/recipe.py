from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field
import yaml

from .actions import ActionOrigin, ActionType, CleaningAction, CleaningPlan
from .inference import infer_semantic_type, is_missing_like, parse_number
from .planner import file_sha256


class RecipeAction(BaseModel):
    type: ActionType
    table: str
    column: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class ColumnSchema(BaseModel):
    inferred_type: str
    missing_percent: float
    unique_count: int
    categories: list[str] | None = None
    numeric_min: float | None = None
    numeric_max: float | None = None


class TableSchema(BaseModel):
    row_count: int
    columns: dict[str, ColumnSchema] = Field(default_factory=dict)


class CleaningRecipe(BaseModel):
    version: int = 1
    schema_: dict[str, TableSchema] = Field(default_factory=dict, alias="schema")
    actions: list[RecipeAction] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class RecipeCompatibilityIssue(BaseModel):
    table: str
    column: str | None = None
    message: str


class SchemaDrift(BaseModel):
    severity: str
    kind: str
    table: str
    column: str | None = None
    message: str


class RecipeCompatibilityReport(BaseModel):
    compatible: bool
    table_mapping: dict[str, str] = Field(default_factory=dict)
    issues: list[RecipeCompatibilityIssue] = Field(default_factory=list)
    drift: list[SchemaDrift] = Field(default_factory=list)


def _column_schema(column: str, series: pd.Series) -> dict[str, Any]:
    inferred_type, _, _ = infer_semantic_type(column, series)
    values = [value for value in series.tolist() if not is_missing_like(value)]
    missing_count = len(series) - len(values)
    unique_values = list(dict.fromkeys(str(value) for value in values))

    numbers = [parse_number(value) for value in values]
    numbers = [value for value in numbers if value is not None]

    categories = None
    if inferred_type not in {"identifier", "numeric", "date"} and len(unique_values) <= 30:
        categories = sorted(unique_values)

    return {
        "inferred_type": inferred_type,
        "missing_percent": (missing_count / len(series) * 100.0) if len(series) else 0.0,
        "unique_count": len(unique_values),
        "categories": categories,
        "numeric_min": min(numbers) if numbers else None,
        "numeric_max": max(numbers) if numbers else None,
    }


def build_schema_snapshot(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for table_name, frame in tables.items():
        snapshot[table_name] = {
            "row_count": len(frame),
            "columns": {
                str(column): _column_schema(str(column), frame[column])
                for column in frame.columns
            },
        }
    return snapshot


def load_recipe(path: str | Path) -> CleaningRecipe:
    payload = yaml.safe_load(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Recipe must contain a YAML mapping.")
    return CleaningRecipe.model_validate(payload)


def _table_mapping(
    recipe: CleaningRecipe,
    target_tables: dict[str, Any],
) -> tuple[dict[str, str], list[RecipeCompatibilityIssue]]:
    action_tables = {action.table for action in recipe.actions}
    schema_tables = set(recipe.schema_)
    recipe_tables = sorted(action_tables | schema_tables)
    target_names = list(target_tables)
    issues: list[RecipeCompatibilityIssue] = []

    if not recipe_tables:
        return {}, issues

    if len(recipe_tables) == 1 and len(target_names) == 1:
        return {recipe_tables[0]: target_names[0]}, issues

    mapping: dict[str, str] = {}
    for table in recipe_tables:
        if table in target_tables:
            mapping[table] = table
        elif table in action_tables:
            issues.append(
                RecipeCompatibilityIssue(
                    table=table,
                    message=f"Recipe table {table!r} is missing from the target dataset.",
                )
            )
    return mapping, issues


def _schema_drift(
    recipe: CleaningRecipe,
    target_tables: dict[str, pd.DataFrame],
    mapping: dict[str, str],
) -> list[SchemaDrift]:
    if not recipe.schema_:
        return []

    target_snapshot = build_schema_snapshot(target_tables)
    drift: list[SchemaDrift] = []

    for source_table, source_schema in recipe.schema_.items():
        mapped_table = mapping.get(source_table)
        if mapped_table is None or mapped_table not in target_snapshot:
            drift.append(
                SchemaDrift(
                    severity="warning",
                    kind="missing_table",
                    table=source_table,
                    message=(
                        f"Baseline table {source_table!r} is absent from the target dataset."
                    ),
                )
            )
            continue
        target_schema = TableSchema.model_validate(target_snapshot[mapped_table])

        source_columns = set(source_schema.columns)
        target_columns = set(target_schema.columns)

        for column in sorted(target_columns - source_columns):
            drift.append(
                SchemaDrift(
                    severity="info",
                    kind="new_column",
                    table=mapped_table,
                    column=column,
                    message=f"New column {column!r} was not present when the recipe was created.",
                )
            )

        for column in sorted(source_columns - target_columns):
            drift.append(
                SchemaDrift(
                    severity="warning",
                    kind="missing_column",
                    table=mapped_table,
                    column=column,
                    message=f"Baseline column {column!r} is absent from the target dataset.",
                )
            )

        for column in sorted(source_columns & target_columns):
            before = source_schema.columns[column]
            after = target_schema.columns[column]

            if before.inferred_type != after.inferred_type:
                drift.append(
                    SchemaDrift(
                        severity="warning",
                        kind="type_change",
                        table=mapped_table,
                        column=column,
                        message=(
                            f"Inferred type changed from {before.inferred_type!r} "
                            f"to {after.inferred_type!r}."
                        ),
                    )
                )

            missing_delta = after.missing_percent - before.missing_percent
            if abs(missing_delta) >= 5.0:
                severity = "warning" if abs(missing_delta) >= 20.0 else "info"
                drift.append(
                    SchemaDrift(
                        severity=severity,
                        kind="missingness_change",
                        table=mapped_table,
                        column=column,
                        message=(
                            f"Missingness changed from {before.missing_percent:.1f}% "
                            f"to {after.missing_percent:.1f}% "
                            f"({missing_delta:+.1f} percentage points)."
                        ),
                    )
                )

            if before.categories is not None and after.categories is not None:
                new_categories = sorted(set(after.categories) - set(before.categories))
                if new_categories:
                    preview = ", ".join(repr(value) for value in new_categories[:8])
                    suffix = " …" if len(new_categories) > 8 else ""
                    drift.append(
                        SchemaDrift(
                            severity="warning",
                            kind="new_categories",
                            table=mapped_table,
                            column=column,
                            message=f"New category values: {preview}{suffix}",
                        )
                    )

            if (
                before.numeric_min is not None
                and after.numeric_min is not None
                and after.numeric_min < before.numeric_min
            ):
                drift.append(
                    SchemaDrift(
                        severity="info",
                        kind="range_expansion",
                        table=mapped_table,
                        column=column,
                        message=(
                            f"Numeric minimum expanded from {before.numeric_min:g} "
                            f"to {after.numeric_min:g}."
                        ),
                    )
                )
            if (
                before.numeric_max is not None
                and after.numeric_max is not None
                and after.numeric_max > before.numeric_max
            ):
                drift.append(
                    SchemaDrift(
                        severity="info",
                        kind="range_expansion",
                        table=mapped_table,
                        column=column,
                        message=(
                            f"Numeric maximum expanded from {before.numeric_max:g} "
                            f"to {after.numeric_max:g}."
                        ),
                    )
                )

    mapped_targets = set(mapping.values())
    for target_table in sorted(set(target_snapshot) - mapped_targets):
        drift.append(
            SchemaDrift(
                severity="info",
                kind="new_table",
                table=target_table,
                message=f"New table {target_table!r} was not present when the recipe was created.",
            )
        )

    return drift


def check_recipe_compatibility(
    recipe: CleaningRecipe,
    target_tables: dict[str, pd.DataFrame],
) -> RecipeCompatibilityReport:
    mapping, issues = _table_mapping(recipe, target_tables)

    for action in recipe.actions:
        mapped_table = mapping.get(action.table)
        if mapped_table is None:
            continue
        if action.column is None:
            continue
        frame = target_tables[mapped_table]
        if action.column not in frame.columns:
            issues.append(
                RecipeCompatibilityIssue(
                    table=action.table,
                    column=action.column,
                    message=(
                        f"Column {action.column!r} is missing from target table "
                        f"{mapped_table!r}."
                    ),
                )
            )

    return RecipeCompatibilityReport(
        compatible=not issues,
        table_mapping=mapping,
        issues=issues,
        drift=_schema_drift(recipe, target_tables, mapping),
    )


def _recipe_action_id(action: RecipeAction, mapped_table: str) -> str:
    payload = json.dumps(
        {
            "type": action.type.value,
            "table": mapped_table,
            "column": action.column,
            "parameters": action.parameters,
        },
        sort_keys=True,
        default=str,
    )
    return f"recipe_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def recipe_to_plan(
    input_path: str | Path,
    recipe: CleaningRecipe,
    target_tables: dict[str, pd.DataFrame],
) -> tuple[CleaningPlan, RecipeCompatibilityReport]:
    report = check_recipe_compatibility(recipe, target_tables)
    if not report.compatible:
        return (
            CleaningPlan(
                source_path=str(Path(input_path).expanduser().resolve()),
                source_sha256=file_sha256(input_path),
                actions=[],
            ),
            report,
        )

    actions: list[CleaningAction] = []
    for recipe_action in recipe.actions:
        mapped_table = report.table_mapping[recipe_action.table]
        actions.append(
            CleaningAction(
                id=_recipe_action_id(recipe_action, mapped_table),
                type=recipe_action.type,
                table=mapped_table,
                column=recipe_action.column,
                title=f"Replay {recipe_action.type.value}",
                description=(
                    "Action loaded from an approved cleaning recipe. "
                    "Review the preview before applying it to this dataset."
                ),
                parameters=recipe_action.parameters,
                confidence=1.0,
                issue_ids=[],
                origin=ActionOrigin.RECIPE,
            )
        )

    return (
        CleaningPlan(
            source_path=str(Path(input_path).expanduser().resolve()),
            source_sha256=file_sha256(input_path),
            actions=actions,
        ),
        report,
    )
