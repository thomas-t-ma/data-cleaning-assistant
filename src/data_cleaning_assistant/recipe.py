from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
import yaml

from .actions import ActionType, CleaningAction, CleaningPlan
from .planner import file_sha256


class RecipeAction(BaseModel):
    type: ActionType
    table: str
    column: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class CleaningRecipe(BaseModel):
    version: int = 1
    actions: list[RecipeAction] = Field(default_factory=list)


class RecipeCompatibilityIssue(BaseModel):
    table: str
    column: str | None = None
    message: str


class RecipeCompatibilityReport(BaseModel):
    compatible: bool
    table_mapping: dict[str, str] = Field(default_factory=dict)
    issues: list[RecipeCompatibilityIssue] = Field(default_factory=list)


def load_recipe(path: str | Path) -> CleaningRecipe:
    payload = yaml.safe_load(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Recipe must contain a YAML mapping.")
    return CleaningRecipe.model_validate(payload)


def _table_mapping(
    recipe: CleaningRecipe,
    target_tables: dict[str, Any],
) -> tuple[dict[str, str], list[RecipeCompatibilityIssue]]:
    recipe_tables = sorted({action.table for action in recipe.actions})
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
        else:
            issues.append(
                RecipeCompatibilityIssue(
                    table=table,
                    message=f"Recipe table {table!r} is missing from the target dataset.",
                )
            )
    return mapping, issues


def check_recipe_compatibility(
    recipe: CleaningRecipe,
    target_tables: dict[str, Any],
) -> RecipeCompatibilityReport:
    mapping, issues = _table_mapping(recipe, target_tables)

    for action in recipe.actions:
        mapped_table = mapping.get(action.table)
        if mapped_table is None:
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
    target_tables: dict[str, Any],
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
