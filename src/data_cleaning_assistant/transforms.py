from __future__ import annotations

import math
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .actions import (
    ActionDecision,
    ActionExecution,
    ActionPreview,
    ActionType,
    CellChange,
    ChangeSet,
    CleaningAction,
    CleaningPlan,
)
from .inference import is_missing_like, is_null, parse_date, parse_number
from .planner import file_sha256


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _canonical_number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else float(value)


def _new_value(action: CleaningAction, value: Any) -> tuple[bool, Any]:
    if action.type == ActionType.NORMALIZE_CATEGORIES:
        mapping = {
            str(key): replacement
            for key, replacement in action.parameters.get("mapping", {}).items()
        }
        key = str(value)
        if key in mapping and value != mapping[key]:
            return True, mapping[key]
        return False, value

    if action.type == ActionType.NORMALIZE_MISSING:
        tokens = {str(item) for item in action.parameters.get("values", [])}
        if not is_null(value) and str(value) in tokens:
            return True, None
        return False, value

    if action.type == ActionType.TRIM_WHITESPACE:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped != value:
                return True, stripped
        return False, value

    if action.type == ActionType.STANDARDIZE_DATES:
        if is_missing_like(value):
            return False, value
        parsed = parse_date(value)
        if parsed is None:
            return False, value
        fmt = str(action.parameters.get("format", "%Y-%m-%d"))
        standardized = parsed.strftime(fmt)
        if value != standardized:
            return True, standardized
        return False, value

    if action.type == ActionType.COERCE_NUMERIC:
        if is_missing_like(value):
            return False, value
        parsed = parse_number(value)
        if parsed is None:
            return False, value
        number = _canonical_number(parsed)
        if value != number or type(value) is not type(number):
            return True, number
        return False, value

    raise ValueError(f"Unsupported action type: {action.type}")


def preview_action(
    tables: dict[str, pd.DataFrame],
    action: CleaningAction,
    sample_limit: int = 10,
) -> ActionPreview:
    if action.table not in tables:
        raise KeyError(f"Unknown table: {action.table}")
    frame = tables[action.table]
    if action.column not in frame.columns:
        raise KeyError(f"Unknown column {action.column!r} in table {action.table!r}")

    affected = 0
    samples: list[CellChange] = []
    for row_index, value in frame[action.column].items():
        changed, new_value = _new_value(action, value)
        if not changed:
            continue
        affected += 1
        if len(samples) < sample_limit:
            samples.append(
                CellChange(
                    table=action.table,
                    row_index=int(row_index),
                    column=action.column,
                    old_value=_jsonable(value),
                    new_value=_jsonable(new_value),
                )
            )
    return ActionPreview(
        action_id=action.id,
        affected_count=affected,
        sample_changes=samples,
    )


def apply_action(
    tables: dict[str, pd.DataFrame],
    action: CleaningAction,
) -> ActionExecution:
    if action.table not in tables:
        raise KeyError(f"Unknown table: {action.table}")
    frame = tables[action.table]
    if action.column not in frame.columns:
        raise KeyError(f"Unknown column {action.column!r} in table {action.table!r}")

    changes: list[CellChange] = []
    for row_index, value in frame[action.column].items():
        changed, new_value = _new_value(action, value)
        if not changed:
            continue
        changes.append(
            CellChange(
                table=action.table,
                row_index=int(row_index),
                column=action.column,
                old_value=_jsonable(value),
                new_value=_jsonable(new_value),
            )
        )
        frame.at[row_index, action.column] = new_value

    return ActionExecution(
        action_id=action.id,
        action_type=action.type,
        table=action.table,
        column=action.column,
        affected_count=len(changes),
        changes=changes,
    )


def apply_plan(
    path: str | Path,
    tables: dict[str, pd.DataFrame],
    plan: CleaningPlan,
    *,
    approved_ids: set[str] | None = None,
    approve_all: bool = False,
) -> tuple[dict[str, pd.DataFrame], ChangeSet]:
    resolved = Path(path).expanduser().resolve()
    actual_hash = file_sha256(resolved)
    if actual_hash != plan.source_sha256:
        raise ValueError(
            "The input file has changed since this cleaning plan was created. "
            "Generate a new plan before applying transformations."
        )

    approved_ids = approved_ids or set()
    cleaned = {name: frame.copy(deep=True) for name, frame in tables.items()}
    executions: list[ActionExecution] = []

    for action in plan.actions:
        approved = (
            approve_all
            or action.decision == ActionDecision.APPROVED
            or action.id in approved_ids
        )
        if not approved or action.decision == ActionDecision.REJECTED:
            continue
        executions.append(apply_action(cleaned, action))

    return cleaned, ChangeSet(
        source_path=str(resolved),
        source_sha256=actual_hash,
        executions=executions,
    )


def undo_changeset(
    tables: dict[str, pd.DataFrame],
    changeset: ChangeSet,
) -> dict[str, pd.DataFrame]:
    restored = {name: frame.copy(deep=True) for name, frame in tables.items()}
    for execution in reversed(changeset.executions):
        if execution.table not in restored:
            raise KeyError(f"Unknown table during undo: {execution.table}")
        frame = restored[execution.table]
        for change in reversed(execution.changes):
            if change.column not in frame.columns:
                raise KeyError(
                    f"Unknown column during undo: {execution.table}.{change.column}"
                )
            frame.at[change.row_index, change.column] = change.old_value
    return restored


def write_tables(tables: dict[str, pd.DataFrame], output_path: str | Path) -> Path:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()

    if suffix in {".xlsx", ".xlsm"}:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            for table_name, frame in tables.items():
                frame.to_excel(writer, sheet_name=table_name[:31], index=False)
        return output

    if len(tables) != 1:
        raise ValueError(
            "CSV and Parquet outputs support one table. Use .xlsx for multi-sheet datasets."
        )

    frame = next(iter(tables.values()))
    if suffix == ".csv":
        frame.to_csv(output, index=False)
        return output
    if suffix == ".parquet":
        frame.to_parquet(output, index=False)
        return output
    raise ValueError("Output must end in .csv, .xlsx, .xlsm, or .parquet")


def write_recipe(plan: CleaningPlan, output_path: str | Path) -> Path:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    approved = [
        {
            "type": action.type.value,
            "table": action.table,
            "column": action.column,
            "parameters": deepcopy(action.parameters),
        }
        for action in plan.actions
        if action.decision == ActionDecision.APPROVED
    ]
    output.write_text(
        yaml.safe_dump({"version": 1, "actions": approved}, sort_keys=False),
        encoding="utf-8",
    )
    return output
