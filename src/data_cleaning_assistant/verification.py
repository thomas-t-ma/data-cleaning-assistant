from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from pydantic import BaseModel, Field

from .actions import ChangeSet
from .inference import is_null
from .io import load_dataset
from .planner import file_sha256
from .recipe import load_recipe, recipe_to_plan
from .transforms import (
    apply_plan,
    undo_changeset,
    write_cleaned_dataset,
)


class VerificationCheck(BaseModel):
    name: str
    passed: bool
    detail: str


class VerificationMetrics(BaseModel):
    execution_count: int = 0
    recipe_action_count: int = 0
    cell_change_events: int = 0
    unique_touched_cells: int = 0
    repeated_touch_events: int = 0
    row_deletions: int = 0
    net_different_cells: int | None = None
    replay_byte_identical: bool | None = None


class RunVerificationReport(BaseModel):
    source_path: str
    cleaned_path: str
    audit_path: str
    recipe_path: str
    passed: bool
    checks: list[VerificationCheck] = Field(default_factory=list)
    metrics: VerificationMetrics = Field(default_factory=VerificationMetrics)


def _value_equal(left: Any, right: Any) -> bool:
    if is_null(left) and is_null(right):
        return True
    try:
        result = left == right
        if hasattr(result, "item"):
            result = result.item()
        return bool(result)
    except (TypeError, ValueError):
        return repr(left) == repr(right)


def _table_pairs(
    left: dict[str, Any],
    right: dict[str, Any],
) -> tuple[list[tuple[str, str]], str | None]:
    if set(left) == set(right):
        return [(name, name) for name in left], None
    if len(left) == 1 and len(right) == 1:
        return [(next(iter(left)), next(iter(right)))], None
    return [], (
        "Table/sheet names differ and cannot be unambiguously matched: "
        f"{sorted(left)} vs {sorted(right)}"
    )


def _tables_equal(
    left: dict[str, Any],
    right: dict[str, Any],
) -> tuple[bool, str]:
    pairs, error = _table_pairs(left, right)
    if error:
        return False, error

    for left_name, right_name in pairs:
        left_frame = left[left_name]
        right_frame = right[right_name]
        if list(left_frame.columns) != list(right_frame.columns):
            return (
                False,
                f"Column layout differs for {left_name!r}: "
                f"{list(left_frame.columns)!r} vs {list(right_frame.columns)!r}",
            )
        if left_frame.shape != right_frame.shape:
            return (
                False,
                f"Shape differs for {left_name!r}: "
                f"{left_frame.shape!r} vs {right_frame.shape!r}",
            )

        for row_position in range(len(left_frame)):
            for column in left_frame.columns:
                left_value = left_frame.iloc[row_position][column]
                right_value = right_frame.iloc[row_position][column]
                if not _value_equal(left_value, right_value):
                    return (
                        False,
                        f"First differing value at {left_name!r} row "
                        f"{row_position}, column {column!r}: "
                        f"{left_value!r} vs {right_value!r}",
                    )
    return True, "Logical table values are equivalent."


def _net_different_cells(
    source_tables: dict[str, Any],
    cleaned_tables: dict[str, Any],
) -> int | None:
    pairs, error = _table_pairs(source_tables, cleaned_tables)
    if error:
        return None

    total = 0
    for source_name, cleaned_name in pairs:
        source = source_tables[source_name]
        cleaned = cleaned_tables[cleaned_name]
        if list(source.columns) != list(cleaned.columns) or source.shape != cleaned.shape:
            return None
        for row_position in range(len(source)):
            for column in source.columns:
                if not _value_equal(
                    source.iloc[row_position][column],
                    cleaned.iloc[row_position][column],
                ):
                    total += 1
    return total


def _check(
    checks: list[VerificationCheck],
    name: str,
    passed: bool,
    detail: str,
) -> None:
    checks.append(
        VerificationCheck(
            name=name,
            passed=bool(passed),
            detail=detail,
        )
    )


def verify_run(
    source_path: str | Path,
    cleaned_path: str | Path,
    audit_path: str | Path,
    recipe_path: str | Path,
) -> RunVerificationReport:
    """Independently verify one completed cleaning run and its provenance."""

    source = Path(source_path).expanduser().resolve()
    cleaned = Path(cleaned_path).expanduser().resolve()
    audit = Path(audit_path).expanduser().resolve()
    recipe_file = Path(recipe_path).expanduser().resolve()

    changeset = ChangeSet.model_validate_json(audit.read_text(encoding="utf-8"))
    recipe = load_recipe(recipe_file)
    source_loaded = load_dataset(source)
    cleaned_loaded = load_dataset(cleaned)

    checks: list[VerificationCheck] = []

    source_hash = file_sha256(source)
    _check(
        checks,
        "source_unchanged",
        source_hash == changeset.source_sha256,
        (
            f"Current source SHA256 {source_hash}; "
            f"audit source SHA256 {changeset.source_sha256}."
        ),
    )

    cleaned_hash = file_sha256(cleaned)
    result_hash_present = bool(changeset.result_sha256)
    _check(
        checks,
        "cleaned_matches_audit",
        result_hash_present and cleaned_hash == changeset.result_sha256,
        (
            f"Current cleaned SHA256 {cleaned_hash}; "
            f"audit result SHA256 {changeset.result_sha256 or '<missing>'}."
        ),
    )

    execution_consistent = all(
        execution.affected_count
        == len(execution.changes) + len(execution.deleted_rows)
        for execution in changeset.executions
    )
    _check(
        checks,
        "audit_counts_consistent",
        execution_consistent,
        (
            "Every execution affected_count equals its recorded cell changes "
            "plus deleted rows."
            if execution_consistent
            else "At least one audit execution has an inconsistent affected_count."
        ),
    )

    recipe_count_matches = len(recipe.actions) == len(changeset.executions)
    _check(
        checks,
        "recipe_action_count_matches_audit",
        recipe_count_matches,
        (
            f"Recipe actions: {len(recipe.actions)}; "
            f"audit executions: {len(changeset.executions)}."
        ),
    )

    sequence_matches = recipe_count_matches and all(
        recipe_action.type == execution.action_type
        and recipe_action.table == execution.table
        and recipe_action.column == execution.column
        for recipe_action, execution in zip(
            recipe.actions,
            changeset.executions,
            strict=True,
        )
    )
    _check(
        checks,
        "recipe_action_sequence_matches_audit",
        sequence_matches,
        (
            "Recipe action type/table/column sequence matches the audit."
            if sequence_matches
            else "Recipe action sequence differs from the audit executions."
        ),
    )

    plan, compatibility = recipe_to_plan(
        source,
        recipe,
        source_loaded.tables,
    )
    _check(
        checks,
        "recipe_compatible_with_source",
        compatibility.compatible,
        (
            "Recipe is compatible with the current source schema."
            if compatibility.compatible
            else "; ".join(issue.message for issue in compatibility.issues)
        ),
    )

    replay_equal = False
    replay_detail = "Recipe replay was not attempted because compatibility failed."
    replay_byte_identical: bool | None = None
    if compatibility.compatible:
        try:
            replayed_tables, _replay_changeset = apply_plan(
                source,
                source_loaded.tables,
                plan,
                approve_all=True,
            )
            with TemporaryDirectory(prefix="data-cleaner-verify-replay-") as tmpdir:
                replay_path = Path(tmpdir) / f"replayed{cleaned.suffix}"
                write_cleaned_dataset(
                    source,
                    source_loaded.tables,
                    replayed_tables,
                    replay_path,
                )
                replay_loaded = load_dataset(replay_path)
                replay_equal, replay_detail = _tables_equal(
                    replay_loaded.tables,
                    cleaned_loaded.tables,
                )
                replay_byte_identical = replay_path.read_bytes() == cleaned.read_bytes()
        except Exception as exc:  # verification should report, not crash, on replay failure
            replay_detail = f"Recipe replay failed: {type(exc).__name__}: {exc}"

    _check(
        checks,
        "recipe_replay_equivalent",
        replay_equal,
        replay_detail,
    )

    undo_equal = False
    undo_detail = "Undo verification was not completed."
    try:
        restored_tables = undo_changeset(
            cleaned_loaded.tables,
            changeset,
        )
        inserted_rows: dict[str, list] = {}
        for execution in changeset.executions:
            if execution.deleted_rows:
                inserted_rows.setdefault(execution.table, []).extend(
                    execution.deleted_rows
                )

        with TemporaryDirectory(prefix="data-cleaner-verify-undo-") as tmpdir:
            restored_path = Path(tmpdir) / f"restored{source.suffix}"
            write_cleaned_dataset(
                cleaned,
                cleaned_loaded.tables,
                restored_tables,
                restored_path,
                inserted_rows=inserted_rows,
            )
            restored_loaded = load_dataset(restored_path)
            undo_equal, undo_detail = _tables_equal(
                restored_loaded.tables,
                source_loaded.tables,
            )
    except Exception as exc:  # verification should report, not crash, on undo failure
        undo_detail = f"Audit undo failed: {type(exc).__name__}: {exc}"

    _check(
        checks,
        "audit_undo_restores_source",
        undo_equal,
        undo_detail,
    )

    touched = [
        (change.table, change.row_index, change.column)
        for execution in changeset.executions
        for change in execution.changes
    ]
    unique_touched = len(set(touched))
    cell_change_events = len(touched)

    metrics = VerificationMetrics(
        execution_count=len(changeset.executions),
        recipe_action_count=len(recipe.actions),
        cell_change_events=cell_change_events,
        unique_touched_cells=unique_touched,
        repeated_touch_events=cell_change_events - unique_touched,
        row_deletions=sum(
            len(execution.deleted_rows)
            for execution in changeset.executions
        ),
        net_different_cells=_net_different_cells(
            source_loaded.tables,
            cleaned_loaded.tables,
        ),
        replay_byte_identical=replay_byte_identical,
    )

    return RunVerificationReport(
        source_path=str(source),
        cleaned_path=str(cleaned),
        audit_path=str(audit),
        recipe_path=str(recipe_file),
        passed=all(check.passed for check in checks),
        checks=checks,
        metrics=metrics,
    )
