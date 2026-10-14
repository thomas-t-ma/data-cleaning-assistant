from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from pydantic import BaseModel, Field

from .inference import is_null
from .io import load_dataset
from .planner import propose_actions
from .transforms import apply_plan, write_cleaned_dataset


class GoldenDifference(BaseModel):
    table: str
    row_position: int | None = None
    column: str | None = None
    kind: str
    source: Any = None
    expected: Any = None
    actual: Any = None


class GoldenValidationMetrics(BaseModel):
    planned_actions: int = 0
    executed_actions: int = 0
    source_alignment_available: bool = True
    expected_removed_rows: int = 0
    actual_removed_rows: int = 0
    expected_changed_cells: int = 0
    actual_changed_cells: int = 0
    matched_expected_changes: int = 0
    missed_expected_changes: int = 0
    unexpected_changes: int = 0
    wrong_replacements: int = 0
    remaining_different_cells: int = 0
    structural_mismatches: int = 0


class GoldenValidationReport(BaseModel):
    source_path: str
    expected_path: str
    passed: bool
    metrics: GoldenValidationMetrics
    differences: list[GoldenDifference] = Field(default_factory=list)
    planned_action_summaries: list[str] = Field(default_factory=list)


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
    source: dict[str, Any],
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> tuple[list[tuple[str, str, str]], list[GoldenDifference]]:
    if set(source) == set(actual) == set(expected):
        return [(name, name, name) for name in source], []

    if len(source) == len(actual) == len(expected) == 1:
        return [
            (
                next(iter(source)),
                next(iter(actual)),
                next(iter(expected)),
            )
        ], []

    detail = GoldenDifference(
        table="(dataset)",
        kind="table_layout_mismatch",
        source=sorted(source),
        actual=sorted(actual),
        expected=sorted(expected),
    )
    return [], [detail]


def _compare_tables(
    source: dict[str, Any],
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    difference_limit: int,
) -> tuple[GoldenValidationMetrics, list[GoldenDifference]]:
    metrics = GoldenValidationMetrics()
    differences: list[GoldenDifference] = []

    pairs, layout_differences = _table_pairs(source, actual, expected)
    if layout_differences:
        metrics.structural_mismatches += len(layout_differences)
        return metrics, layout_differences[:difference_limit]

    for source_name, actual_name, expected_name in pairs:
        source_frame = source[source_name]
        actual_frame = actual[actual_name]
        expected_frame = expected[expected_name]

        if (
            list(source_frame.columns) != list(actual_frame.columns)
            or list(source_frame.columns) != list(expected_frame.columns)
        ):
            metrics.structural_mismatches += 1
            if len(differences) < difference_limit:
                differences.append(
                    GoldenDifference(
                        table=source_name,
                        kind="column_layout_mismatch",
                        source=list(source_frame.columns),
                        actual=list(actual_frame.columns),
                        expected=list(expected_frame.columns),
                    )
                )
            continue

        # The output must structurally match the golden file. The source is
        # allowed to have extra rows because duplicate-row removal is a valid
        # cleaning action.
        if actual_frame.shape != expected_frame.shape:
            metrics.structural_mismatches += 1
            if len(differences) < difference_limit:
                differences.append(
                    GoldenDifference(
                        table=source_name,
                        kind="shape_mismatch",
                        source=list(source_frame.shape),
                        actual=list(actual_frame.shape),
                        expected=list(expected_frame.shape),
                    )
                )
            continue

        source_aligned = source_frame.shape == expected_frame.shape
        if not source_aligned:
            metrics.source_alignment_available = False
            if len(source_frame) < len(expected_frame):
                metrics.structural_mismatches += 1
                if len(differences) < difference_limit:
                    differences.append(
                        GoldenDifference(
                            table=source_name,
                            kind="expected_row_insertion",
                            source=list(source_frame.shape),
                            actual=list(actual_frame.shape),
                            expected=list(expected_frame.shape),
                        )
                    )
                continue
            metrics.expected_removed_rows += len(source_frame) - len(expected_frame)
            metrics.actual_removed_rows += len(source_frame) - len(actual_frame)

        for row_position in range(len(expected_frame)):
            for column in expected_frame.columns:
                actual_value = actual_frame.iloc[row_position][column]
                expected_value = expected_frame.iloc[row_position][column]
                actual_matches_expected = _value_equal(actual_value, expected_value)

                if not source_aligned:
                    if actual_matches_expected:
                        continue
                    metrics.remaining_different_cells += 1
                    if len(differences) < difference_limit:
                        differences.append(
                            GoldenDifference(
                                table=source_name,
                                row_position=row_position,
                                column=str(column),
                                kind="output_mismatch_after_row_change",
                                expected=expected_value,
                                actual=actual_value,
                            )
                        )
                    continue

                source_value = source_frame.iloc[row_position][column]
                expected_changed = not _value_equal(source_value, expected_value)
                actual_changed = not _value_equal(source_value, actual_value)

                if expected_changed:
                    metrics.expected_changed_cells += 1
                if actual_changed:
                    metrics.actual_changed_cells += 1

                if actual_matches_expected:
                    if expected_changed:
                        metrics.matched_expected_changes += 1
                    continue

                metrics.remaining_different_cells += 1
                if expected_changed and not actual_changed:
                    kind = "missed_expected_change"
                    metrics.missed_expected_changes += 1
                elif not expected_changed and actual_changed:
                    kind = "unexpected_change"
                    metrics.unexpected_changes += 1
                else:
                    kind = "wrong_replacement"
                    metrics.wrong_replacements += 1

                if len(differences) < difference_limit:
                    differences.append(
                        GoldenDifference(
                            table=source_name,
                            row_position=row_position,
                            column=str(column),
                            kind=kind,
                            source=source_value,
                            expected=expected_value,
                            actual=actual_value,
                        )
                    )

    return metrics, differences

def run_golden_validation(
    source_path: str,
    expected_path: str,
    *,
    difference_limit: int = 50,
) -> GoldenValidationReport:
    """Run deterministic proposals against a human-approved golden output.

    This is evaluation mode: every non-rejected deterministic proposal is
    approved so the golden output, rather than an interactive reviewer, is the
    authority for whether each proposed transformation was correct.
    """
    source_loaded = load_dataset(source_path)
    expected_loaded = load_dataset(expected_path)
    plan = propose_actions(source_path)

    actual_tables, changeset = apply_plan(
        source_path,
        source_loaded.tables,
        plan,
        approve_all=True,
    )

    # Golden validation must judge the artifact a user would actually receive,
    # not transient in-memory dtypes. For example, numeric coercion may turn
    # "10" into integer 10 internally, but a CSV round-trip represents both as
    # the same textual value. Serialize through the normal writer and reload.
    expected_suffix = Path(expected_path).suffix
    with TemporaryDirectory(prefix="data-cleaner-golden-") as temp_dir:
        actual_path = Path(temp_dir) / f"actual{expected_suffix}"
        write_cleaned_dataset(
            source_path,
            source_loaded.tables,
            actual_tables,
            actual_path,
        )
        actual_loaded = load_dataset(actual_path)

        metrics, differences = _compare_tables(
            source_loaded.tables,
            actual_loaded.tables,
            expected_loaded.tables,
            difference_limit=difference_limit,
        )
    metrics.planned_actions = len(plan.actions)
    metrics.executed_actions = len(changeset.executions)

    summaries = [
        f"{action.type.value}:{action.table}.{action.column or '(table)'}"
        for action in plan.actions
    ]

    passed = (
        metrics.structural_mismatches == 0
        and metrics.remaining_different_cells == 0
    )
    return GoldenValidationReport(
        source_path=str(source_path),
        expected_path=str(expected_path),
        passed=passed,
        metrics=metrics,
        differences=differences,
        planned_action_summaries=summaries,
    )
