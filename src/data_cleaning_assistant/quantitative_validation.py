from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel, Field

from .golden_validation import GoldenValidationMetrics, run_golden_validation
from .planner import propose_actions
from .quantitative_stress import StressCorpusCase, quantitative_stress_cases


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class QuantitativeCaseResult(BaseModel):
    case_id: str
    domain: str
    row_count: int
    passed: bool

    expected_actions: int
    proposed_actions: int
    action_true_positives: int
    action_false_positives: int
    action_false_negatives: int
    action_precision: float
    action_recall: float
    action_f1: float

    expected_cell_edits: int
    actual_cell_edits: int
    cell_true_positives: int
    cell_false_positives: int
    cell_false_negatives: int
    cell_precision: float
    cell_recall: float
    cell_f1: float

    structural_mismatches: int
    protected_columns_exposed: list[str] = Field(default_factory=list)
    missing_action_keys: list[str] = Field(default_factory=list)
    unexpected_action_keys: list[str] = Field(default_factory=list)
    golden_metrics: GoldenValidationMetrics


class QuantitativeStressReport(BaseModel):
    passed: bool
    case_count: int
    passed_cases: int
    total_rows: int

    expected_actions: int
    proposed_actions: int
    action_true_positives: int
    action_false_positives: int
    action_false_negatives: int
    action_precision: float
    action_recall: float
    action_f1: float

    expected_cell_edits: int
    actual_cell_edits: int
    cell_true_positives: int
    cell_false_positives: int
    cell_false_negatives: int
    cell_precision: float
    cell_recall: float
    cell_f1: float

    cases: list[QuantitativeCaseResult] = Field(default_factory=list)


def run_quantitative_cases(
    cases: list[StressCorpusCase],
    *,
    temp_prefix: str = "data-cleaner-quantitative-",
) -> QuantitativeStressReport:
    """Measure executable-action and edited-cell precision/recall for golden cases."""
    results: list[QuantitativeCaseResult] = []

    with TemporaryDirectory(prefix=temp_prefix) as temp_dir:
        root = Path(temp_dir)

        for case in cases:
            source = root / f"{case.id}.dirty.csv"
            golden = root / f"{case.id}.golden.csv"
            source.write_text(case.dirty_csv, encoding="utf-8")
            golden.write_text(case.golden_csv, encoding="utf-8")

            plan = propose_actions(source)
            golden_report = run_golden_validation(
                source,
                golden,
                difference_limit=25,
            )
            metrics = golden_report.metrics

            expected_keys = {item.key for item in case.expected_actions}
            proposed_keys = [
                (action.type.value, action.column)
                for action in plan.actions
            ]

            matched_expected: set[tuple[str, str]] = set()
            unexpected_records: list[tuple[str, str | None]] = []
            for action_key in proposed_keys:
                if (
                    action_key[1] is not None
                    and (action_key[0], action_key[1]) in expected_keys
                    and (action_key[0], action_key[1]) not in matched_expected
                ):
                    matched_expected.add((action_key[0], action_key[1]))
                else:
                    unexpected_records.append(action_key)

            action_tp = len(matched_expected)
            action_fp = len(unexpected_records)
            action_fn = len(expected_keys - matched_expected)
            action_precision = _safe_ratio(
                action_tp,
                action_tp + action_fp,
            )
            action_recall = _safe_ratio(
                action_tp,
                action_tp + action_fn,
            )

            cell_tp = metrics.matched_expected_changes
            cell_fp = max(0, metrics.actual_changed_cells - cell_tp)
            cell_fn = max(0, metrics.expected_changed_cells - cell_tp)
            cell_precision = _safe_ratio(
                cell_tp,
                cell_tp + cell_fp,
            )
            cell_recall = _safe_ratio(
                cell_tp,
                cell_tp + cell_fn,
            )

            protected = set(case.protected_columns)
            exposed = sorted(
                {
                    str(action.column)
                    for action in plan.actions
                    if action.column is not None
                    and action.column in protected
                }
            )

            missing = sorted(
                f"{action_type}:{column}"
                for action_type, column in expected_keys - matched_expected
            )
            unexpected = sorted(
                f"{action_type}:{column or '(table)'}"
                for action_type, column in unexpected_records
            )

            passed = (
                metrics.structural_mismatches == 0
                and metrics.source_alignment_available
                and golden_report.passed
                and action_fp == 0
                and action_fn == 0
                and not exposed
            )

            results.append(
                QuantitativeCaseResult(
                    case_id=case.id,
                    domain=case.domain,
                    row_count=case.row_count,
                    passed=passed,
                    expected_actions=len(expected_keys),
                    proposed_actions=len(proposed_keys),
                    action_true_positives=action_tp,
                    action_false_positives=action_fp,
                    action_false_negatives=action_fn,
                    action_precision=action_precision,
                    action_recall=action_recall,
                    action_f1=_f1(action_precision, action_recall),
                    expected_cell_edits=metrics.expected_changed_cells,
                    actual_cell_edits=metrics.actual_changed_cells,
                    cell_true_positives=cell_tp,
                    cell_false_positives=cell_fp,
                    cell_false_negatives=cell_fn,
                    cell_precision=cell_precision,
                    cell_recall=cell_recall,
                    cell_f1=_f1(cell_precision, cell_recall),
                    structural_mismatches=metrics.structural_mismatches,
                    protected_columns_exposed=exposed,
                    missing_action_keys=missing,
                    unexpected_action_keys=unexpected,
                    golden_metrics=metrics,
                )
            )

    expected_actions = sum(item.expected_actions for item in results)
    proposed_actions = sum(item.proposed_actions for item in results)
    action_tp = sum(item.action_true_positives for item in results)
    action_fp = sum(item.action_false_positives for item in results)
    action_fn = sum(item.action_false_negatives for item in results)
    action_precision = _safe_ratio(action_tp, action_tp + action_fp)
    action_recall = _safe_ratio(action_tp, action_tp + action_fn)

    expected_cells = sum(item.expected_cell_edits for item in results)
    actual_cells = sum(item.actual_cell_edits for item in results)
    cell_tp = sum(item.cell_true_positives for item in results)
    cell_fp = sum(item.cell_false_positives for item in results)
    cell_fn = sum(item.cell_false_negatives for item in results)
    cell_precision = _safe_ratio(cell_tp, cell_tp + cell_fp)
    cell_recall = _safe_ratio(cell_tp, cell_tp + cell_fn)

    passed_cases = sum(item.passed for item in results)
    return QuantitativeStressReport(
        passed=passed_cases == len(results),
        case_count=len(results),
        passed_cases=passed_cases,
        total_rows=sum(item.row_count for item in results),
        expected_actions=expected_actions,
        proposed_actions=proposed_actions,
        action_true_positives=action_tp,
        action_false_positives=action_fp,
        action_false_negatives=action_fn,
        action_precision=action_precision,
        action_recall=action_recall,
        action_f1=_f1(action_precision, action_recall),
        expected_cell_edits=expected_cells,
        actual_cell_edits=actual_cells,
        cell_true_positives=cell_tp,
        cell_false_positives=cell_fp,
        cell_false_negatives=cell_fn,
        cell_precision=cell_precision,
        cell_recall=cell_recall,
        cell_f1=_f1(cell_precision, cell_recall),
        cases=results,
    )


def run_quantitative_stress_validation() -> QuantitativeStressReport:
    """Run the fixed 2,000-row quantitative release baseline."""
    return run_quantitative_cases(quantitative_stress_cases())
