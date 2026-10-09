from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel, Field

from .ai_candidates import build_ai_candidates, semantic_review_candidates
from .benchmark import BenchmarkCase, BenchmarkEffect, benchmark_cases
from .engine import profile_dataset
from .io import load_dataset
from .planner import propose_actions
from .semantic_stress import SemanticCase, semantic_cases


class OfflineCaseResult(BaseModel):
    suite: str
    case_id: str
    description: str
    passed: bool
    required_total: int = 0
    required_hits: int = 0
    deterministic_actions: int = 0
    semantic_review_candidates: int = 0
    unsafe_executable_proposals: int = 0
    prohibited_candidate_exposures: int = 0
    risky_semantic_candidates: int = 0
    unmatched_deterministic_actions: int = 0
    notes: list[str] = Field(default_factory=list)


class OfflineSuiteSummary(BaseModel):
    suite: str
    case_count: int
    passed_cases: int
    required_total: int
    required_hits: int
    unsafe_executable_proposals: int
    prohibited_candidate_exposures: int


class SemanticCoverageSummary(BaseModel):
    total_cases: int
    safe_cases: int
    risky_cases: int
    safe_exposed: int
    risky_exposed: int
    safe_coverage: float
    risky_exposure_rate: float
    unexpected_candidate_exposures: int = 0


class OfflineEvaluationReport(BaseModel):
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    benchmark_case_count: int
    benchmark_passed_cases: int
    benchmark_pass_rate: float
    required_total: int
    required_hits: int
    required_recall: float
    deterministic_actions: int
    semantic_review_candidates: int
    unsafe_executable_proposals: int
    prohibited_candidate_exposures: int
    risky_semantic_candidates: int
    unmatched_deterministic_actions: int
    safety_pass: bool
    coverage_pass: bool
    overall_pass: bool
    semantic_coverage: SemanticCoverageSummary
    suite_summaries: list[OfflineSuiteSummary] = Field(default_factory=list)
    cases: list[OfflineCaseResult] = Field(default_factory=list)


def _matches_any(action, effects: list[BenchmarkEffect]) -> bool:
    return any(effect.matches_action(action) for effect in effects)


def _evaluate_benchmark_case(
    case: BenchmarkCase,
    *,
    suite: str,
    folder: Path,
) -> OfflineCaseResult:
    path = folder / f"{case.id}.csv"
    path.write_text(case.csv_text, encoding="utf-8")

    plan = propose_actions(path)
    report = profile_dataset(path)
    tables = load_dataset(path).tables
    all_candidates = build_ai_candidates(report, tables)
    semantic_candidates = semantic_review_candidates(all_candidates)

    required_hits = sum(
        any(effect.matches_action(action) for action in plan.actions)
        or any(
            effect.matches_action(candidate.action)
            for candidate in semantic_candidates
        )
        for effect in case.required
    )

    unsafe_executable = sum(
        _matches_any(action, case.forbidden)
        for action in plan.actions
    )
    prohibited_exposure = sum(
        _matches_any(candidate.action, case.must_not_expose)
        for candidate in all_candidates
    )
    risky_semantic = sum(
        _matches_any(candidate.action, case.forbidden)
        for candidate in semantic_candidates
    )

    unmatched_deterministic = sum(
        not _matches_any(action, case.required)
        and not _matches_any(action, case.forbidden)
        for action in plan.actions
    )

    notes: list[str] = []
    if required_hits < len(case.required):
        notes.append(
            f"Missed {len(case.required) - required_hits} required effect(s)."
        )
    if unsafe_executable:
        notes.append(
            f"{unsafe_executable} forbidden effect(s) appeared as executable proposals."
        )
    if prohibited_exposure:
        notes.append(
            f"{prohibited_exposure} prohibited effect(s) were exposed as AI candidates."
        )
    if risky_semantic:
        notes.append(
            f"{risky_semantic} risky pair(s) were surfaced for semantic review."
        )

    passed = (
        required_hits == len(case.required)
        and unsafe_executable == 0
        and prohibited_exposure == 0
    )

    return OfflineCaseResult(
        suite=suite,
        case_id=case.id,
        description=case.description,
        passed=passed,
        required_total=len(case.required),
        required_hits=required_hits,
        deterministic_actions=len(plan.actions),
        semantic_review_candidates=len(semantic_candidates),
        unsafe_executable_proposals=unsafe_executable,
        prohibited_candidate_exposures=prohibited_exposure,
        risky_semantic_candidates=risky_semantic,
        unmatched_deterministic_actions=unmatched_deterministic,
        notes=notes,
    )


def _semantic_case_csv(case: SemanticCase) -> str:
    rows = [f"id,{case.column}"]
    index = 1
    for _ in range(case.canonical_count):
        rows.append(f"S{index:03d},{case.canonical}")
        index += 1
    for _ in range(case.variant_count):
        rows.append(f"S{index:03d},{case.variant}")
        index += 1
    return "\n".join(rows) + "\n"


def _candidate_matches_semantic_case(candidate, case: SemanticCase) -> bool:
    mapping = candidate.action.parameters.get("mapping")
    if not isinstance(mapping, dict) or len(mapping) != 1:
        return False
    source, target = next(iter(mapping.items()))
    return {str(source), str(target)} == {case.canonical, case.variant}


def _evaluate_semantic_coverage(folder: Path) -> SemanticCoverageSummary:
    safe_total = 0
    risky_total = 0
    safe_exposed = 0
    risky_exposed = 0
    unexpected_exposures = 0

    for case in semantic_cases():
        path = folder / f"semantic_{case.id}.csv"
        path.write_text(_semantic_case_csv(case), encoding="utf-8")
        report = profile_dataset(path)
        tables = load_dataset(path).tables
        candidates = semantic_review_candidates(
            build_ai_candidates(report, tables)
        )
        matched = [
            candidate
            for candidate in candidates
            if _candidate_matches_semantic_case(candidate, case)
        ]
        unexpected_exposures += len(candidates) - len(matched)

        if case.expected == "accept":
            safe_total += 1
            safe_exposed += int(bool(matched))
        else:
            risky_total += 1
            risky_exposed += int(bool(matched))

    total = safe_total + risky_total
    return SemanticCoverageSummary(
        total_cases=total,
        safe_cases=safe_total,
        risky_cases=risky_total,
        safe_exposed=safe_exposed,
        risky_exposed=risky_exposed,
        safe_coverage=(safe_exposed / safe_total) if safe_total else 1.0,
        risky_exposure_rate=(risky_exposed / risky_total) if risky_total else 0.0,
        unexpected_candidate_exposures=unexpected_exposures,
    )


def run_offline_evaluation() -> OfflineEvaluationReport:
    """Run deterministic and semantic-detector evaluation without Ollama or downloads."""

    results: list[OfflineCaseResult] = []
    with TemporaryDirectory(prefix="data-cleaner-evaluation-") as temp_dir:
        root = Path(temp_dir)

        for suite in ("core", "research", "general"):
            suite_dir = root / suite
            suite_dir.mkdir(parents=True, exist_ok=True)
            for case in benchmark_cases(suite):
                results.append(
                    _evaluate_benchmark_case(
                        case,
                        suite=suite,
                        folder=suite_dir,
                    )
                )

        semantic_dir = root / "semantic"
        semantic_dir.mkdir(parents=True, exist_ok=True)
        semantic_coverage = _evaluate_semantic_coverage(semantic_dir)

    required_total = sum(item.required_total for item in results)
    required_hits = sum(item.required_hits for item in results)
    unsafe_executable = sum(
        item.unsafe_executable_proposals for item in results
    )
    prohibited_exposure = sum(
        item.prohibited_candidate_exposures for item in results
    )
    passed_cases = sum(item.passed for item in results)

    safety_pass = (
        unsafe_executable == 0
        and prohibited_exposure == 0
        and semantic_coverage.unexpected_candidate_exposures == 0
    )
    coverage_pass = required_hits == required_total
    overall_pass = safety_pass and coverage_pass

    suite_summaries: list[OfflineSuiteSummary] = []
    for suite in ("core", "research", "general"):
        suite_results = [item for item in results if item.suite == suite]
        suite_summaries.append(
            OfflineSuiteSummary(
                suite=suite,
                case_count=len(suite_results),
                passed_cases=sum(item.passed for item in suite_results),
                required_total=sum(item.required_total for item in suite_results),
                required_hits=sum(item.required_hits for item in suite_results),
                unsafe_executable_proposals=sum(
                    item.unsafe_executable_proposals for item in suite_results
                ),
                prohibited_candidate_exposures=sum(
                    item.prohibited_candidate_exposures for item in suite_results
                ),
            )
        )

    return OfflineEvaluationReport(
        benchmark_case_count=len(results),
        benchmark_passed_cases=passed_cases,
        benchmark_pass_rate=(passed_cases / len(results)) if results else 1.0,
        required_total=required_total,
        required_hits=required_hits,
        required_recall=(required_hits / required_total) if required_total else 1.0,
        deterministic_actions=sum(item.deterministic_actions for item in results),
        semantic_review_candidates=sum(
            item.semantic_review_candidates for item in results
        ),
        unsafe_executable_proposals=unsafe_executable,
        prohibited_candidate_exposures=prohibited_exposure,
        risky_semantic_candidates=sum(
            item.risky_semantic_candidates for item in results
        ),
        unmatched_deterministic_actions=sum(
            item.unmatched_deterministic_actions for item in results
        ),
        safety_pass=safety_pass,
        coverage_pass=coverage_pass,
        overall_pass=overall_pass,
        semantic_coverage=semantic_coverage,
        suite_summaries=suite_summaries,
        cases=results,
    )
