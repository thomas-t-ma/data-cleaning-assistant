from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .actions import ActionType, CleaningAction
from .ai_models import CandidateDecision
from .ai_planner import propose_ai_plan
from .ai_provider import AIProviderError, StructuredAIProvider


class BenchmarkEffect(BaseModel):
    type: ActionType
    column: str | None = None
    parameters: dict[str, Any] | None = None

    def matches_action(self, action: CleaningAction) -> bool:
        if action.type != self.type or action.column != self.column:
            return False
        if self.parameters is None:
            return True
        return _normalized_parameters(action.parameters) == _normalized_parameters(self.parameters)


class BenchmarkCase(BaseModel):
    id: str
    description: str
    csv_text: str
    required: list[BenchmarkEffect] = Field(default_factory=list)
    forbidden: list[BenchmarkEffect] = Field(default_factory=list)


class BenchmarkCaseResult(BaseModel):
    case_id: str
    description: str
    run_index: int
    latency_seconds: float
    raw_proposals: int
    accepted_actions: int
    validator_rejections: int
    required_total: int
    required_hits: int
    required_misses: int
    forbidden_raw_proposals: int
    forbidden_accepted: int
    forbidden_rejected: int
    unexpected_accepted: int
    passed: bool
    model_summary: str = ""
    accepted: list[dict[str, Any]] = Field(default_factory=list)
    rejected: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


class BenchmarkSummary(BaseModel):
    model: str
    provider: str
    suite: str = "custom"
    runs_per_case: int
    case_runs: int
    passed_case_runs: int
    pass_rate: float
    required_total: int
    required_hits: int
    required_recall: float
    accepted_total: int
    accepted_correct: int
    accepted_precision: float
    forbidden_raw_proposals: int
    forbidden_accepted: int
    forbidden_rejected: int
    forbidden_validator_catch_rate: float | None = None
    validator_rejections: int
    mean_latency_seconds: float
    results: list[BenchmarkCaseResult] = Field(default_factory=list)


def _normalized_parameters(parameters: dict[str, Any]) -> str:
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"), default=str)


def _action_record(action: CleaningAction) -> dict[str, Any]:
    return {
        "type": action.type.value,
        "column": action.column,
        "parameters": action.parameters,
        "confidence": action.confidence,
    }


def core_benchmark_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            id="fuzzy_typo_merge",
            description=(
                "A clear one-character category typo should be merged to the correctly spelled "
                "observed label."
            ),
            csv_text=(
                "patient_id,group\n"
                "P001,Baseline\n"
                "P002,Basline\n"
                "P003,Baseline\n"
                "P004,Baseline\n"
            ),
            required=[
                BenchmarkEffect(
                    type=ActionType.NORMALIZE_CATEGORIES,
                    column="group",
                    parameters={"mapping": {"Basline": "Baseline"}},
                )
            ],
        ),
        BenchmarkCase(
            id="hidden_missing_token",
            description="An explicit N/A token should be proposed for missing-value normalization.",
            csv_text=(
                "patient_id,status\n"
                "P001,Active\n"
                "P002,N/A\n"
                "P003,Inactive\n"
                "P004,Active\n"
            ),
            required=[
                BenchmarkEffect(
                    type=ActionType.NORMALIZE_MISSING,
                    column="status",
                    parameters={"values": ["N/A"]},
                )
            ],
        ),
        BenchmarkCase(
            id="mixed_numeric_column",
            description=(
                "A mostly numeric non-identifier column with one text value should preserve the "
                "text value while coercing parseable numbers."
            ),
            csv_text=(
                "patient_id,age\n"
                "P001,30\n"
                "P002,40\n"
                "P003,forty\n"
                "P004,50\n"
                "P005,60\n"
            ),
            required=[
                BenchmarkEffect(
                    type=ActionType.COERCE_NUMERIC,
                    column="age",
                    parameters={"unparseable": "keep"},
                )
            ],
        ),
        BenchmarkCase(
            id="mixed_date_formats",
            description="Parseable date variants should be standardized while bad-date remains unchanged.",
            csv_text=(
                "patient_id,visit_date\n"
                "P001,2026-01-01\n"
                "P002,01/03/2026\n"
                "P003,\"January 5, 2026\"\n"
                "P004,bad-date\n"
            ),
            required=[
                BenchmarkEffect(
                    type=ActionType.STANDARDIZE_DATES,
                    column="visit_date",
                    parameters={"format": "%Y-%m-%d", "unparseable": "keep"},
                )
            ],
        ),
        BenchmarkCase(
            id="surrounding_whitespace",
            description="Leading/trailing whitespace should be trimmed without changing internal text.",
            csv_text=(
                "patient_id,status\n"
                "P001, Active \n"
                "P002,Inactive\n"
                "P003,Pending\n"
            ),
            required=[
                BenchmarkEffect(
                    type=ActionType.TRIM_WHITESPACE,
                    column="status",
                    parameters={},
                )
            ],
        ),
        BenchmarkCase(
            id="semantically_distinct_similar_categories",
            description=(
                "Stage II and Stage III are lexically similar but scientifically distinct and "
                "must not be merged."
            ),
            csv_text=(
                "patient_id,stage\n"
                "P001,Stage II\n"
                "P002,Stage III\n"
                "P003,Stage II\n"
                "P004,Stage III\n"
            ),
            forbidden=[
                BenchmarkEffect(
                    type=ActionType.NORMALIZE_CATEGORIES,
                    column="stage",
                    parameters=None,
                )
            ],
        ),
        BenchmarkCase(
            id="ambiguous_exact_duplicates",
            description=(
                "Exact repeated observations are ambiguous without provenance and must remain a "
                "human-review decision, not an AI candidate."
            ),
            csv_text=(
                "patient_id,status\n"
                "P001,Active\n"
                "P001,Active\n"
                "P002,Inactive\n"
            ),
            forbidden=[
                BenchmarkEffect(
                    type=ActionType.REMOVE_EXACT_DUPLICATES,
                    column=None,
                    parameters=None,
                )
            ],
        ),
        BenchmarkCase(
            id="identifier_numeric_coercion",
            description="Identifier-like columns must never be exposed for numeric coercion.",
            csv_text=(
                "patient_id,value\n"
                "001,10\n"
                "002,20\n"
                "bad,30\n"
                "004,40\n"
                "005,50\n"
            ),
            forbidden=[
                BenchmarkEffect(
                    type=ActionType.COERCE_NUMERIC,
                    column="patient_id",
                    parameters=None,
                )
            ],
        ),
    ]



def research_benchmark_cases() -> list[BenchmarkCase]:
    """Harder research-oriented cases that stress semantic accept/defer judgment."""
    return [
        BenchmarkCase(
            id="research_visit_zero_padding",
            description=(
                "Visit 1 and Visit 01 are formatting variants of the same visit label and "
                "should be normalized."
            ),
            csv_text=(
                "subject_id,visit_label\n"
                "S001,Visit 1\n"
                "S002,Visit 1\n"
                "S003,Visit 01\n"
                "S004,Visit 1\n"
            ),
            required=[
                BenchmarkEffect(
                    type=ActionType.NORMALIZE_CATEGORIES,
                    column="visit_label",
                    parameters={"mapping": {"Visit 01": "Visit 1"}},
                )
            ],
        ),
        BenchmarkCase(
            id="research_specimen_typo",
            description=(
                "Peripheral Blod is a clear typo of Peripheral Blood and should be normalized."
            ),
            csv_text=(
                "specimen_id,specimen_source\n"
                "SP001,Peripheral Blood\n"
                "SP002,Peripheral Blood\n"
                "SP003,Peripheral Blod\n"
                "SP004,Peripheral Blood\n"
            ),
            required=[
                BenchmarkEffect(
                    type=ActionType.NORMALIZE_CATEGORIES,
                    column="specimen_source",
                    parameters={"mapping": {"Peripheral Blod": "Peripheral Blood"}},
                )
            ],
        ),
        BenchmarkCase(
            id="research_cohort_zero_padding",
            description=(
                "Cohort 1 and Cohort 01 are formatting variants of the same cohort label."
            ),
            csv_text=(
                "subject_id,cohort\n"
                "S001,Cohort 1\n"
                "S002,Cohort 1\n"
                "S003,Cohort 01\n"
                "S004,Cohort 1\n"
            ),
            required=[
                BenchmarkEffect(
                    type=ActionType.NORMALIZE_CATEGORIES,
                    column="cohort",
                    parameters={"mapping": {"Cohort 01": "Cohort 1"}},
                )
            ],
        ),
        BenchmarkCase(
            id="research_lab_numeric_with_censoring",
            description=(
                "A lab-value column is mostly numeric but contains one censored value '<5'. "
                "Parseable values may be coerced while the censored value remains unchanged."
            ),
            csv_text=(
                "subject_id,lab_value\n"
                "S001,12.4\n"
                "S002,18.1\n"
                "S003,<5\n"
                "S004,9.7\n"
                "S005,15.0\n"
            ),
            required=[
                BenchmarkEffect(
                    type=ActionType.COERCE_NUMERIC,
                    column="lab_value",
                    parameters={"unparseable": "keep"},
                )
            ],
        ),
        BenchmarkCase(
            id="research_specimen_whitespace",
            description=(
                "A specimen-source value with only surrounding whitespace should be trimmed."
            ),
            csv_text=(
                "specimen_id,specimen_source\n"
                "SP001, Bone Marrow \n"
                "SP002,Peripheral Blood\n"
                "SP003,Serum\n"
            ),
            required=[
                BenchmarkEffect(
                    type=ActionType.TRIM_WHITESPACE,
                    column="specimen_source",
                    parameters={},
                )
            ],
        ),
        BenchmarkCase(
            id="research_dose_level_distinct",
            description=(
                "Dose 5 mg and Dose 50 mg are lexically similar but clinically distinct dose levels."
            ),
            csv_text=(
                "subject_id,dose_level\n"
                "S001,Dose 5 mg\n"
                "S002,Dose 50 mg\n"
                "S003,Dose 5 mg\n"
                "S004,Dose 50 mg\n"
            ),
            forbidden=[
                BenchmarkEffect(
                    type=ActionType.NORMALIZE_CATEGORIES,
                    column="dose_level",
                    parameters=None,
                )
            ],
        ),
        BenchmarkCase(
            id="research_dose_unit_distinct",
            description=(
                "Dose 5 mg and Dose 5 mcg differ by unit and must never be merged."
            ),
            csv_text=(
                "subject_id,dose_text\n"
                "S001,Dose 5 mg\n"
                "S002,Dose 5 mcg\n"
                "S003,Dose 5 mg\n"
                "S004,Dose 5 mcg\n"
            ),
            forbidden=[
                BenchmarkEffect(
                    type=ActionType.NORMALIZE_CATEGORIES,
                    column="dose_text",
                    parameters=None,
                )
            ],
        ),
        BenchmarkCase(
            id="research_timepoint_distinct",
            description=(
                "Post-dose Day 1 and Post-dose Day 10 are distinct study timepoints."
            ),
            csv_text=(
                "subject_id,timepoint\n"
                "S001,Post-dose Day 1\n"
                "S002,Post-dose Day 10\n"
                "S003,Post-dose Day 1\n"
                "S004,Post-dose Day 10\n"
            ),
            forbidden=[
                BenchmarkEffect(
                    type=ActionType.NORMALIZE_CATEGORIES,
                    column="timepoint",
                    parameters=None,
                )
            ],
        ),
        BenchmarkCase(
            id="research_treatment_arm_distinct",
            description=(
                "Treatment Arm A and Treatment Arm B are distinct randomized groups."
            ),
            csv_text=(
                "subject_id,treatment_arm\n"
                "S001,Treatment Arm A\n"
                "S002,Treatment Arm B\n"
                "S003,Treatment Arm A\n"
                "S004,Treatment Arm B\n"
            ),
            forbidden=[
                BenchmarkEffect(
                    type=ActionType.NORMALIZE_CATEGORIES,
                    column="treatment_arm",
                    parameters=None,
                )
            ],
        ),
        BenchmarkCase(
            id="research_specimen_type_distinct",
            description=(
                "Specimen Type A and Specimen Type B are distinct specimen categories."
            ),
            csv_text=(
                "specimen_id,specimen_type\n"
                "SP001,Specimen Type A\n"
                "SP002,Specimen Type B\n"
                "SP003,Specimen Type A\n"
                "SP004,Specimen Type B\n"
            ),
            forbidden=[
                BenchmarkEffect(
                    type=ActionType.NORMALIZE_CATEGORIES,
                    column="specimen_type",
                    parameters=None,
                )
            ],
        ),
        BenchmarkCase(
            id="research_unknown_outcome_category",
            description=(
                "Unknown is an explicit outcome category here; converting it to null would erase "
                "a meaningful coded state, so the missing-token candidate should be deferred."
            ),
            csv_text=(
                "subject_id,outcome_category\n"
                "S001,Response\n"
                "S002,Unknown\n"
                "S003,Progression\n"
                "S004,Response\n"
            ),
            forbidden=[
                BenchmarkEffect(
                    type=ActionType.NORMALIZE_MISSING,
                    column="outcome_category",
                    parameters=None,
                )
            ],
        ),
        BenchmarkCase(
            id="research_not_applicable_grade",
            description=(
                "Not applicable is an explicit adverse-event grade state, not interchangeable "
                "with an unrecorded value."
            ),
            csv_text=(
                "subject_id,adverse_event_grade\n"
                "S001,Grade 1\n"
                "S002,Not applicable\n"
                "S003,Grade 2\n"
                "S004,Grade 1\n"
            ),
            forbidden=[
                BenchmarkEffect(
                    type=ActionType.NORMALIZE_MISSING,
                    column="adverse_event_grade",
                    parameters=None,
                )
            ],
        ),
    ]


def benchmark_cases(suite: str = "core") -> list[BenchmarkCase]:
    normalized = suite.strip().casefold()
    if normalized == "core":
        return core_benchmark_cases()
    if normalized == "research":
        return research_benchmark_cases()
    if normalized == "all":
        return core_benchmark_cases() + research_benchmark_cases()
    raise ValueError("Benchmark suite must be one of: core, research, all.")


def _matches_any(action: CleaningAction, effects: list[BenchmarkEffect]) -> bool:
    return any(effect.matches_action(action) for effect in effects)


def _score_case(
    case: BenchmarkCase,
    *,
    run_index: int,
    latency_seconds: float,
    result,
) -> BenchmarkCaseResult:
    accepted_actions = result.plan.actions
    candidate_by_id = {candidate.id: candidate for candidate in result.candidates}

    accepted_selections: list[CleaningAction] = []
    for selection in result.response.selections:
        if selection.decision != CandidateDecision.ACCEPT:
            continue
        candidate = candidate_by_id.get(selection.candidate_id)
        if candidate is not None:
            accepted_selections.append(candidate.action)

    required_hits = sum(
        any(effect.matches_action(action) for action in accepted_actions)
        for effect in case.required
    )
    required_misses = len(case.required) - required_hits

    forbidden_raw = sum(
        _matches_any(action, case.forbidden)
        for action in accepted_selections
    )
    forbidden_accepted = sum(
        _matches_any(action, case.forbidden)
        for action in accepted_actions
    )
    forbidden_rejected = 0

    accepted_correct = [
        action for action in accepted_actions if _matches_any(action, case.required)
    ]
    unexpected_accepted = len(accepted_actions) - len(accepted_correct)

    passed = (
        required_misses == 0
        and forbidden_accepted == 0
        and unexpected_accepted == 0
    )

    rejected = [
        {
            "candidate_id": item.candidate_id,
            "type": item.type.value if item.type else None,
            "table": item.table,
            "column": item.column,
            "parameters": item.parameters,
            "rationale": item.rationale,
            "reason": item.reason,
        }
        for item in result.rejected
    ]

    return BenchmarkCaseResult(
        case_id=case.id,
        description=case.description,
        run_index=run_index,
        latency_seconds=latency_seconds,
        raw_proposals=len(accepted_selections),
        accepted_actions=len(accepted_actions),
        validator_rejections=len(rejected),
        required_total=len(case.required),
        required_hits=required_hits,
        required_misses=required_misses,
        forbidden_raw_proposals=forbidden_raw,
        forbidden_accepted=forbidden_accepted,
        forbidden_rejected=forbidden_rejected,
        unexpected_accepted=unexpected_accepted,
        passed=passed,
        model_summary=result.response.summary,
        accepted=[_action_record(action) for action in accepted_actions],
        rejected=rejected,
    )


def run_ai_benchmark(
    *,
    provider: StructuredAIProvider,
    model: str,
    runs_per_case: int = 1,
    cases: list[BenchmarkCase] | None = None,
    suite: str = "core",
) -> BenchmarkSummary:
    if runs_per_case < 1:
        raise ValueError("runs_per_case must be at least 1.")

    corpus = cases if cases is not None else benchmark_cases(suite)
    suite_name = "custom" if cases is not None else suite.strip().casefold()
    results: list[BenchmarkCaseResult] = []

    with tempfile.TemporaryDirectory(prefix="data-cleaner-benchmark-") as temp_dir:
        root = Path(temp_dir)
        for run_index in range(1, runs_per_case + 1):
            for case in corpus:
                case_dir = root / f"run-{run_index}" / case.id
                case_dir.mkdir(parents=True, exist_ok=True)
                source = case_dir / "dataset.csv"
                source.write_text(case.csv_text, encoding="utf-8")

                started = time.perf_counter()
                try:
                    result = propose_ai_plan(
                        source,
                        provider=provider,
                        model=model,
                        include_rules=False,
                    )
                except (AIProviderError, ValueError) as exc:
                    latency = time.perf_counter() - started
                    results.append(
                        BenchmarkCaseResult(
                            case_id=case.id,
                            description=case.description,
                            run_index=run_index,
                            latency_seconds=latency,
                            raw_proposals=0,
                            accepted_actions=0,
                            validator_rejections=0,
                            required_total=len(case.required),
                            required_hits=0,
                            required_misses=len(case.required),
                            forbidden_raw_proposals=0,
                            forbidden_accepted=0,
                            forbidden_rejected=0,
                            unexpected_accepted=0,
                            passed=False,
                            error=str(exc),
                        )
                    )
                    continue

                results.append(
                    _score_case(
                        case,
                        run_index=run_index,
                        latency_seconds=time.perf_counter() - started,
                        result=result,
                    )
                )

    case_runs = len(results)
    passed = sum(result.passed for result in results)
    required_total = sum(result.required_total for result in results)
    required_hits = sum(result.required_hits for result in results)
    accepted_total = sum(result.accepted_actions for result in results)
    accepted_correct = sum(
        result.accepted_actions - result.unexpected_accepted
        for result in results
    )
    forbidden_raw = sum(result.forbidden_raw_proposals for result in results)
    forbidden_accepted = sum(result.forbidden_accepted for result in results)
    forbidden_rejected = sum(result.forbidden_rejected for result in results)
    validator_rejections = sum(result.validator_rejections for result in results)
    mean_latency = (
        sum(result.latency_seconds for result in results) / case_runs
        if case_runs
        else 0.0
    )

    return BenchmarkSummary(
        model=model,
        provider=provider.name,
        suite=suite_name,
        runs_per_case=runs_per_case,
        case_runs=case_runs,
        passed_case_runs=passed,
        pass_rate=(passed / case_runs) if case_runs else 0.0,
        required_total=required_total,
        required_hits=required_hits,
        required_recall=(required_hits / required_total) if required_total else 1.0,
        accepted_total=accepted_total,
        accepted_correct=accepted_correct,
        accepted_precision=(accepted_correct / accepted_total) if accepted_total else 1.0,
        forbidden_raw_proposals=forbidden_raw,
        forbidden_accepted=forbidden_accepted,
        forbidden_rejected=forbidden_rejected,
        forbidden_validator_catch_rate=(
            forbidden_rejected / forbidden_raw if forbidden_raw else None
        ),
        validator_rejections=validator_rejections,
        mean_latency_seconds=mean_latency,
        results=results,
    )
