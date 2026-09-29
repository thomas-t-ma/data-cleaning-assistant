from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .actions import CleaningAction
from .codebook import suggest_codebooks
from .engine import profile_dataset
from .io import load_dataset
from .planner import propose_actions


class RecallSignal(BaseModel):
    kind: Literal["action", "issue", "codebook"]
    name: str
    column: str | None = None
    parameters: dict[str, Any] | None = None


class RecallBenchmarkCase(BaseModel):
    id: str
    description: str
    csv_text: str
    required: list[RecallSignal] = Field(default_factory=list)
    forbidden: list[RecallSignal] = Field(default_factory=list)


class RecallBenchmarkCaseResult(BaseModel):
    case_id: str
    description: str
    required_total: int
    required_hits: int
    required_misses: list[RecallSignal] = Field(default_factory=list)
    forbidden_total: int
    forbidden_hits: list[RecallSignal] = Field(default_factory=list)
    passed: bool


class RecallBenchmarkSummary(BaseModel):
    suite: str = "v0.10"
    case_count: int
    passed_cases: int
    pass_rate: float
    required_total: int
    required_hits: int
    required_recall: float
    forbidden_total: int
    forbidden_hits: int
    results: list[RecallBenchmarkCaseResult] = Field(default_factory=list)
    interpretation_note: str = (
        "Synthetic deterministic engineering benchmark. Passing does not establish "
        "complete issue recall, scientific correctness, or safety on unseen datasets."
    )


def _parameters_equal(
    observed: dict[str, Any],
    expected: dict[str, Any] | None,
) -> bool:
    if expected is None:
        return True
    return json.dumps(observed, sort_keys=True, default=str) == json.dumps(
        expected,
        sort_keys=True,
        default=str,
    )


def _matches_action(signal: RecallSignal, action: CleaningAction) -> bool:
    return (
        signal.kind == "action"
        and action.type.value == signal.name
        and (signal.column is None or action.column == signal.column)
        and _parameters_equal(action.parameters, signal.parameters)
    )


def _observed(path: Path) -> tuple[list[CleaningAction], list, list]:
    loaded = load_dataset(path)
    profile = profile_dataset(path)
    plan = propose_actions(path)
    codebooks = suggest_codebooks(path, loaded.tables)
    return plan.actions, profile.issues, codebooks.suggestions


def _signal_present(
    signal: RecallSignal,
    *,
    actions: list[CleaningAction],
    issues: list,
    codebooks: list,
) -> bool:
    if signal.kind == "action":
        return any(_matches_action(signal, action) for action in actions)
    if signal.kind == "issue":
        return any(
            issue.type == signal.name
            and (signal.column is None or issue.column == signal.column)
            for issue in issues
        )
    return any(
        signal.name == "codebook"
        and (signal.column is None or suggestion.column == signal.column)
        for suggestion in codebooks
    )


def v010_recall_benchmark_cases() -> list[RecallBenchmarkCase]:
    return [
        RecallBenchmarkCase(
            id="multiselect_component_variant",
            description="A repeated multi-select component case variant should be reviewable.",
            csv_text=(
                "subject_id,language\n"
                'S001,"R;BUGS"\n'
                'S002,"R;Bugs"\n'
                'S003,"Python;BUGS"\n'
                'S004,"R;BUGS"\n'
                'S005,"R;BUGS"\n'
                'S006,"Python;BUGS"\n'
            ),
            required=[
                RecallSignal(
                    kind="action",
                    name="normalize_multiselect",
                    column="language",
                    parameters={
                        "delimiter": ";",
                        "mapping": {"Bugs": "BUGS"},
                    },
                )
            ],
        ),
        RecallBenchmarkCase(
            id="multiselect_subset_preserved",
            description="An additional selected option must never be treated as a typo.",
            csv_text=(
                "subject_id,language\n"
                'S001,"R"\n'
                'S002,"R"\n'
                'S003,"R;BUGS"\n'
                'S004,"R;BUGS"\n'
                'S005,"R;BUGS"\n'
                'S006,"R;BUGS"\n'
            ),
            forbidden=[
                RecallSignal(
                    kind="action",
                    name="normalize_categories",
                    column="language",
                ),
                RecallSignal(
                    kind="action",
                    name="normalize_multiselect",
                    column="language",
                    parameters={
                        "delimiter": ";",
                        "mapping": {"BUGS": "R"},
                    },
                ),
            ],
        ),
        RecallBenchmarkCase(
            id="semicolon_scalar_prose",
            description="Occasional semicolons in scalar prose must not create a multi-select action.",
            csv_text=(
                "subject_id,citation_count\n"
                "S001,0\n"
                "S002,1\n"
                "S003,2\n"
                "S004,3\n"
                "S005,4\n"
                'S006,"bat = 0 ; LiDAR = 5"\n'
                'S007,"1; however the citation may be misassigned"\n'
                "S008,5\n"
                "S009,6\n"
                "S010,7\n"
            ),
            forbidden=[
                RecallSignal(
                    kind="action",
                    name="normalize_multiselect",
                    column="citation_count",
                )
            ],
        ),
        RecallBenchmarkCase(
            id="multiselect_tied_canonical",
            description="Tied component spellings must remain unresolved.",
            csv_text=(
                "subject_id,language\n"
                'S001,"R;BUGS"\n'
                'S002,"R;Bugs"\n'
                'S003,"Python;BUGS"\n'
                'S004,"Python;Bugs"\n'
                'S005,"Julia;BUGS"\n'
                'S006,"Julia;Bugs"\n'
            ),
            forbidden=[
                RecallSignal(
                    kind="action",
                    name="normalize_multiselect",
                    column="language",
                )
            ],
        ),
        RecallBenchmarkCase(
            id="conditional_if_yes",
            description="Explicit If yes dependency contradictions should be surfaced.",
            csv_text=(
                'subject_id,has_license,"If yes, what license?"\n'
                "S001,Yes,MIT\n"
                "S002,No,\n"
                "S003,No,CC BY\n"
                "S004,Yes,NA\n"
                "S005,Unclear,Unclear\n"
            ),
            required=[
                RecallSignal(
                    kind="issue",
                    name="conditional_dependency_inconsistency",
                    column="If yes, what license?",
                )
            ],
        ),
        RecallBenchmarkCase(
            id="conditional_nonbinary_parent",
            description="An explicit dependent phrase must not attach to a nonbinary parent.",
            csv_text=(
                'subject_id,status,"If yes, explain"\n'
                "S001,Alpha,detail one\n"
                "S002,Beta,detail two\n"
                "S003,Gamma,detail three\n"
                "S004,Delta,detail four\n"
            ),
            forbidden=[
                RecallSignal(
                    kind="issue",
                    name="conditional_dependency_inconsistency",
                    column="If yes, explain",
                )
            ],
        ),
        RecallBenchmarkCase(
            id="conditional_nonadjacent_generic",
            description=(
                "A non-adjacent generic If yes field must not guess an unrelated "
                "binary parent without lexical evidence."
            ),
            csv_text=(
                'subject_id,has_license,reviewer_note,"If yes, explain"\n'
                "S001,Yes,checked,detail one\n"
                "S002,No,checked,unexpected detail\n"
                "S003,Yes,checked,detail three\n"
                "S004,No,checked,\n"
            ),
            forbidden=[
                RecallSignal(
                    kind="issue",
                    name="conditional_dependency_inconsistency",
                    column="If yes, explain",
                )
            ],
        ),
        RecallBenchmarkCase(
            id="conditional_count_zero",
            description="Zero in a count field is numeric, not a Boolean spelling.",
            csv_text=(
                'subject_id,cited,"If yes, how many times?"\n'
                "S001,No,0\n"
                "S002,Yes,1\n"
                "S003,Yes,0\n"
                "S004,No,Unclear\n"
                "S005,Yes,NA\n"
                "S006,No,No\n"
            ),
            required=[
                RecallSignal(
                    kind="issue",
                    name="conditional_dependency_inconsistency",
                    column="If yes, how many times?",
                )
            ],
        ),
        RecallBenchmarkCase(
            id="normalization_equivalent_duplicate",
            description="Rows equal after safe normalization should be QA-only near duplicates.",
            csv_text=(
                "subject_id,label,status\n"
                "S001,Alpha,Active\n"
                "S001, alpha ,active\n"
                "S002,Beta,Active\n"
            ),
            required=[
                RecallSignal(
                    kind="issue",
                    name="normalization_equivalent_rows",
                )
            ],
        ),
        RecallBenchmarkCase(
            id="identifier_case_not_duplicate",
            description=(
                "Identifier case differences must not be normalized into a "
                "near-duplicate record."
            ),
            csv_text=(
                "subject_id,label,status\n"
                "ABC123,Alpha,Active\n"
                "abc123, alpha ,active\n"
                "XYZ999,Beta,Active\n"
            ),
            forbidden=[
                RecallSignal(
                    kind="issue",
                    name="normalization_equivalent_rows",
                )
            ],
        ),
        RecallBenchmarkCase(
            id="exact_duplicate_not_near_duplicate",
            description="Exact duplicates stay with the exact-duplicate detector.",
            csv_text=(
                "subject_id,label\n"
                "S001,Alpha\n"
                "S001,Alpha\n"
            ),
            forbidden=[
                RecallSignal(
                    kind="issue",
                    name="normalization_equivalent_rows",
                )
            ],
        ),
        RecallBenchmarkCase(
            id="codebook_core_with_writein",
            description="A dominant codebook should preserve rare values as review context.",
            csv_text=(
                "status\n"
                + "Active\n" * 30
                + "active\n" * 3
                + "Inactive\n" * 6
                + "reviewer-specific note\n"
            ),
            required=[
                RecallSignal(
                    kind="codebook",
                    name="codebook",
                    column="status",
                )
            ],
        ),
        RecallBenchmarkCase(
            id="codebook_count_negative",
            description="Explicit count fields are not categorical codebooks.",
            csv_text=(
                '"If yes, how many times? Enter number or Unclear"\n'
                + "0\n" * 12
                + "1\n" * 8
                + "2\n" * 4
                + "Unclear\n" * 6
            ),
            forbidden=[
                RecallSignal(
                    kind="codebook",
                    name="codebook",
                    column="If yes, how many times? Enter number or Unclear",
                )
            ],
        ),
    ]


def run_v010_recall_benchmark(
    cases: list[RecallBenchmarkCase] | None = None,
) -> RecallBenchmarkSummary:
    corpus = cases or v010_recall_benchmark_cases()
    results: list[RecallBenchmarkCaseResult] = []

    with tempfile.TemporaryDirectory(prefix="data-cleaner-recall-") as temp_dir:
        root = Path(temp_dir)
        for case in corpus:
            source = root / f"{case.id}.csv"
            source.write_text(case.csv_text, encoding="utf-8")
            actions, issues, codebooks = _observed(source)

            missing = [
                signal
                for signal in case.required
                if not _signal_present(
                    signal,
                    actions=actions,
                    issues=issues,
                    codebooks=codebooks,
                )
            ]
            forbidden_hits = [
                signal
                for signal in case.forbidden
                if _signal_present(
                    signal,
                    actions=actions,
                    issues=issues,
                    codebooks=codebooks,
                )
            ]
            results.append(
                RecallBenchmarkCaseResult(
                    case_id=case.id,
                    description=case.description,
                    required_total=len(case.required),
                    required_hits=len(case.required) - len(missing),
                    required_misses=missing,
                    forbidden_total=len(case.forbidden),
                    forbidden_hits=forbidden_hits,
                    passed=not missing and not forbidden_hits,
                )
            )

    required_total = sum(result.required_total for result in results)
    required_hits = sum(result.required_hits for result in results)
    forbidden_total = sum(result.forbidden_total for result in results)
    forbidden_hits = sum(len(result.forbidden_hits) for result in results)
    passed = sum(result.passed for result in results)
    return RecallBenchmarkSummary(
        case_count=len(results),
        passed_cases=passed,
        pass_rate=passed / len(results) if results else 1.0,
        required_total=required_total,
        required_hits=required_hits,
        required_recall=required_hits / required_total if required_total else 1.0,
        forbidden_total=forbidden_total,
        forbidden_hits=forbidden_hits,
        results=results,
    )
