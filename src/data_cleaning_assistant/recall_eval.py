from __future__ import annotations

from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from .actions import ActionType
from .codebook import CodebookSuggestion, suggest_codebooks
from .engine import profile_dataset
from .io import load_dataset
from .planner import propose_actions


V010_RECALL_ISSUE_TYPES = {
    "multiselect_component_inconsistency",
    "conditional_dependency_inconsistency",
    "normalization_equivalent_rows",
}


class RecallSurfaceReport(BaseModel):
    source_path: str
    source_sha256: str
    row_counts: dict[str, int] = Field(default_factory=dict)
    column_counts: dict[str, int] = Field(default_factory=dict)
    new_issue_counts: dict[str, int] = Field(default_factory=dict)
    new_issue_affected_counts: dict[str, int] = Field(default_factory=dict)
    multiselect_action_count: int = 0
    multiselect_action_affected_count: int = 0
    conditional_pair_count: int = 0
    conditional_affected_count: int = 0
    normalized_duplicate_group_issue_count: int = 0
    normalized_duplicate_affected_count: int = 0
    codebook_suggestion_count: int = 0
    codebook_suggestions: list[CodebookSuggestion] = Field(default_factory=list)
    interpretation_note: str = (
        "This report measures the deterministic v0.10 review surface. It does not "
        "establish scientific correctness, complete issue recall, or permission to "
        "apply any transformation."
    )


def build_recall_surface_report(path: str | Path) -> RecallSurfaceReport:
    resolved = Path(path).expanduser().resolve()
    loaded = load_dataset(resolved)
    profile = profile_dataset(resolved)
    plan = propose_actions(resolved)
    codebooks = suggest_codebooks(resolved, loaded.tables)

    issue_counts: Counter[str] = Counter()
    affected_counts: Counter[str] = Counter()
    for issue in profile.issues:
        if issue.type not in V010_RECALL_ISSUE_TYPES:
            continue
        issue_counts[issue.type] += 1
        affected_counts[issue.type] += issue.affected_count

    multiselect_actions = [
        action
        for action in plan.actions
        if action.type == ActionType.NORMALIZE_MULTISELECT
    ]

    # Action affected counts are intentionally approximated from the linked
    # detector findings rather than by applying edits. This report stays read-only.
    issue_by_id = {issue.id: issue for issue in profile.issues}
    multiselect_affected = sum(
        sum(
            issue_by_id[issue_id].affected_count
            for issue_id in action.issue_ids
            if issue_id in issue_by_id
        )
        for action in multiselect_actions
    )

    return RecallSurfaceReport(
        source_path=str(resolved),
        source_sha256=plan.source_sha256,
        row_counts={
            table: len(frame)
            for table, frame in loaded.tables.items()
        },
        column_counts={
            table: len(frame.columns)
            for table, frame in loaded.tables.items()
        },
        new_issue_counts=dict(sorted(issue_counts.items())),
        new_issue_affected_counts=dict(sorted(affected_counts.items())),
        multiselect_action_count=len(multiselect_actions),
        multiselect_action_affected_count=multiselect_affected,
        conditional_pair_count=issue_counts["conditional_dependency_inconsistency"],
        conditional_affected_count=affected_counts["conditional_dependency_inconsistency"],
        normalized_duplicate_group_issue_count=issue_counts["normalization_equivalent_rows"],
        normalized_duplicate_affected_count=affected_counts["normalization_equivalent_rows"],
        codebook_suggestion_count=len(codebooks.suggestions),
        codebook_suggestions=codebooks.suggestions,
    )
