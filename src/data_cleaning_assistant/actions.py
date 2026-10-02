from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    NORMALIZE_CATEGORIES = "normalize_categories"
    NORMALIZE_MISSING = "normalize_missing"
    STANDARDIZE_DATES = "standardize_dates"
    COERCE_NUMERIC = "coerce_numeric"
    TRIM_WHITESPACE = "trim_whitespace"
    REMOVE_EXACT_DUPLICATES = "remove_exact_duplicates"


class ActionDecision(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"


class ActionOrigin(str, Enum):
    RULE = "rule"
    AI = "ai"
    RECIPE = "recipe"


class CleaningAction(BaseModel):
    id: str
    type: ActionType
    table: str
    column: str | None = None
    title: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    issue_ids: list[str] = Field(default_factory=list)
    decision: ActionDecision = ActionDecision.PROPOSED
    reversible: bool = True
    origin: ActionOrigin = ActionOrigin.RULE
    parent_action_id: str | None = None


class DeferredReview(BaseModel):
    """Non-executable candidate retained for explicit human judgment."""

    candidate_id: str
    proposed_action: CleaningAction
    question: str
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: dict[str, Any] = Field(default_factory=dict)
    status: str = "pending"
    promoted_action_id: str | None = None
    promoted_action_created: bool = False
    promoted_action_previous_decision: ActionDecision | None = None


class PlanningNote(BaseModel):
    """A non-executable reason an issue did not become a cleaning action."""

    table: str
    column: str
    issue_ids: list[str] = Field(default_factory=list)
    reason: str


class ReviewEvent(BaseModel):
    """One explicit human-review decision or deferral."""

    scope: str
    action_ids: list[str] = Field(default_factory=list)
    action_type: ActionType | None = None
    origin: ActionOrigin | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    member_count: int = Field(ge=0)
    affected_count: int = Field(ge=0)
    decision: str
    bulk_confirmed: bool = False
    duration_seconds: float = Field(ge=0.0)
    targets_preview: list[str] = Field(default_factory=list)


class ReviewSession(BaseModel):
    """Provenance for an interactive human review session."""

    started_at: datetime
    completed_at: datetime | None = None
    duration_seconds: float | None = Field(default=None, ge=0.0)
    events: list[ReviewEvent] = Field(default_factory=list)


class PlanAmendment(BaseModel):
    """Auditable supported edit to a saved or in-progress cleaning plan."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    kind: str
    action_id: str | None = None
    deferred_candidate_id: str | None = None
    before: dict[str, Any] = Field(default_factory=dict)
    after: dict[str, Any] = Field(default_factory=dict)
    reason: str


class CleaningPlan(BaseModel):
    source_path: str
    source_sha256: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    actions: list[CleaningAction] = Field(default_factory=list)
    deferred_reviews: list[DeferredReview] = Field(default_factory=list)
    planning_notes: list[PlanningNote] = Field(default_factory=list)
    study_schema_sha256: str | None = None
    planner: str = "deterministic"
    model: str | None = None
    review_session: ReviewSession | None = None
    amendments: list[PlanAmendment] = Field(default_factory=list)


class ValueSnapshot(BaseModel):
    kind: str
    value: Any = None


class CellChange(BaseModel):
    table: str
    row_index: int
    column: str
    old_value: ValueSnapshot
    new_value: ValueSnapshot


class RowSnapshot(BaseModel):
    table: str
    row_position: int
    values: dict[str, ValueSnapshot] = Field(default_factory=dict)


class ActionPreview(BaseModel):
    action_id: str
    affected_count: int
    sample_changes: list[CellChange] = Field(default_factory=list)
    sample_rows: list[RowSnapshot] = Field(default_factory=list)


class ActionExecution(BaseModel):
    action_id: str
    action_type: ActionType
    table: str
    column: str | None = None
    affected_count: int
    changes: list[CellChange] = Field(default_factory=list)
    deleted_rows: list[RowSnapshot] = Field(default_factory=list)


class ChangeSet(BaseModel):
    source_path: str
    source_sha256: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    executions: list[ActionExecution] = Field(default_factory=list)
    study_schema_sha256: str | None = None
    review_session: ReviewSession | None = None
    result_path: str | None = None
    result_sha256: str | None = None

    @property
    def affected_cell_count(self) -> int:
        return sum(len(execution.changes) for execution in self.executions)

    @property
    def affected_row_count(self) -> int:
        return sum(len(execution.deleted_rows) for execution in self.executions)
