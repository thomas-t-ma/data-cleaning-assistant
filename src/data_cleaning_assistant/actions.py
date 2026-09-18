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


class CleaningPlan(BaseModel):
    source_path: str
    source_sha256: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    actions: list[CleaningAction] = Field(default_factory=list)
    planner: str = "deterministic"
    model: str | None = None


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
    result_path: str | None = None
    result_sha256: str | None = None

    @property
    def affected_cell_count(self) -> int:
        return sum(len(execution.changes) for execution in self.executions)

    @property
    def affected_row_count(self) -> int:
        return sum(len(execution.deleted_rows) for execution in self.executions)
