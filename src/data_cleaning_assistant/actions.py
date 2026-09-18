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


class ActionDecision(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"


class CleaningAction(BaseModel):
    id: str
    type: ActionType
    table: str
    column: str
    title: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    issue_ids: list[str] = Field(default_factory=list)
    decision: ActionDecision = ActionDecision.PROPOSED
    reversible: bool = True


class CleaningPlan(BaseModel):
    source_path: str
    source_sha256: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    actions: list[CleaningAction] = Field(default_factory=list)


class CellChange(BaseModel):
    table: str
    row_index: int
    column: str
    old_value: Any = None
    new_value: Any = None


class ActionPreview(BaseModel):
    action_id: str
    affected_count: int
    sample_changes: list[CellChange] = Field(default_factory=list)


class ActionExecution(BaseModel):
    action_id: str
    action_type: ActionType
    table: str
    column: str
    affected_count: int
    changes: list[CellChange] = Field(default_factory=list)


class ChangeSet(BaseModel):
    source_path: str
    source_sha256: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    executions: list[ActionExecution] = Field(default_factory=list)

    @property
    def affected_cell_count(self) -> int:
        return sum(execution.affected_count for execution in self.executions)
