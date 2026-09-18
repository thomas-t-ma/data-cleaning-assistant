from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .actions import ActionType


class AIColumnContext(BaseModel):
    name: str
    inferred_type: str
    type_confidence: float
    null_percent: float
    unique_count: int
    unique_percent: float
    numeric_summary: dict[str, float | int | None] | None = None


class AITableContext(BaseModel):
    name: str
    row_count: int
    column_count: int
    grain: str | None = None
    grain_confidence: float | None = None
    candidate_keys: list[list[str]] = Field(default_factory=list)
    columns: list[AIColumnContext] = Field(default_factory=list)


class AIIssueContext(BaseModel):
    id: str
    type: str
    severity: str
    table: str
    column: str | None = None
    title: str
    confidence: float
    affected_count: int
    evidence: dict[str, Any] = Field(default_factory=dict)


class AIRelationshipContext(BaseModel):
    left: str
    right: str
    relationship: str
    confidence: float


class AIPlannerContext(BaseModel):
    context_version: int = 1
    privacy_note: str = (
        "No source path, raw row samples, or identifier values are included. "
        "Detector evidence may include category labels, missing tokens, or numeric/date examples "
        "needed to evaluate supported cleaning actions."
    )
    tables: list[AITableContext] = Field(default_factory=list)
    issues: list[AIIssueContext] = Field(default_factory=list)
    relationships: list[AIRelationshipContext] = Field(default_factory=list)


class AIActionProposal(BaseModel):
    type: ActionType
    table: str
    column: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    issue_ids: list[str] = Field(default_factory=list)


class AIPlannerResponse(BaseModel):
    summary: str
    actions: list[AIActionProposal] = Field(default_factory=list)
    deferred_issue_ids: list[str] = Field(default_factory=list)


class AIRejectedProposal(BaseModel):
    type: ActionType
    table: str
    column: str | None = None
    reason: str
