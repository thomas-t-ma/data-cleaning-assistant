from __future__ import annotations

from enum import Enum
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


class AICandidateContext(BaseModel):
    id: str
    type: ActionType
    table: str
    column: str | None = None
    proposed_change: dict[str, Any] = Field(default_factory=dict)
    detector_confidence: float
    evidence: dict[str, Any] = Field(default_factory=dict)
    question: str


class AIPlannerContext(BaseModel):
    context_version: int = 2
    privacy_note: str = (
        "No source path, raw row samples, or identifier values are included. "
        "The model receives deterministic candidate actions and only decides accept versus defer. "
        "Detector evidence may include category labels, missing tokens, or numeric/date examples "
        "needed to judge those candidates."
    )
    tables: list[AITableContext] = Field(default_factory=list)
    issues: list[AIIssueContext] = Field(default_factory=list)
    relationships: list[AIRelationshipContext] = Field(default_factory=list)
    candidates: list[AICandidateContext] = Field(default_factory=list)


class CandidateDecision(str, Enum):
    ACCEPT = "accept"
    DEFER = "defer"


class AICandidateSelection(BaseModel):
    candidate_id: str
    decision: CandidateDecision
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


class AIPlannerResponse(BaseModel):
    summary: str
    selections: list[AICandidateSelection] = Field(default_factory=list)


class AIRejectedProposal(BaseModel):
    candidate_id: str
    type: ActionType | None = None
    table: str | None = None
    column: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    reason: str
