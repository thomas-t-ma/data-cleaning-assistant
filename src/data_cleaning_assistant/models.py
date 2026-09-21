from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Issue(BaseModel):
    id: str
    type: str
    severity: Severity
    table: str
    column: str | None = None
    title: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    affected_count: int = Field(ge=0)
    row_indices_sample: list[int] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    suggested_action: str | None = None
    auto_fix_safe: bool = False


class CandidateKey(BaseModel):
    columns: list[str] = Field(min_length=1)
    uniqueness: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


class GrainInference(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class CrossTableRelationship(BaseModel):
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    relationship: str
    shared_value_count: int = Field(ge=0)
    left_coverage: float = Field(ge=0.0, le=1.0)
    right_coverage: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = "inferred"


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    row_count: int
    null_count: int
    null_percent: float
    unique_count: int
    unique_percent: float
    inferred_type: str
    type_confidence: float
    sample_values: list[Any] = Field(default_factory=list)
    numeric_summary: dict[str, float | int | None] | None = None


class TableProfile(BaseModel):
    name: str
    row_count: int
    column_count: int
    columns: list[ColumnProfile]
    candidate_keys: list[CandidateKey] = Field(default_factory=list)
    grain: GrainInference | None = None


class DatasetReport(BaseModel):
    source_path: str
    source_type: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tables: list[TableProfile]
    issues: list[Issue]
    relationships: list[CrossTableRelationship] = Field(default_factory=list)

    @property
    def issue_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for issue in self.issues:
            counts[issue.severity.value] = counts.get(issue.severity.value, 0) + 1
        return counts
