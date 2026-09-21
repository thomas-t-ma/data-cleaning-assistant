from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .inference import is_missing_like, looks_like_identifier_name


class ColumnSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    allowed_values: list[str] | None = None
    canonical_map: dict[str, str] = Field(default_factory=dict)
    missing_tokens: list[str] = Field(default_factory=list)
    unit: str | None = None

    @model_validator(mode="after")
    def validate_mappings(self) -> ColumnSpec:
        if len(set(self.missing_tokens)) != len(self.missing_tokens):
            raise ValueError("missing_tokens may not contain duplicates")
        if set(self.canonical_map) & set(self.missing_tokens):
            raise ValueError("The same token cannot be both canonicalized and set to null")
        if self.allowed_values is not None:
            unknown_targets = set(self.canonical_map.values()) - set(self.allowed_values)
            if unknown_targets:
                raise ValueError(
                    "canonical_map targets must be listed in allowed_values: "
                    + ", ".join(sorted(unknown_targets))
                )
        return self


class TableSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    grain_key: list[str] = Field(default_factory=list)
    subject_id: str | None = None
    time_column: str | None = None
    order_column: str | None = None
    columns: dict[str, ColumnSpec] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_time(self) -> TableSpec:
        if bool(self.time_column) != bool(self.order_column):
            raise ValueError("time_column and order_column must be specified together")
        if self.time_column and not self.subject_id:
            raise ValueError("subject_id is required for chronological checks")
        return self


class ForeignKeySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    child_table: str
    child_column: str
    parent_table: str
    parent_column: str


class StudySchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = 1
    tables: dict[str, TableSpec] = Field(default_factory=dict)
    foreign_keys: list[ForeignKeySpec] = Field(default_factory=list)
    source_path: str | None = Field(default=None, exclude=True)
    source_sha256: str | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def validate_version(self) -> StudySchema:
        if self.version != 1:
            raise ValueError("Unsupported study schema version; expected version: 1")
        return self

    def validate_dataset(self, tables: dict[str, pd.DataFrame]) -> None:
        for table_name, table_spec in self.tables.items():
            if table_name not in tables:
                raise ValueError(f"Schema references missing table/sheet {table_name!r}")
            frame = tables[table_name]
            required = (
                set(table_spec.grain_key)
                | set(table_spec.columns)
                | {
                    item
                    for item in (
                        table_spec.subject_id,
                        table_spec.time_column,
                        table_spec.order_column,
                    )
                    if item
                }
            )
            unknown = required - set(frame.columns)
            if unknown:
                raise ValueError(
                    f"Schema references unknown columns in {table_name}: "
                    + ", ".join(sorted(unknown))
                )
            identifiers = set(table_spec.grain_key)
            if table_spec.subject_id:
                identifiers.add(table_spec.subject_id)
            for column, spec in table_spec.columns.items():
                if (column in identifiers or looks_like_identifier_name(column)) and (
                    spec.canonical_map or spec.missing_tokens
                ):
                    raise ValueError(
                        f"Schema cannot propose identifier edits: {table_name}.{column}"
                    )
                if any(is_missing_like(value) for value in spec.canonical_map.values()):
                    raise ValueError(
                        f"Canonical targets cannot be structural nulls: {table_name}.{column}"
                    )

        for relation in self.foreign_keys:
            for table, column in (
                (relation.child_table, relation.child_column),
                (relation.parent_table, relation.parent_column),
            ):
                if table not in tables or column not in tables[table].columns:
                    raise ValueError(f"Schema references missing foreign key {table}.{column}")


def load_study_schema(path: str | Path) -> StudySchema:
    resolved = Path(path).expanduser().resolve()
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("Study schema must be a YAML/JSON mapping")
    schema = StudySchema.model_validate(raw)
    schema.source_path = str(resolved)
    schema.source_sha256 = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return schema
