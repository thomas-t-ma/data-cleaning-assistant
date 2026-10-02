from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .inference import is_missing_like, normalize_column_name


class CanonicalValueSpec(BaseModel):
    """One protected canonical category and its explicitly authorized aliases."""

    model_config = ConfigDict(extra="forbid")
    aliases: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_aliases(self) -> CanonicalValueSpec:
        if len(set(self.aliases)) != len(self.aliases):
            raise ValueError("canonical value aliases may not contain duplicates")
        return self


class ColumnSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    allowed_values: list[str] | None = None
    canonical_map: dict[str, str] = Field(default_factory=dict)
    canonical_values: dict[str, CanonicalValueSpec] = Field(default_factory=dict)
    missing_tokens: list[str] = Field(default_factory=list)
    unit: str | None = None

    def effective_canonical_map(self) -> dict[str, str]:
        """Return legacy and vocabulary aliases as one deterministic mapping."""
        mapping = dict(self.canonical_map)
        for canonical, spec in self.canonical_values.items():
            for alias in spec.aliases:
                mapping[alias] = canonical
        return mapping

    def effective_allowed_values(self) -> set[str] | None:
        """Return the authoritative category set, if this column declares one."""
        if self.allowed_values is None and not self.canonical_values:
            return None
        allowed = set(self.allowed_values or [])
        allowed.update(self.canonical_values)
        return allowed

    def has_category_dictionary(self) -> bool:
        return (
            self.allowed_values is not None
            or bool(self.canonical_map)
            or bool(self.canonical_values)
        )

    @model_validator(mode="after")
    def validate_mappings(self) -> ColumnSpec:
        if len(set(self.missing_tokens)) != len(self.missing_tokens):
            raise ValueError("missing_tokens may not contain duplicates")

        canonical_labels = set(self.canonical_values)
        vocabulary_alias_targets: dict[str, str] = {}
        for canonical, spec in self.canonical_values.items():
            for alias in spec.aliases:
                if alias == canonical:
                    raise ValueError(
                        f"Alias {alias!r} duplicates its canonical label"
                    )
                previous = vocabulary_alias_targets.get(alias)
                if previous is not None and previous != canonical:
                    raise ValueError(
                        f"Alias {alias!r} maps to multiple canonical values: "
                        f"{previous!r}, {canonical!r}"
                    )
                vocabulary_alias_targets[alias] = canonical

        protected_aliases = canonical_labels & set(vocabulary_alias_targets)
        legacy_protected_aliases = {
            source
            for source, target in self.canonical_map.items()
            if source in canonical_labels and source != target
        }
        protected_aliases |= legacy_protected_aliases
        if protected_aliases:
            alias = sorted(protected_aliases)[0]
            raise ValueError(
                f"Canonical label {alias!r} cannot be an alias of another "
                "canonical value"
            )

        for alias, target in self.canonical_map.items():
            vocabulary_target = vocabulary_alias_targets.get(alias)
            if vocabulary_target is not None and vocabulary_target != target:
                raise ValueError(
                    f"Alias {alias!r} has conflicting canonical targets: "
                    f"{target!r}, {vocabulary_target!r}"
                )

        effective_mapping = self.effective_canonical_map()
        declared_category_tokens = (
            set(effective_mapping)
            | canonical_labels
        )
        overlap = declared_category_tokens & set(self.missing_tokens)
        if overlap:
            raise ValueError(
                "The same token cannot be both a category/canonical alias and "
                "set to null: "
                + ", ".join(sorted(overlap))
            )

        allowed = self.effective_allowed_values()
        if allowed is not None:
            unknown_targets = set(effective_mapping.values()) - allowed
            if unknown_targets:
                raise ValueError(
                    "canonical mapping targets must be declared canonical/allowed "
                    "values: "
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
    numeric_groups: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_time(self) -> TableSpec:
        if bool(self.time_column) != bool(self.order_column):
            raise ValueError("time_column and order_column must be specified together")
        if self.time_column and not self.subject_id:
            raise ValueError("subject_id is required for chronological checks")
        for numeric, by in self.numeric_groups.items():
            if not by or len(set(by)) != len(by) or numeric in by:
                raise ValueError(
                    "numeric_groups requires distinct grouping columns other than the measured column"
                )
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
                | set(table_spec.numeric_groups)
                | {column for by in table_spec.numeric_groups.values() for column in by}
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
                normalized = normalize_column_name(column)
                named_identifier = (
                    normalized in {"id", "mrn", "identifier"}
                    or normalized.endswith(("_id", "_mrn", "_identifier"))
                )
                if (column in identifiers or named_identifier) and (
                    spec.effective_canonical_map() or spec.missing_tokens
                ):
                    raise ValueError(
                        f"Schema cannot propose identifier edits: {table_name}.{column}"
                    )
                if any(
                    is_missing_like(value)
                    for value in spec.effective_canonical_map().values()
                ):
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
