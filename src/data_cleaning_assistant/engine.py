from __future__ import annotations

from pathlib import Path

from .detectors import (
    detect_basic_issues,
    detect_fuzzy_categories,
    detect_multiselect_component_variants,
)
from .detectors.basic import _iqr_outliers
from .io import load_dataset
from .inference import is_missing_like, parse_number
from .models import CrossTableRelationship, DatasetReport, GrainInference
from .numeric_scope import heterogeneous_measurement_metadata, unscoped_measurement_issue
from .profiling import profile_table
from .schema_qa import schema_issues
from .study_schema import StudySchema
from .structure import (
    infer_candidate_keys,
    infer_grain,
    infer_relationships,
    repeated_entity_issues,
)


def _declared_relationships(tables, schema: StudySchema) -> list[CrossTableRelationship]:
    relationships = []
    for fk in schema.foreign_keys:
        child = tables[fk.child_table][fk.child_column]
        parent = tables[fk.parent_table][fk.parent_column]
        child_values = [
            str(v) for v in child.tolist() if not is_missing_like(v)
        ]
        parent_values = [
            str(v) for v in parent.tolist() if not is_missing_like(v)
        ]
        child_set, parent_set = set(child_values), set(parent_values)
        shared = child_set & parent_set
        parent_unique = len(parent_values) == len(parent_set)
        child_unique = len(child_values) == len(child_set)
        if parent_unique and child_unique:
            cardinality = "one-to-one"
        elif parent_unique:
            cardinality = "one-to-many"
        elif child_unique:
            cardinality = "many-to-one"
        else:
            cardinality = "many-to-many"
        relationships.append(
            CrossTableRelationship(
                left_table=fk.parent_table,
                left_column=fk.parent_column,
                right_table=fk.child_table,
                right_column=fk.child_column,
                relationship=cardinality,
                shared_value_count=len(shared),
                left_coverage=len(shared) / len(parent_set) if parent_set else 0.0,
                right_coverage=len(shared) / len(child_set) if child_set else 0.0,
                confidence=1.0,
                source="declared",
            )
        )
    return relationships


def _grouped_outliers(table: str, frame, column: str, by: list[str]):
    """No pooling across declared assays/units; no inference of grouping."""
    issues = []
    eligible = frame[by].apply(
        lambda row: all(not is_missing_like(v) for v in row),
        axis=1,
    )
    subset = frame.loc[eligible]
    for _, group in subset.groupby(by, dropna=True, sort=False):
        findings = _iqr_outliers(table, column, group[column])
        for issue in findings:
            # The detector uses positions relative to the group.
            issue.row_indices_sample = [
                int(group.index[position]) for position in issue.row_indices_sample
            ]
            issue.description += (
                " Comparison is within the study-defined assay/unit group only."
            )
            issue.evidence = {
                "grouped": True,
                "group_by": by,
                "group_size": len(group),
                "lower_fence": issue.evidence["lower_fence"],
                "upper_fence": issue.evidence["upper_fence"],
            }
            issues.append(issue)
    return issues


def profile_dataset(path: str | Path, *, schema: StudySchema | None = None) -> DatasetReport:
    loaded = load_dataset(path)
    if schema is not None:
        schema.validate_dataset(loaded.tables)
    table_profiles = []
    issues = []

    for table_name, frame in loaded.tables.items():
        profile = profile_table(table_name, frame)
        profile.candidate_keys = infer_candidate_keys(frame)
        profile.grain = infer_grain(table_name, frame)
        table_spec = schema.tables.get(table_name) if schema else None
        if table_spec and table_spec.grain_key:
            profile.grain = GrainInference(
                label="declared: " + ", ".join(table_spec.grain_key),
                confidence=1.0,
                evidence=["Declared in the study schema; not inferred from name or repetition"],
            )
        table_profiles.append(profile)

        basic = detect_basic_issues(table_name, frame)
        if table_spec and table_spec.numeric_groups:
            basic = [
                issue for issue in basic
                if not (
                    issue.type == "statistical_outlier"
                    and issue.column in table_spec.numeric_groups
                )
            ]
            for column, by in table_spec.numeric_groups.items():
                basic.extend(_grouped_outliers(table_name, frame, column, by))

        scoped = table_spec.numeric_groups if table_spec else {}
        # A pooled test is inappropriate even if it happens to detect *zero*
        # outliers. Identify heterogeneous generic measurement columns first.
        unscoped = {
            str(column)
            for column in frame.columns
            if column not in scoped
            and heterogeneous_measurement_metadata(frame, str(column))
            and sum(parse_number(v) is not None for v in frame[column]) >= 8
        }
        pooled_findings = {}
        safe_basic = []
        for issue in basic:
            if (
                issue.type == "statistical_outlier"
                and issue.column in unscoped
            ):
                pooled_findings[issue.column] = issue
                continue
            safe_basic.append(issue)
        for column in sorted(unscoped):
            safe_basic.append(
                unscoped_measurement_issue(
                    table_name, column, frame, pooled_findings.get(column)
                )
            )
        issues.extend(safe_basic)
        issues.extend(detect_fuzzy_categories(table_name, frame))
        issues.extend(detect_multiselect_component_variants(table_name, frame))
        issues.extend(repeated_entity_issues(table_name, frame))

    if schema is not None:
        issues.extend(schema_issues(loaded.tables, schema))

    return DatasetReport(
        source_path=str(loaded.path),
        source_type=loaded.source_type,
        tables=table_profiles,
        issues=issues,
        relationships=(
            _declared_relationships(loaded.tables, schema)
            if schema is not None and schema.foreign_keys
            else infer_relationships(loaded.tables)
        ),
    )
