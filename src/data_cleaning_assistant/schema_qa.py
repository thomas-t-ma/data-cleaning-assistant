from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .inference import is_missing_like, parse_date, parse_number
from .models import Issue, Severity
from .study_schema import StudySchema


def _issue(
    table: str,
    kind: str,
    column: str | None,
    indices: list[int],
    *,
    title: str,
    description: str,
    evidence: dict | None = None,
) -> Issue:
    return Issue(
        id=f"schema_{kind}_{table}_{column or 'table'}",
        type=kind,
        severity=Severity.HIGH,
        table=table,
        column=column,
        title=title,
        description=description,
        confidence=1.0,
        affected_count=len(indices),
        row_indices_sample=indices[:10],
        evidence=evidence or {},
        suggested_action="Review against the study dictionary; no automatic data change.",
        auto_fix_safe=False,
    )


def _chronology_issues(
    table: str,
    frame: pd.DataFrame,
    subject_id: str,
    time_column: str,
    order_column: str,
) -> list[Issue]:
    groups: dict[str, list[tuple[float, pd.Timestamp, int]]] = defaultdict(list)
    for index, (subject, raw_time, raw_order) in enumerate(
        zip(
            frame[subject_id].tolist(),
            frame[time_column].tolist(),
            frame[order_column].tolist(),
            strict=True,
        )
    ):
        if is_missing_like(subject):
            continue
        visit = parse_number(raw_order)
        date = parse_date(raw_time)
        if visit is None or date is None:
            continue
        groups[str(subject)].append((visit, pd.Timestamp(date), index))

    affected: set[int] = set()
    for entries in groups.values():
        entries.sort(key=lambda item: (item[0], item[2]))
        latest: pd.Timestamp | None = None
        latest_order: float | None = None
        for visit, date, index in entries:
            if latest is not None and latest_order is not None and visit > latest_order and date < latest:
                affected.add(index)
            if latest_order is None or visit > latest_order:
                latest, latest_order = date, visit
            elif visit == latest_order and latest is not None:
                latest = max(latest, date)

    if not affected:
        return []
    return [
        _issue(
            table,
            "schema_visit_chronology",
            time_column,
            sorted(affected),
            title="Visit chronology conflicts with declared visit order",
            description=(
                "A later ordered visit has an earlier date for the same subject. "
                "The data are preserved for study review."
            ),
            evidence={
                "subject_column": subject_id,
                "order_column": order_column,
                "affected_rows": len(affected),
            },
        )
    ]


def schema_issues(
    tables: dict[str, pd.DataFrame],
    schema: StudySchema,
) -> list[Issue]:
    """Validate explicitly declared study constraints without editing values."""
    schema.validate_dataset(tables)
    issues: list[Issue] = []
    for table_name, spec in schema.tables.items():
        frame = tables[table_name]

        if spec.grain_key:
            complete = ~frame[spec.grain_key].apply(
                lambda row: any(is_missing_like(value) for value in row),
                axis=1,
            )
            missing = [int(index) for index in frame.index[~complete]]
            if missing:
                issues.append(
                    _issue(
                        table_name,
                        "schema_missing_grain_key",
                        None,
                        missing,
                        title="Declared grain key contains missing values",
                        description="One or more required grain-key fields are missing.",
                        evidence={"grain_key": spec.grain_key},
                    )
                )

            repeated = frame.loc[complete].duplicated(subset=spec.grain_key, keep=False)
            duplicate_indices = [int(i) for i in repeated.index[repeated]]
            if duplicate_indices:
                issues.append(
                    _issue(
                        table_name,
                        "schema_duplicate_grain",
                        None,
                        duplicate_indices,
                        title="Declared grain key is not unique",
                        description=(
                            "Multiple rows share the same declared grain key. This is a QA "
                            "finding, not permission to delete repeated measurements."
                        ),
                        evidence={"grain_key": spec.grain_key},
                    )
                )

        for column, column_spec in spec.columns.items():
            allowed = column_spec.effective_allowed_values()
            if allowed is None:
                continue
            authorized_variants = set(column_spec.effective_canonical_map())
            declared_missing = set(column_spec.missing_tokens)
            unexpected = [
                int(index)
                for index, value in frame[column].items()
                if not is_missing_like(value)
                and str(value) not in allowed
                and str(value) not in authorized_variants
                and str(value) not in declared_missing
            ]
            if unexpected:
                issues.append(
                    _issue(
                        table_name,
                        "schema_unexpected_category",
                        column,
                        unexpected,
                        title="Values outside the declared category dictionary",
                        description=(
                            "Observed values are not canonical/allowed values, authorized "
                            "aliases, or explicitly declared null tokens."
                        ),
                        evidence={"allowed_value_count": len(allowed)},
                    )
                )

        if spec.subject_id and spec.time_column and spec.order_column:
            issues.extend(
                _chronology_issues(
                    table_name,
                    frame,
                    spec.subject_id,
                    spec.time_column,
                    spec.order_column,
                )
            )

    for relation in schema.foreign_keys:
        child = tables[relation.child_table][relation.child_column]
        parent = tables[relation.parent_table][relation.parent_column]
        valid_parent = {str(v) for v in parent if not is_missing_like(v)}
        unmatched = [
            int(i)
            for i, value in child.items()
            if not is_missing_like(value) and str(value) not in valid_parent
        ]
        if unmatched:
            issues.append(
                _issue(
                    relation.child_table,
                    "schema_unmatched_foreign_key",
                    relation.child_column,
                    unmatched,
                    title="Child records lack a declared parent record",
                    description=(
                        "Some child-table keys are absent from the declared parent table. "
                        "No keys or rows are altered automatically."
                    ),
                    evidence={
                        "parent_table": relation.parent_table,
                        "parent_column": relation.parent_column,
                    },
                )
            )

        parent_duplicates = [
            int(i)
            for i in parent.index[
                parent.map(lambda value: not is_missing_like(value))
                & parent.astype(str).duplicated(keep=False)
            ]
        ]
        if parent_duplicates:
            issues.append(
                _issue(
                    relation.parent_table,
                    "schema_nonunique_parent_key",
                    relation.parent_column,
                    parent_duplicates,
                    title="Declared parent key is not unique",
                    description="A declared foreign-key target occurs more than once.",
                    evidence={
                        "child_table": relation.child_table,
                        "child_column": relation.child_column,
                    },
                )
            )

    return issues
