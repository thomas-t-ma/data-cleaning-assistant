from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from .actions import ActionType, CleaningAction, CleaningPlan
from .models import DatasetReport, Issue
from .review import ActionFamily, group_actions
from .schema_draft import SchemaSuggestion, SchemaSuggestionKind
from .transforms import preview_action


class SheetCell(BaseModel):
    text: str
    changed: bool = False
    before: str | None = None
    after: str | None = None


class SheetRow(BaseModel):
    row_number: int
    cells: dict[str, SheetCell] = Field(default_factory=dict)


class SheetPreview(BaseModel):
    table: str
    columns: list[str]
    target_columns: list[str] = Field(default_factory=list)
    rows: list[SheetRow] = Field(default_factory=list)
    caption: str | None = None


class HumanExplanation(BaseModel):
    title: str
    summary: str
    context: str | None = None
    caution: str | None = None
    technical_label: str | None = None


class FamilyPresentation(BaseModel):
    id: str
    action_ids: list[str]
    action_type: str
    member_count: int
    affected_count: int
    decision: str
    explanation: HumanExplanation
    previews: list[SheetPreview] = Field(default_factory=list)


def display_value(value: Any) -> str:
    if value is None:
        return "(blank)"
    try:
        if pd.isna(value):
            return "(blank)"
    except (TypeError, ValueError):
        pass
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float):
        if math.isnan(value):
            return "(blank)"
        if math.isinf(value):
            return str(value)
        if value.is_integer():
            return str(int(value))
    text = str(value)
    if isinstance(value, str) and value != value.strip():
        return f'"{value}"'
    return text


def readable_name(name: str) -> str:
    cleaned = name.replace("_", " ").replace("-", " ").strip()
    words = cleaned.split()
    if not words:
        return "this field"

    acronyms = {
        "id": "ID",
        "dob": "DOB",
        "mrn": "MRN",
        "url": "URL",
        "uri": "URI",
        "uuid": "UUID",
        "doi": "DOI",
        "icd": "ICD",
    }
    output: list[str] = []
    for word in words:
        lowered = word.casefold()
        if lowered in acronyms:
            output.append(acronyms[lowered])
        elif word.isupper() and len(word) <= 8:
            output.append(word)
        else:
            output.append(word.title())
    return " ".join(output)


def _profile_for(
    report: DatasetReport | None,
    table: str,
    column: str | None,
):
    if report is None or column is None:
        return None
    for table_profile in report.tables:
        if table_profile.name != table:
            continue
        for column_profile in table_profile.columns:
            if column_profile.name == column:
                return column_profile
    return None


def _context_sentence(
    report: DatasetReport | None,
    table: str,
    column: str | None,
) -> str | None:
    profile = _profile_for(report, table, column)
    if profile is None or column is None:
        return None

    label = readable_name(column)
    type_phrase = {
        "date": "a date field",
        "datetime": "a date/time field",
        "numeric": "a numeric field",
        "identifier": "an identifier field",
        "categorical": "a category field",
        "free_text": "a free-text field",
        "text": "a text field",
    }.get(profile.inferred_type, f"a {profile.inferred_type.replace('_', ' ')} field")

    samples = [display_value(value) for value in profile.sample_values[:3]]
    if samples:
        return (
            f"{label} looks like {type_phrase}. "
            f"Example values include {', '.join(repr(value) for value in samples)}."
        )
    return f"{label} looks like {type_phrase}."


def action_explanation(
    action: CleaningAction,
    *,
    affected_count: int,
    report: DatasetReport | None = None,
) -> HumanExplanation:
    table = readable_name(action.table)
    column = readable_name(action.column) if action.column else None
    count_text = f"{affected_count:,} value" + ("" if affected_count == 1 else "s")

    if action.type == ActionType.STANDARDIZE_DATES and column:
        return HumanExplanation(
            title=f"Make {column} consistent",
            summary=(
                f"{count_text.capitalize()} in {column} can be written in one date format "
                "(YYYY-MM-DD). Values that cannot be parsed confidently will stay exactly as they are."
            ),
            context=_context_sentence(report, action.table, action.column),
            caution=(
                "Check that the highlighted examples are really dates and that ambiguous "
                "day/month values are not being reinterpreted."
            ),
            technical_label=action.type.value,
        )

    if action.type == ActionType.NORMALIZE_MISSING and column:
        tokens = [display_value(value) for value in action.parameters.get("values", [])]
        token_text = ", ".join(repr(value) for value in tokens[:4]) or "the flagged token"
        return HumanExplanation(
            title=f"Treat {token_text} as blank in {column}",
            summary=(
                f"{count_text.capitalize()} in {column} use a structural missing-value marker. "
                "The proposal converts only those exact markers to blank cells."
            ),
            context=_context_sentence(report, action.table, action.column),
            caution=(
                "Approve only if these labels truly mean 'no value recorded' rather than a "
                "meaningful category such as Unknown or Not applicable."
            ),
            technical_label=action.type.value,
        )

    if action.type == ActionType.TRIM_WHITESPACE and column:
        return HumanExplanation(
            title=f"Remove extra spaces from {column}",
            summary=(
                f"{count_text.capitalize()} have spaces before or after the visible text. "
                "The wording itself will not change."
            ),
            context=_context_sentence(report, action.table, action.column),
            caution="Internal spaces are preserved; only leading and trailing spaces are removed.",
            technical_label=action.type.value,
        )

    if action.type == ActionType.COERCE_NUMERIC and column:
        return HumanExplanation(
            title=f"Store {column} as numbers where possible",
            summary=(
                f"{count_text.capitalize()} look numeric but are stored as text or mixed types. "
                "Only values that parse cleanly will become numbers; everything else stays unchanged."
            ),
            context=_context_sentence(report, action.table, action.column),
            caution=(
                "Check that this field represents quantities rather than identifiers, codes, "
                "or values where formatting such as leading zeroes matters."
            ),
            technical_label=action.type.value,
        )

    if action.type == ActionType.NORMALIZE_CATEGORIES and column:
        mapping = action.parameters.get("mapping", {})
        pairs = list(mapping.items())
        if len(pairs) == 1:
            source, target = pairs[0]
            title = f'Use "{display_value(target)}" instead of "{display_value(source)}" in {column}'
            summary = (
                f"{count_text.capitalize()} use the first label. The proposal treats it as the "
                f"same category as {display_value(target)!r}."
            )
        else:
            title = f"Use consistent labels in {column}"
            summary = (
                f"{count_text.capitalize()} use alternate spellings, capitalization, or labels "
                "that the current plan would consolidate."
            )
        return HumanExplanation(
            title=title,
            summary=summary,
            context=_context_sentence(report, action.table, action.column),
            caution=(
                "Approve only when the before and after labels genuinely mean the same thing "
                "in this study."
            ),
            technical_label=action.type.value,
        )

    if action.type == ActionType.REMOVE_EXACT_DUPLICATES:
        return HumanExplanation(
            title=f"Remove exact duplicate rows from {table}",
            summary=(
                f"{affected_count:,} row" + ("" if affected_count == 1 else "s")
                + " exactly repeat another row. The first copy would be kept."
            ),
            context=(
                f"This review concerns entire rows in {table}, not just one column."
            ),
            caution=(
                "Identical rows can still represent legitimate repeated observations. "
                "Review the examples before deleting them."
            ),
            technical_label=action.type.value,
        )

    return HumanExplanation(
        title=action.title,
        summary=action.description,
        context=_context_sentence(report, action.table, action.column),
        technical_label=action.type.value,
    )


def schema_explanation(
    suggestion: SchemaSuggestion,
    *,
    report: DatasetReport | None = None,
) -> HumanExplanation:
    table = readable_name(suggestion.table)
    column = readable_name(suggestion.column) if suggestion.column else None

    if suggestion.kind == SchemaSuggestionKind.GRAIN_KEY:
        columns = [
            readable_name(str(value))
            for value in suggestion.proposed.get("columns", [])
        ]
        key_text = ", ".join(columns) or "the proposed key"
        return HumanExplanation(
            title=f"Use {key_text} to identify rows in {table}",
            summary=(
                f"The profiler found {key_text} to be a strong candidate for identifying "
                "individual rows in this table."
            ),
            caution=(
                "Approve only if this matches how the study actually defines a unique record."
            ),
            technical_label=suggestion.kind.value,
        )

    if suggestion.kind == SchemaSuggestionKind.CANONICAL_MAPPING and column:
        source = display_value(suggestion.proposed.get("source"))
        target = display_value(suggestion.proposed.get("target"))
        return HumanExplanation(
            title=f'Treat "{source}" as the same label as "{target}" in {column}',
            summary=(
                f"The dataset contains both {source!r} and {target!r}. "
                "Approving this tells later cleaning that the first is an alias of the second."
            ),
            context=_context_sentence(report, suggestion.table, suggestion.column),
            caution=(
                "This becomes part of the study dictionary, so approve only if the two labels "
                "are semantically equivalent."
            ),
            technical_label=suggestion.kind.value,
        )

    if suggestion.kind == SchemaSuggestionKind.MISSING_TOKEN and column:
        token = display_value(suggestion.proposed.get("token"))
        return HumanExplanation(
            title=f'Treat "{token}" as blank in {column}',
            summary=(
                f"The profiler found {token!r} being used like a missing-value marker in {column}."
            ),
            context=_context_sentence(report, suggestion.table, suggestion.column),
            caution=(
                "Approve only if this token means no value was recorded in this study."
            ),
            technical_label=suggestion.kind.value,
        )

    if suggestion.kind == SchemaSuggestionKind.FOREIGN_KEY:
        child_table = readable_name(str(suggestion.proposed.get("child_table", "")))
        child_column = readable_name(str(suggestion.proposed.get("child_column", "")))
        parent_table = readable_name(str(suggestion.proposed.get("parent_table", "")))
        parent_column = readable_name(str(suggestion.proposed.get("parent_column", "")))
        return HumanExplanation(
            title=f"Link {child_table} to {parent_table} using {child_column}",
            summary=(
                f"Values in {child_table}.{child_column} appear to reference "
                f"{parent_table}.{parent_column}."
            ),
            caution=(
                "Approve only if this is an intentional relationship in the study design."
            ),
            technical_label=suggestion.kind.value,
        )

    return HumanExplanation(
        title=suggestion.rationale,
        summary=suggestion.rationale,
        technical_label=suggestion.kind.value,
    )


def _selected_columns(
    frame: pd.DataFrame,
    target_columns: list[str],
    *,
    max_columns: int = 5,
) -> list[str]:
    columns = [str(column) for column in frame.columns]
    if not target_columns:
        return columns[:max_columns]

    positions = [
        columns.index(column)
        for column in target_columns
        if column in columns
    ]
    if not positions:
        return columns[:max_columns]

    selected: list[str] = []
    for position in positions:
        start = max(0, position - 2)
        end = min(len(columns), position + 3)
        for column in columns[start:end]:
            if column not in selected:
                selected.append(column)
    if len(selected) > max_columns:
        # Preserve every target and then fill remaining slots by proximity.
        targets = [column for column in target_columns if column in columns]
        extras = [column for column in selected if column not in targets]
        selected = targets + extras[: max(0, max_columns - len(targets))]
    return selected[:max_columns]


def _row_position(frame: pd.DataFrame, row_index: Any) -> int | None:
    for position, index in enumerate(frame.index):
        try:
            if index == row_index:
                return position
        except (TypeError, ValueError):
            continue
    return None


def action_sheet_preview(
    tables: dict[str, pd.DataFrame],
    action: CleaningAction,
    *,
    sample_limit: int = 5,
) -> SheetPreview:
    frame = tables[action.table]
    preview = preview_action(tables, action, sample_limit=sample_limit)

    if action.type == ActionType.REMOVE_EXACT_DUPLICATES:
        columns = _selected_columns(frame, [], max_columns=6)
        rows: list[SheetRow] = []
        for snapshot in preview.sample_rows:
            cells = {
                column: SheetCell(
                    text=display_value(
                        snapshot.values[column].value
                        if column in snapshot.values
                        else None
                    ),
                    changed=True,
                )
                for column in columns
            }
            rows.append(
                SheetRow(
                    row_number=snapshot.row_position + 2,
                    cells=cells,
                )
            )
        return SheetPreview(
            table=action.table,
            columns=columns,
            rows=rows,
            caption="Rows shown below are exact duplicates that would be removed.",
        )

    if action.column is None:
        return SheetPreview(table=action.table, columns=[], rows=[])

    columns = _selected_columns(frame, [action.column])
    changed_by_row = {change.row_index: change for change in preview.sample_changes}
    rows: list[SheetRow] = []
    for row_index, change in changed_by_row.items():
        position = _row_position(frame, row_index)
        if position is None:
            continue
        row = frame.iloc[position]
        cells: dict[str, SheetCell] = {}
        for column in columns:
            if column == action.column:
                cells[column] = SheetCell(
                    text=display_value(row[column]),
                    changed=True,
                    before=display_value(change.old_value.value),
                    after=display_value(change.new_value.value),
                )
            else:
                cells[column] = SheetCell(text=display_value(row[column]))
        rows.append(SheetRow(row_number=position + 2, cells=cells))

    return SheetPreview(
        table=action.table,
        columns=columns,
        target_columns=[action.column],
        rows=rows,
        caption=(
            f"Example rows for {readable_name(action.column)} "
            f"in {readable_name(action.table)}."
        ),
    )


def schema_sheet_previews(
    tables: dict[str, pd.DataFrame],
    suggestion: SchemaSuggestion,
    *,
    sample_limit: int = 5,
) -> list[SheetPreview]:
    if suggestion.kind == SchemaSuggestionKind.FOREIGN_KEY:
        previews: list[SheetPreview] = []
        for table_key, column_key in (
            ("child_table", "child_column"),
            ("parent_table", "parent_column"),
        ):
            table = str(suggestion.proposed.get(table_key, ""))
            column = str(suggestion.proposed.get(column_key, ""))
            if table not in tables or column not in tables[table].columns:
                continue
            frame = tables[table]
            columns = _selected_columns(frame, [column])
            rows = [
                SheetRow(
                    row_number=position + 2,
                    cells={
                        col: SheetCell(text=display_value(frame.iloc[position][col]))
                        for col in columns
                    },
                )
                for position in range(min(len(frame), sample_limit))
            ]
            previews.append(
                SheetPreview(
                    table=table,
                    columns=columns,
                    target_columns=[column],
                    rows=rows,
                )
            )
        return previews

    if suggestion.table not in tables:
        return []
    frame = tables[suggestion.table]

    target_columns: list[str] = []
    if suggestion.column and suggestion.column in frame.columns:
        target_columns = [suggestion.column]
    elif suggestion.kind == SchemaSuggestionKind.GRAIN_KEY:
        target_columns = [
            str(column)
            for column in suggestion.proposed.get("columns", [])
            if str(column) in frame.columns
        ]

    columns = _selected_columns(frame, target_columns)
    positions: list[int] = []

    if suggestion.kind == SchemaSuggestionKind.CANONICAL_MAPPING and suggestion.column:
        source = str(suggestion.proposed.get("source"))
        positions = [
            position
            for position, value in enumerate(frame[suggestion.column].tolist())
            if str(value) == source
        ][:sample_limit]
    elif suggestion.kind == SchemaSuggestionKind.MISSING_TOKEN and suggestion.column:
        token = str(suggestion.proposed.get("token"))
        positions = [
            position
            for position, value in enumerate(frame[suggestion.column].tolist())
            if str(value) == token
        ][:sample_limit]

    if not positions:
        positions = list(range(min(len(frame), sample_limit)))

    rows = [
        SheetRow(
            row_number=position + 2,
            cells={
                column: SheetCell(text=display_value(frame.iloc[position][column]))
                for column in columns
            },
        )
        for position in positions
    ]
    return [
        SheetPreview(
            table=suggestion.table,
            columns=columns,
            target_columns=target_columns,
            rows=rows,
        )
    ]




def issue_sheet_preview(
    tables: dict[str, pd.DataFrame],
    issue: Issue,
    *,
    sample_limit: int = 5,
) -> SheetPreview | None:
    if issue.table not in tables:
        return None
    frame = tables[issue.table]
    target_columns = (
        [issue.column]
        if issue.column is not None and issue.column in frame.columns
        else []
    )
    columns = _selected_columns(frame, target_columns, max_columns=6)
    positions = [
        int(position)
        for position in issue.row_indices_sample[:sample_limit]
        if 0 <= int(position) < len(frame)
    ]
    if not positions:
        positions = list(range(min(len(frame), sample_limit)))

    rows = [
        SheetRow(
            row_number=position + 2,
            cells={
                column: SheetCell(text=display_value(frame.iloc[position][column]))
                for column in columns
            },
        )
        for position in positions
    ]
    return SheetPreview(
        table=issue.table,
        columns=columns,
        target_columns=target_columns,
        rows=rows,
        caption=(
            f"Example rows related to {readable_name(issue.column)}."
            if issue.column
            else f"Example rows from {readable_name(issue.table)}."
        ),
    )

def family_id(family: ActionFamily) -> str:
    payload = json.dumps(
        {
            "origin": family.origin,
            "action_type": family.action_type,
            "parameters": family.parameters,
            "action_ids": sorted(family.action_ids),
        },
        sort_keys=True,
        default=str,
    )
    return "family_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _family_decision(actions: list[CleaningAction]) -> str:
    decisions = {action.decision.value for action in actions}
    if len(decisions) == 1:
        return next(iter(decisions))
    return "mixed"


def family_presentations(
    tables: dict[str, pd.DataFrame],
    plan: CleaningPlan,
    *,
    report: DatasetReport | None = None,
    max_previews: int = 3,
) -> list[FamilyPresentation]:
    output: list[FamilyPresentation] = []
    for family in group_actions(tables, plan):
        members = [member.action for member in family.members]
        representative = members[0]
        previews = [
            action_sheet_preview(tables, action, sample_limit=3)
            for action in members[:max_previews]
        ]
        explanation = action_explanation(
            representative,
            affected_count=family.affected_count,
            report=report,
        )
        if len(members) > 1:
            targets = [
                readable_name(action.column or action.table)
                for action in members[:4]
            ]
            explanation = explanation.model_copy(
                update={
                    "title": (
                        f"{explanation.title} across {len(members)} fields"
                    ),
                    "summary": (
                        f"This same cleanup rule affects {family.affected_count:,} "
                        f"cells across {len(members)} fields: {', '.join(targets)}"
                        + ("…" if len(members) > 4 else ".")
                    ),
                }
            )
        output.append(
            FamilyPresentation(
                id=family_id(family),
                action_ids=family.action_ids,
                action_type=family.action_type,
                member_count=len(members),
                affected_count=family.affected_count,
                decision=_family_decision(members),
                explanation=explanation,
                previews=previews,
            )
        )
    return output


@dataclass(frozen=True)
class ActionContextPayload:
    dataset_name: str
    table: str
    column: str | None
    nearby_columns: list[str]
    inferred_type: str | None
    sample_values: list[str]
    deterministic_title: str
    deterministic_summary: str
    before_after_examples: list[tuple[str, str]]


def action_context_payload(
    *,
    dataset_name: str,
    tables: dict[str, pd.DataFrame],
    action: CleaningAction,
    explanation: HumanExplanation,
    report: DatasetReport | None,
) -> ActionContextPayload:
    frame = tables[action.table]
    nearby_columns = _selected_columns(
        frame,
        [action.column] if action.column else [],
        max_columns=7,
    )
    profile = _profile_for(report, action.table, action.column)
    samples = (
        [display_value(value) for value in profile.sample_values[:5]]
        if profile is not None
        else []
    )
    preview = preview_action(tables, action, sample_limit=5)
    examples = [
        (
            display_value(change.old_value.value),
            display_value(change.new_value.value),
        )
        for change in preview.sample_changes
    ]
    return ActionContextPayload(
        dataset_name=dataset_name,
        table=action.table,
        column=action.column,
        nearby_columns=nearby_columns,
        inferred_type=profile.inferred_type if profile is not None else None,
        sample_values=samples,
        deterministic_title=explanation.title,
        deterministic_summary=explanation.summary,
        before_after_examples=examples,
    )
