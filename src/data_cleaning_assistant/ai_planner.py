from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from .actions import ActionOrigin, ActionType, CleaningAction, CleaningPlan
from .ai_context import build_ai_context
from .ai_models import (
    AIActionProposal,
    AIPlannerContext,
    AIPlannerResponse,
    AIRejectedProposal,
)
from .ai_provider import StructuredAIProvider
from .engine import profile_dataset
from .inference import looks_like_identifier_name
from .io import load_dataset
from .models import DatasetReport, Issue
from .planner import file_sha256, propose_actions


SYSTEM_PROMPT = """You are a conservative data-quality planning assistant.

You NEVER edit data directly. You only propose structured CleaningAction candidates that
will be independently validated and reviewed by a human.

Scientific/research safety rules:
- Treat suspicious as different from wrong.
- Never impute identifiers or missing values.
- Never delete outliers.
- Never infer clinical truth from column names.
- Never propose row deletion. Exact duplicate rows are handled by deterministic rules and human review.
- Never remove non-exact duplicates.
- Never merge fuzzy categories unless the supplied detector evidence strongly suggests the
  two observed labels mean the same thing.
- Never invent category labels or missing-value tokens.
- Defer ambiguous cases rather than forcing a change.
- Use only the supported action types and parameter shapes described by the user message.
- issue_ids are optional provenance. If you are not certain of the exact supplied ID, leave the list empty.
- Never invent issue IDs.
"""


@dataclass(slots=True)
class AIPlanResult:
    plan: CleaningPlan
    context: AIPlannerContext
    response: AIPlannerResponse
    rejected: list[AIRejectedProposal]
    model: str
    provider: str


def _stable_ai_action_id(
    action_type: ActionType,
    table: str,
    column: str | None,
    parameters: dict[str, Any],
) -> str:
    payload = json.dumps(
        {
            "type": action_type.value,
            "table": table,
            "column": column,
            "parameters": parameters,
        },
        sort_keys=True,
        default=str,
    )
    return f"ai_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def _effect_key(action: CleaningAction) -> str:
    return json.dumps(
        {
            "type": action.type.value,
            "table": action.table,
            "column": action.column,
            "parameters": action.parameters,
        },
        sort_keys=True,
        default=str,
    )


def _matching_issues(
    report: DatasetReport,
    *,
    table: str,
    column: str | None,
    issue_types: set[str],
) -> list[Issue]:
    return [
        issue
        for issue in report.issues
        if issue.type in issue_types
        and issue.table == table
        and issue.column == column
    ]


def _allowed_category_pairs(issues: list[Issue]) -> set[frozenset[str]]:
    pairs: set[frozenset[str]] = set()
    for issue in issues:
        if issue.type == "category_inconsistency":
            for group in issue.evidence.get("variant_groups", []):
                variants = [str(value) for value in group]
                for left, right in combinations(variants, 2):
                    pairs.add(frozenset({left, right}))
        elif issue.type == "fuzzy_category_match":
            for item in issue.evidence.get("candidate_pairs", []):
                if not isinstance(item, dict):
                    continue
                left = item.get("left")
                right = item.get("right")
                if left is not None and right is not None:
                    pairs.add(frozenset({str(left), str(right)}))
    return pairs


def _validated_issue_ids(
    proposal: AIActionProposal,
    matching: list[Issue],
) -> list[str]:
    """Attach only independently matched detector provenance.

    Model-supplied issue IDs are advisory metadata, not a mutation-safety gate.
    If the model copies an ID incorrectly, local table/column/type evidence still
    determines whether the action is allowed.
    """
    matching_ids = {issue.id for issue in matching}
    supplied = set(proposal.issue_ids)
    valid_supplied = supplied & matching_ids
    return sorted(valid_supplied or matching_ids)


def _column_exists(
    tables: dict[str, pd.DataFrame],
    table: str,
    column: str | None,
) -> bool:
    return table in tables and column is not None and column in tables[table].columns


def _validated_action(
    proposal: AIActionProposal,
    report: DatasetReport,
    tables: dict[str, pd.DataFrame],
) -> tuple[CleaningAction | None, str | None]:
    if proposal.table not in tables:
        return None, f"Unknown table {proposal.table!r}."

    parameters: dict[str, Any]
    matching: list[Issue]
    title: str

    if proposal.type == ActionType.NORMALIZE_CATEGORIES:
        if not _column_exists(tables, proposal.table, proposal.column):
            return None, "Category normalization requires an existing column."
        matching = _matching_issues(
            report,
            table=proposal.table,
            column=proposal.column,
            issue_types={"category_inconsistency", "fuzzy_category_match"},
        )
        if not matching:
            return None, "No category-variant detector evidence supports this proposal."

        raw_mapping = proposal.parameters.get("mapping")
        if not isinstance(raw_mapping, dict) or not raw_mapping:
            return None, "Category normalization requires a non-empty mapping."

        observed = {str(value) for value in tables[proposal.table][proposal.column].tolist()}
        allowed_pairs = _allowed_category_pairs(matching)
        mapping: dict[str, str] = {}
        for raw_key, raw_value in raw_mapping.items():
            key, value = str(raw_key), str(raw_value)
            if key == value:
                continue
            if key not in observed or value not in observed:
                return None, "AI category mappings may only use observed labels."
            if frozenset({key, value}) not in allowed_pairs:
                return None, (
                    f"Mapping {key!r} -> {value!r} was not supported by detector candidate pairs."
                )
            mapping[key] = value
        if not mapping:
            return None, "Category mapping had no effective supported replacements."
        if set(mapping).intersection(mapping.values()):
            return None, (
                "Category mappings may not contain cycles or chains; map variants directly "
                "to a canonical observed label."
            )
        parameters = {"mapping": mapping}
        title = "AI: merge evidence-supported category variants"

    elif proposal.type == ActionType.NORMALIZE_MISSING:
        if not _column_exists(tables, proposal.table, proposal.column):
            return None, "Missing-value normalization requires an existing column."
        matching = _matching_issues(
            report,
            table=proposal.table,
            column=proposal.column,
            issue_types={"hidden_missing_values"},
        )
        if not matching:
            return None, "No hidden-missing detector evidence supports this proposal."
        allowed = {
            str(token)
            for issue in matching
            for token in issue.evidence.get("tokens", {})
        }
        raw_values = proposal.parameters.get("values")
        if not isinstance(raw_values, list) or not raw_values:
            return None, "Missing normalization requires a non-empty values list."
        values = sorted({str(value) for value in raw_values})
        if not set(values).issubset(allowed):
            return None, "AI may only normalize tokens already detected as missing-like."
        parameters = {"values": values}
        title = "AI: normalize detected missing-value tokens"

    elif proposal.type == ActionType.STANDARDIZE_DATES:
        if not _column_exists(tables, proposal.table, proposal.column):
            return None, "Date standardization requires an existing column."
        matching = _matching_issues(
            report,
            table=proposal.table,
            column=proposal.column,
            issue_types={"malformed_or_mixed_dates"},
        )
        if not matching:
            return None, "No mixed-date detector evidence supports this proposal."
        parameters = {"format": "%Y-%m-%d", "unparseable": "keep"}
        title = "AI: standardize parseable dates"

    elif proposal.type == ActionType.COERCE_NUMERIC:
        if not _column_exists(tables, proposal.table, proposal.column):
            return None, "Numeric coercion requires an existing column."
        if looks_like_identifier_name(proposal.column or ""):
            return None, "Identifier-like columns cannot be proposed for numeric coercion."
        matching = _matching_issues(
            report,
            table=proposal.table,
            column=proposal.column,
            issue_types={"mixed_numeric_values"},
        )
        if not matching:
            return None, "No mixed-numeric detector evidence supports this proposal."
        parameters = {"unparseable": "keep"}
        title = "AI: coerce parseable numeric values"

    elif proposal.type == ActionType.TRIM_WHITESPACE:
        if not _column_exists(tables, proposal.table, proposal.column):
            return None, "Whitespace trimming requires an existing column."
        matching = _matching_issues(
            report,
            table=proposal.table,
            column=proposal.column,
            issue_types={"surrounding_whitespace"},
        )
        if not matching:
            return None, "No whitespace detector evidence supports this proposal."
        parameters = {}
        title = "AI: trim surrounding whitespace"

    elif proposal.type == ActionType.REMOVE_EXACT_DUPLICATES:
        return None, (
            "AI is not allowed to propose row deletion. Exact duplicates are surfaced "
            "by deterministic rules for explicit human review."
        )

    else:
        return None, f"Unsupported action type {proposal.type.value!r}."

    issue_ids = _validated_issue_ids(proposal, matching)

    detector_confidence = max(issue.confidence for issue in matching)
    confidence = min(proposal.confidence, detector_confidence)

    return (
        CleaningAction(
            id=_stable_ai_action_id(
                proposal.type,
                proposal.table,
                proposal.column,
                parameters,
            ),
            type=proposal.type,
            table=proposal.table,
            column=proposal.column,
            title=title,
            description=proposal.rationale.strip(),
            parameters=parameters,
            confidence=confidence,
            issue_ids=issue_ids,
            origin=ActionOrigin.AI,
        ),
        None,
    )


def _build_user_prompt(context: AIPlannerContext) -> str:
    rules = {
        "normalize_categories": {"mapping": {"observed_label": "observed_label"}},
        "normalize_missing": {"values": ["detected_missing_token"]},
        "standardize_dates": {"format": "%Y-%m-%d", "unparseable": "keep"},
        "coerce_numeric": {"unparseable": "keep"},
        "trim_whitespace": {},
    }
    return (
        "Review the following compact profiler context and propose only defensible cleaning "
        "actions. It is acceptable to propose no AI actions. The deterministic rule engine will "
        "still provide baseline actions separately.\n\n"
        "Supported parameter shapes:\n"
        + json.dumps(rules, indent=2, sort_keys=True)
        + "\n\nProfiler context:\n"
        + context.model_dump_json(indent=2)
    )


def resolve_model(
    provider: StructuredAIProvider,
    requested: str | None,
) -> str:
    models = provider.list_models()
    if requested:
        if requested not in models:
            available = ", ".join(models) if models else "<none>"
            raise ValueError(
                f"Model {requested!r} is not installed in {provider.name}. "
                f"Installed models: {available}"
            )
        return requested

    if not models:
        raise ValueError(
            f"No models are installed in {provider.name}. Install or pull a local model first."
        )
    if len(models) > 1:
        raise ValueError(
            "Multiple local models are installed. Choose one with --model. "
            f"Available: {', '.join(models)}"
        )
    return models[0]


def propose_ai_plan(
    path: str | Path,
    *,
    provider: StructuredAIProvider,
    model: str,
    include_rules: bool = True,
) -> AIPlanResult:
    resolved = Path(path).expanduser().resolve()
    loaded = load_dataset(resolved)
    report = profile_dataset(resolved)
    context = build_ai_context(report)

    response = provider.generate_structured(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=_build_user_prompt(context),
        response_model=AIPlannerResponse,
    )

    accepted: list[CleaningAction] = []
    rejected: list[AIRejectedProposal] = []
    for proposal in response.actions:
        action, error = _validated_action(proposal, report, loaded.tables)
        if action is None:
            rejected.append(
                AIRejectedProposal(
                    type=proposal.type,
                    table=proposal.table,
                    column=proposal.column,
                    parameters=proposal.parameters,
                    rationale=proposal.rationale,
                    reason=error or "Proposal failed validation.",
                )
            )
        else:
            accepted.append(action)

    actions: list[CleaningAction] = []
    if include_rules:
        actions.extend(propose_actions(resolved).actions)

    existing = {_effect_key(action) for action in actions}
    for action in accepted:
        key = _effect_key(action)
        if key not in existing:
            actions.append(action)
            existing.add(key)

    return AIPlanResult(
        plan=CleaningPlan(
            source_path=str(resolved),
            source_sha256=file_sha256(resolved),
            actions=actions,
            planner="hybrid_ai" if include_rules else "ai_only",
            model=model,
        ),
        context=context,
        response=response,
        rejected=rejected,
        model=model,
        provider=provider.name,
    )
