from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .actions import ActionOrigin, CleaningAction, CleaningPlan
from .ai_candidates import AICandidate, build_ai_candidates
from .ai_context import build_ai_context
from .ai_models import AIPlannerContext, AIPlannerResponse, AIRejectedProposal, CandidateDecision
from .ai_provider import StructuredAIProvider
from .engine import profile_dataset
from .io import load_dataset
from .planner import file_sha256, propose_actions


SYSTEM_PROMPT = """You are a conservative research-data quality reviewer.

You do NOT author cleaning operations. Local deterministic code has already constructed the only
candidate actions you may consider. Your job is semantic judgment only.

For every supplied candidate:
- choose accept only when the proposed change is clearly justified by the supplied evidence;
- choose defer when scientific meaning is ambiguous;
- never invent a candidate ID;
- never alter table names, column names, mappings, tokens, formats, or parameters;
- treat suspicious as different from wrong;
- for fuzzy category merges, accept only when the labels clearly mean the same thing;
- semantically different stages, grades, visits, timepoints, doses, outcomes, or specimen types
  must remain distinct even when their labels look similar;
- safe formatting operations such as trimming surrounding whitespace may be accepted when the
  proposed change is exactly bounded;
- numeric/date proposals preserve unparseable values, so judge whether the field itself is
  appropriately numeric/date-like rather than trying to repair the bad value;
- if uncertain, defer.

Return exactly one selection for each candidate ID.
"""


@dataclass(slots=True)
class AIPlanResult:
    plan: CleaningPlan
    context: AIPlannerContext
    response: AIPlannerResponse
    rejected: list[AIRejectedProposal]
    candidates: list[AICandidate]
    model: str
    provider: str


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


def _build_user_prompt(context: AIPlannerContext) -> str:
    return (
        "Review every candidate below. Return one accept/defer selection per candidate ID. "
        "Do not create any action yourself.\n\n"
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
    candidates = build_ai_candidates(report, loaded.tables)
    context = build_ai_context(report, candidates)

    response = provider.generate_structured(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=_build_user_prompt(context),
        response_model=AIPlannerResponse,
    )

    candidate_by_id = {candidate.id: candidate for candidate in candidates}
    seen: set[str] = set()
    accepted: list[CleaningAction] = []
    rejected: list[AIRejectedProposal] = []

    for selection in response.selections:
        candidate = candidate_by_id.get(selection.candidate_id)
        if candidate is None:
            rejected.append(
                AIRejectedProposal(
                    candidate_id=selection.candidate_id,
                    rationale=selection.rationale,
                    reason="Model referenced an unknown candidate ID.",
                )
            )
            continue
        if selection.candidate_id in seen:
            rejected.append(
                AIRejectedProposal(
                    candidate_id=selection.candidate_id,
                    type=candidate.action.type,
                    table=candidate.action.table,
                    column=candidate.action.column,
                    parameters=candidate.action.parameters,
                    rationale=selection.rationale,
                    reason="Model returned the same candidate ID more than once.",
                )
            )
            continue
        seen.add(selection.candidate_id)

        if selection.decision == CandidateDecision.DEFER:
            continue

        action = candidate.action.model_copy(deep=True)
        action.description = selection.rationale.strip() or candidate.context.question
        action.confidence = min(action.confidence, selection.confidence)
        action.origin = ActionOrigin.AI
        accepted.append(action)

    # Missing selections are treated as defer, never as acceptance.
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
        candidates=candidates,
        model=model,
        provider=provider.name,
    )
