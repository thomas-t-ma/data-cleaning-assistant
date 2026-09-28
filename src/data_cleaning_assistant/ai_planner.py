from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .actions import ActionOrigin, CleaningAction, CleaningPlan, DeferredReview
from .ai_candidates import AICandidate, build_ai_candidates
from .ai_context import build_ai_context
from .ai_models import (
    AICandidateSelection,
    AIPlannerContext,
    AIPlannerResponse,
    AIRejectedProposal,
    CandidateDecision,
)
from .ai_provider import AIProviderError, StructuredAIProvider
from .engine import profile_dataset
from .io import load_dataset
from .planner import file_sha256, propose_actions
from .study_schema import StudySchema


BASE_SYSTEM_PROMPT = """You are a conservative research-data quality reviewer.

You do NOT author cleaning operations. Local deterministic code has already constructed the only
candidate actions you may consider. Your job is semantic judgment only.

For every supplied candidate:
- choose accept only when the proposed change is clearly justified by the supplied evidence;
- choose defer when scientific meaning is ambiguous;
- never invent a candidate ID;
- never alter table names, column names, mappings, tokens, formats, or parameters;
- treat suspicious as different from wrong;
- for fuzzy category merges, accept only when the labels clearly mean the same thing;
- when evidence says numeric_zero_padding_equivalence, the surrounding text and numeric value are
  deterministically equal and only leading-zero formatting differs; treat that as strong evidence
  of a formatting-only variant unless the supplied context indicates a distinct coded category;
- when evidence says single_edit_typo_pattern, exactly one non-numeric token differs by one
  insertion/deletion/substitution, all surrounding tokens are identical, and short/numeric token
  changes have already been excluded; treat that as strong typo-shaped evidence, but still defer
  if the differing token could encode a real scientific distinction;
- semantically different stages, grades, visits, timepoints, doses, outcomes, or specimen types
  must remain distinct even when their labels look similar;
- contextual missingness labels such as Unknown, Not available, or Not applicable are meaningful
  states unless an explicit study schema says otherwise; they should not be collapsed to null;
- safe formatting operations such as trimming surrounding whitespace may be accepted when the
  proposed change is exactly bounded;
- numeric/date proposals preserve unparseable values, so judge whether the field itself is
  appropriately numeric/date-like rather than trying to repair the bad value;
- if uncertain, defer.

Return exactly one selection for each candidate ID.
"""

EVIDENCE_CALIBRATED_ADDENDUM = """
Use evidence-proportional judgment when reviewing a bounded category candidate.
A merely imaginable difference is not evidence of a scientific distinction.
For a commonplace full-word misspelling with a single nonnumeric character edit,
unchanged surrounding words, and a more frequent correctly spelled form, accept
when there is no concrete competing code, definition, or semantic distinction.
Do not invent historical spellings, abbreviations, procedures, or coding meanings
to explain a deferral; if one is relevant, identify evidence present in the prompt.
Preserve distinct numeric IDs, units, doses, stages, timepoints, analytes, marker
names, treatments, outcome categories, and contextual missingness labels.
These protections take priority over high lexical similarity or label frequency.
If the supplied study dictionary explicitly establishes that two codes differ,
defer even when a numeric-zero-padding heuristic suggests equivalence.
If it explicitly establishes they are equivalent, that specific evidence
outweighs unsupported speculation about different meanings.
Never infer that two assay labels are equivalent from spelling similarity alone.
Return the supplied candidate ID with accept or defer; never invent a mapping.
"""

# Production policy validated synthetically in v0.7.8:
# two independent calibrated reviews, with a third identical review only when
# the first two disagree. Any malformed/model-error response fails closed.
SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + "\n" + EVIDENCE_CALIBRATED_ADDENDUM


@dataclass(slots=True)
class AIPlanResult:
    plan: CleaningPlan
    context: AIPlannerContext
    response: AIPlannerResponse
    rejected: list[AIRejectedProposal]
    candidates: list[AICandidate]
    model: str
    provider: str
    review_call_count: int = 0
    consensus_disagreement_count: int = 0


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


def _selection_map(
    response: AIPlannerResponse | None,
    candidate_by_id: dict[str, AICandidate],
    *,
    review_number: int,
) -> tuple[dict[str, AICandidateSelection], list[AIRejectedProposal]]:
    """Return one valid selection per candidate; malformed entries fail closed."""
    if response is None:
        return {}, []

    grouped: dict[str, list[AICandidateSelection]] = {}
    rejected: list[AIRejectedProposal] = []
    for selection in response.selections:
        candidate = candidate_by_id.get(selection.candidate_id)
        if candidate is None:
            rejected.append(
                AIRejectedProposal(
                    candidate_id=selection.candidate_id,
                    rationale=selection.rationale,
                    reason=(
                        f"Review {review_number} referenced an unknown candidate ID."
                    ),
                )
            )
            continue
        grouped.setdefault(selection.candidate_id, []).append(selection)

    valid: dict[str, AICandidateSelection] = {}
    for candidate_id, selections in grouped.items():
        candidate = candidate_by_id[candidate_id]
        if len(selections) != 1:
            rejected.append(
                AIRejectedProposal(
                    candidate_id=candidate_id,
                    type=candidate.action.type,
                    table=candidate.action.table,
                    column=candidate.action.column,
                    parameters=candidate.action.parameters,
                    rationale="; ".join(item.rationale for item in selections),
                    reason=(
                        f"Review {review_number} returned the same candidate ID "
                        "more than once; this vote was discarded."
                    ),
                )
            )
            continue
        valid[candidate_id] = selections[0]
    return valid, rejected


def _call_review(
    provider: StructuredAIProvider,
    *,
    model: str,
    user_prompt: str,
) -> tuple[AIPlannerResponse | None, str | None]:
    try:
        return (
            provider.generate_structured(
                model=model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=AIPlannerResponse,
            ),
            None,
        )
    except AIProviderError as exc:
        return None, str(exc)


def _consensus_selection(
    candidate: AICandidate,
    first: AICandidateSelection | None,
    second: AICandidateSelection | None,
    third: AICandidateSelection | None,
    *,
    first_error: str | None,
    second_error: str | None,
    third_error: str | None,
) -> AICandidateSelection:
    """Conservatively aggregate calibrated semantic reviews for one candidate."""
    if first_error or second_error or first is None or second is None:
        details = "; ".join(
            item
            for item in [
                first_error,
                second_error,
                "missing or malformed candidate selection"
                if first is None or second is None
                else None,
            ]
            if item
        )
        return AICandidateSelection(
            candidate_id=candidate.id,
            decision=CandidateDecision.DEFER,
            rationale=(
                "Adaptive consensus failed closed to human review"
                + (f": {details}" if details else ".")
            ),
            confidence=0.0,
        )

    if first.decision == second.decision:
        confidence = min(first.confidence, second.confidence)
        rationale = (
            f"Adaptive consensus 2/2 {first.decision.value}: "
            f"{first.rationale.strip()}"
        )
        return AICandidateSelection(
            candidate_id=candidate.id,
            decision=first.decision,
            rationale=rationale,
            confidence=confidence,
        )

    if third_error or third is None:
        details = third_error or "third review missing or malformed"
        return AICandidateSelection(
            candidate_id=candidate.id,
            decision=CandidateDecision.DEFER,
            rationale=(
                "Initial calibrated reviews disagreed and the tie-break review "
                f"failed; deferred for human review: {details}"
            ),
            confidence=0.0,
        )

    votes = [first, second, third]
    accept_votes = [
        item for item in votes if item.decision == CandidateDecision.ACCEPT
    ]
    defer_votes = [
        item for item in votes if item.decision == CandidateDecision.DEFER
    ]
    majority = accept_votes if len(accept_votes) > len(defer_votes) else defer_votes
    decision = majority[0].decision
    confidence = min(item.confidence for item in majority)
    rationale = (
        f"Adaptive consensus 2/3 {decision.value}: "
        f"{majority[0].rationale.strip()}"
    )
    return AICandidateSelection(
        candidate_id=candidate.id,
        decision=decision,
        rationale=rationale,
        confidence=confidence,
    )


def _adaptive_consensus_review(
    provider: StructuredAIProvider,
    *,
    model: str,
    context: AIPlannerContext,
    candidates: list[AICandidate],
) -> tuple[AIPlannerResponse, list[AIRejectedProposal], int, int]:
    """Run two calibrated reviews, using one tie-break call only on disagreement."""
    if not candidates:
        return (
            AIPlannerResponse(
                summary="No AI-review candidates were generated by the deterministic engine.",
                selections=[],
            ),
            [],
            0,
            0,
        )

    prompt = _build_user_prompt(context)
    candidate_by_id = {candidate.id: candidate for candidate in candidates}

    first_response, first_error = _call_review(
        provider, model=model, user_prompt=prompt
    )
    second_response, second_error = _call_review(
        provider, model=model, user_prompt=prompt
    )
    first_map, first_rejected = _selection_map(
        first_response, candidate_by_id, review_number=1
    )
    second_map, second_rejected = _selection_map(
        second_response, candidate_by_id, review_number=2
    )
    rejected = [*first_rejected, *second_rejected]
    review_call_count = 2

    disagreements = [
        candidate.id
        for candidate in candidates
        if first_map.get(candidate.id) is not None
        and second_map.get(candidate.id) is not None
        and first_map[candidate.id].decision != second_map[candidate.id].decision
    ]

    third_map: dict[str, AICandidateSelection] = {}
    third_error: str | None = None
    if disagreements and not first_error and not second_error:
        third_response, third_error = _call_review(
            provider, model=model, user_prompt=prompt
        )
        third_map, third_rejected = _selection_map(
            third_response, candidate_by_id, review_number=3
        )
        rejected.extend(third_rejected)
        review_call_count += 1

    final = [
        _consensus_selection(
            candidate,
            first_map.get(candidate.id),
            second_map.get(candidate.id),
            third_map.get(candidate.id),
            first_error=first_error,
            second_error=second_error,
            third_error=third_error if candidate.id in disagreements else None,
        )
        for candidate in candidates
    ]
    accepted = sum(
        selection.decision == CandidateDecision.ACCEPT for selection in final
    )
    deferred = len(final) - accepted
    error_note = ""
    if first_error or second_error or third_error:
        error_note = " One or more model calls failed and affected candidates were deferred."
    response = AIPlannerResponse(
        summary=(
            f"Adaptive calibrated consensus reviewed {len(final)} candidate(s): "
            f"{accepted} accepted, {deferred} deferred; "
            f"{review_call_count} model call(s), {len(disagreements)} disagreement(s)."
            + error_note
        ),
        selections=final,
    )
    return response, rejected, review_call_count, len(disagreements)


def propose_ai_plan(
    path: str | Path,
    *,
    provider: StructuredAIProvider,
    model: str,
    include_rules: bool = True,
    schema: StudySchema | None = None,
) -> AIPlanResult:
    resolved = Path(path).expanduser().resolve()
    loaded = load_dataset(resolved)
    report = profile_dataset(resolved, schema=schema)
    candidates = build_ai_candidates(report, loaded.tables, schema=schema)
    context = build_ai_context(report, candidates)

    response, rejected, review_call_count, disagreement_count = (
        _adaptive_consensus_review(
            provider,
            model=model,
            context=context,
            candidates=candidates,
        )
    )

    candidate_by_id = {candidate.id: candidate for candidate in candidates}
    seen: set[str] = set()
    accepted: list[CleaningAction] = []
    deferred: list[DeferredReview] = []

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
            deferred.append(
                DeferredReview(
                    candidate_id=candidate.id,
                    proposed_action=candidate.action.model_copy(deep=True),
                    question=candidate.context.question,
                    rationale=selection.rationale,
                    confidence=selection.confidence,
                    evidence=candidate.context.evidence,
                )
            )
            continue

        action = candidate.action.model_copy(deep=True)
        action.description = selection.rationale.strip() or candidate.context.question
        action.confidence = min(action.confidence, selection.confidence)
        action.origin = ActionOrigin.AI
        accepted.append(action)

    # Missing selections are also explicit, non-executable human review items.
    for candidate in candidates:
        if candidate.id not in seen:
            deferred.append(
                DeferredReview(
                    candidate_id=candidate.id,
                    proposed_action=candidate.action.model_copy(deep=True),
                    question=candidate.context.question,
                    rationale="Model did not return a decision; no action was proposed.",
                    confidence=0.0,
                    evidence=candidate.context.evidence,
                )
            )

    # Planning notes must survive hybrid and AI-only modes: omitted
    # structural-only fields remain visible as non-executable coverage QA.
    rule_plan = propose_actions(resolved, schema=schema)
    actions: list[CleaningAction] = []
    if include_rules:
        actions.extend(rule_plan.actions)

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
            deferred_reviews=deferred,
            planning_notes=rule_plan.planning_notes,
            study_schema_sha256=schema.source_sha256 if schema else None,
            planner="hybrid_ai" if include_rules else "ai_only",
            model=model,
        ),
        context=context,
        response=response,
        rejected=rejected,
        candidates=candidates,
        model=model,
        provider=provider.name,
        review_call_count=review_call_count,
        consensus_disagreement_count=disagreement_count,
    )
