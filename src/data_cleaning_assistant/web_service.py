from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from .actions import (
    ActionDecision,
    CleaningAction,
    CleaningPlan,
    ReviewSession,
)
from .guided import (
    GuidedSession,
    GuidedStage,
    create_guided_session,
    finalize_guided_session,
    load_guided_session,
    prepare_guided_plan,
    record_confirmed_schema,
    record_reviewed_plan,
    save_guided_session,
)
from .io import load_dataset
from .mapping_safety import approval_mapping_conflicts, atomize_category_mappings
from .models import DatasetReport
from .planner import file_sha256
from .presentation import (
    HumanExplanation,
    SheetPreview,
    action_context_payload,
    action_explanation,
    action_sheet_preview,
    issue_sheet_preview,
    family_id,
    family_presentations,
    readable_name,
    schema_explanation,
    schema_sheet_previews,
)
from .review import action_target, group_actions, record_review_event
from .schema_draft import (
    SchemaSuggestionDecision,
    load_schema_draft,
    materialize_study_schema,
    write_schema_draft,
    write_study_schema,
)
from .semantic_narrator import ActionNarrator, SemanticNarration
from .transforms import preview_action


class WebSchemaSuggestion(BaseModel):
    id: str
    kind: str
    table: str
    column: str | None
    confidence: float
    decision: str
    explanation: HumanExplanation
    previews: list[SheetPreview] = Field(default_factory=list)
    technical_details: dict[str, Any] = Field(default_factory=dict)


class WebActionPreview(BaseModel):
    id: str
    type: str
    table: str
    column: str | None
    confidence: float
    decision: str
    affected_count: int
    explanation: HumanExplanation
    preview: SheetPreview
    technical_details: dict[str, Any] = Field(default_factory=dict)


class WebFamilyMember(BaseModel):
    action_id: str
    table: str
    column: str | None
    target_label: str
    affected_count: int
    decision: str
    explanation: HumanExplanation
    preview: SheetPreview


class WebActionFamily(BaseModel):
    id: str
    action_type: str
    member_count: int
    affected_count: int
    decision: str
    explanation: HumanExplanation
    members: list[WebFamilyMember] = Field(default_factory=list)


class WebQualityWarning(BaseModel):
    id: str
    severity: str
    table: str
    column: str | None
    title: str
    description: str
    affected_count: int
    suggested_action: str | None = None
    preview: SheetPreview | None = None


class WebSessionState(BaseModel):
    id: str
    stage: str
    source_name: str
    workspace: str
    schema_suggestions: int = 0
    schema_unresolved: int = 0
    actions: int = 0
    action_unresolved: int = 0
    approved_actions: int = 0
    executed_actions: int = 0
    cleaned_ready: bool = False
    verification_passed: bool | None = None
    review_completed: bool = False
    narrator_enabled: bool = False
    warning_count: int = 0


def _safe_upload_name(name: str) -> str:
    cleaned = Path(name).name.strip()
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError("Uploaded file must have a valid filename.")
    suffix = Path(cleaned).suffix.casefold()
    if suffix not in {".csv", ".xlsx", ".xlsm", ".parquet"}:
        raise ValueError("Only CSV, XLSX, XLSM, and Parquet datasets are supported.")
    return cleaned


class LocalWebStore:
    """Filesystem-backed adapter from the web UI to the guided-session engine."""

    def __init__(
        self,
        root: str | Path,
        *,
        narrator: ActionNarrator | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.narrator = narrator

    def new_session_paths(
        self,
        filename: str,
    ) -> tuple[str, Path, Path]:
        safe_name = _safe_upload_name(filename)
        session_id = uuid4().hex[:12]
        session_root = self.root / session_id
        source_dir = session_root / "source"
        workspace = session_root / "run"
        source_dir.mkdir(parents=True)
        return session_id, source_dir / safe_name, workspace

    def workspace(self, session_id: str) -> Path:
        if not session_id or any(char not in "0123456789abcdef" for char in session_id):
            raise ValueError("Invalid session identifier.")
        workspace = (self.root / session_id / "run").resolve()
        if self.root not in workspace.parents:
            raise ValueError("Invalid session identifier.")
        return workspace

    def load(self, session_id: str) -> GuidedSession:
        return load_guided_session(self.workspace(session_id))

    def _report(self, session: GuidedSession) -> DatasetReport:
        return DatasetReport.model_validate_json(
            Path(session.profile_path).read_text(encoding="utf-8")
        )

    def _plan(self, session: GuidedSession) -> CleaningPlan:
        plan_path = session.reviewed_plan_path or session.plan_path
        if plan_path is None:
            raise ValueError("Cleaning plan is missing.")
        return CleaningPlan.model_validate_json(
            Path(plan_path).read_text(encoding="utf-8")
        )

    def start_from_path(
        self,
        source_path: str | Path,
        workspace: str | Path,
    ) -> GuidedSession:
        session = create_guided_session(source_path, workspace)
        return self._prepare_if_ready(session)

    def _prepare_if_ready(self, session: GuidedSession) -> GuidedSession:
        if session.stage == GuidedStage.PLANNING:
            plan = prepare_guided_plan(session)
            if plan.actions:
                atomize_category_mappings(plan)
                Path(session.plan_path).write_text(
                    plan.model_dump_json(indent=2),
                    encoding="utf-8",
                )
        return session

    def _assert_review_source(self, session: GuidedSession) -> None:
        if file_sha256(session.source_path) != session.source_sha256:
            raise ValueError(
                "The source dataset changed after this session started. "
                "Start a new session before continuing."
            )

    def state(self, session_id: str) -> WebSessionState:
        session = self.load(session_id)
        draft = load_schema_draft(
            session.reviewed_schema_draft_path or session.schema_draft_path
        )
        schema_unresolved = sum(
            item.decision == SchemaSuggestionDecision.PROPOSED
            for item in draft.suggestions
        )

        actions = 0
        unresolved = 0
        approved = session.approved_action_count
        review_completed = False
        if session.reviewed_plan_path or session.plan_path:
            plan = self._plan(session)
            actions = len(plan.actions)
            unresolved = sum(
                item.decision == ActionDecision.PROPOSED
                for item in plan.actions
            )
            approved = sum(
                item.decision == ActionDecision.APPROVED
                for item in plan.actions
            )
            review_completed = (
                plan.review_session is not None
                and plan.review_session.outcome == "completed"
            )

        verification_passed: bool | None = None
        if session.verification_path and Path(session.verification_path).exists():
            payload = json.loads(
                Path(session.verification_path).read_text(encoding="utf-8")
            )
            verification_passed = bool(payload.get("passed"))

        report = self._report(session)
        represented_issue_ids: set[str] = set()
        if session.reviewed_plan_path or session.plan_path:
            for action in self._plan(session).actions:
                represented_issue_ids.update(action.issue_ids)
        warning_count = sum(
            issue.severity.value != "info"
            and issue.id not in represented_issue_ids
            for issue in report.issues
        )

        return WebSessionState(
            id=session_id,
            stage=session.stage.value,
            source_name=Path(session.source_path).name,
            workspace=session.workspace,
            schema_suggestions=len(draft.suggestions),
            schema_unresolved=schema_unresolved,
            actions=actions,
            action_unresolved=unresolved,
            approved_actions=approved,
            executed_actions=session.executed_action_count,
            cleaned_ready=Path(session.cleaned_path).exists(),
            verification_passed=verification_passed,
            review_completed=review_completed,
            narrator_enabled=self.narrator is not None,
            warning_count=warning_count,
        )

    def quality_warnings(self, session_id: str) -> list[WebQualityWarning]:
        session = self.load(session_id)
        report = self._report(session)
        tables = load_dataset(session.source_path).tables

        represented_issue_ids: set[str] = set()
        if session.reviewed_plan_path or session.plan_path:
            plan = self._plan(session)
            for action in plan.actions:
                represented_issue_ids.update(action.issue_ids)

        warnings: list[WebQualityWarning] = []
        for issue in report.issues:
            if issue.severity.value == "info":
                continue
            if issue.id in represented_issue_ids:
                continue
            warnings.append(
                WebQualityWarning(
                    id=issue.id,
                    severity=issue.severity.value,
                    table=issue.table,
                    column=issue.column,
                    title=issue.title,
                    description=issue.description,
                    affected_count=issue.affected_count,
                    suggested_action=issue.suggested_action,
                    preview=issue_sheet_preview(tables, issue),
                )
            )
        warnings.sort(
            key=lambda item: (
                {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(
                    item.severity,
                    4,
                ),
                -item.affected_count,
                item.title,
            )
        )
        return warnings

    def schema_suggestions(self, session_id: str) -> list[WebSchemaSuggestion]:
        session = self.load(session_id)
        draft = load_schema_draft(
            session.reviewed_schema_draft_path or session.schema_draft_path
        )
        tables = load_dataset(session.source_path).tables
        report = self._report(session)

        return [
            WebSchemaSuggestion(
                id=item.id,
                kind=item.kind.value,
                table=item.table,
                column=item.column,
                confidence=item.confidence,
                decision=item.decision.value,
                explanation=schema_explanation(item, report=report),
                previews=schema_sheet_previews(tables, item, sample_limit=5),
                technical_details={
                    "proposed": item.proposed,
                    "evidence": item.evidence,
                },
            )
            for item in draft.suggestions
        ]

    def decide_schema(
        self,
        session_id: str,
        suggestion_id: str,
        decision: SchemaSuggestionDecision,
    ) -> WebSessionState:
        session = self.load(session_id)
        if session.stage != GuidedStage.SCHEMA_REVIEW:
            raise ValueError("This session is not in schema review.")
        self._assert_review_source(session)

        path = Path(
            session.reviewed_schema_draft_path or session.schema_draft_path
        )
        draft = load_schema_draft(path)
        matches = [item for item in draft.suggestions if item.id == suggestion_id]
        if len(matches) != 1:
            raise ValueError(f"Unknown schema suggestion: {suggestion_id}")
        matches[0].decision = decision

        reviewed_path = Path(session.workspace) / "schema_draft.reviewed.yaml"
        write_schema_draft(draft, reviewed_path)
        session.reviewed_schema_draft_path = str(reviewed_path)
        save_guided_session(session)
        return self.state(session_id)

    def complete_schema(self, session_id: str) -> WebSessionState:
        session = self.load(session_id)
        if session.stage != GuidedStage.SCHEMA_REVIEW:
            raise ValueError("This session is not in schema review.")
        self._assert_review_source(session)

        reviewed_path = Path(
            session.reviewed_schema_draft_path or session.schema_draft_path
        )
        draft = load_schema_draft(reviewed_path)
        unresolved = [
            item.id
            for item in draft.suggestions
            if item.decision == SchemaSuggestionDecision.PROPOSED
        ]
        if unresolved:
            raise ValueError(
                f"{len(unresolved)} schema suggestion(s) still need a decision."
            )

        schema = materialize_study_schema(
            draft,
            dataset_path=session.source_path,
        )
        schema_path = Path(session.workspace) / "study_schema.yaml"
        write_study_schema(schema, schema_path)
        record_confirmed_schema(
            session,
            schema_path=schema_path,
            reviewed_draft_path=reviewed_path,
        )
        self._prepare_if_ready(session)
        return self.state(session_id)

    def actions(self, session_id: str) -> list[WebActionPreview]:
        session = self.load(session_id)
        if not (session.reviewed_plan_path or session.plan_path):
            return []
        plan = self._plan(session)
        tables = load_dataset(session.source_path).tables
        report = self._report(session)

        output: list[WebActionPreview] = []
        for action in plan.actions:
            preview = preview_action(tables, action, sample_limit=5)
            output.append(
                WebActionPreview(
                    id=action.id,
                    type=action.type.value,
                    table=action.table,
                    column=action.column,
                    confidence=action.confidence,
                    decision=action.decision.value,
                    affected_count=preview.affected_count,
                    explanation=action_explanation(
                        action,
                        affected_count=preview.affected_count,
                        report=report,
                    ),
                    preview=action_sheet_preview(
                        tables,
                        action,
                        sample_limit=5,
                    ),
                    technical_details={
                        "action_type": action.type.value,
                        "parameters": action.parameters,
                        "origin": action.origin.value,
                    },
                )
            )
        return output

    def action_families(self, session_id: str) -> list[WebActionFamily]:
        session = self.load(session_id)
        if not (session.reviewed_plan_path or session.plan_path):
            return []
        plan = self._plan(session)
        tables = load_dataset(session.source_path).tables
        report = self._report(session)

        grouped = group_actions(tables, plan)
        presented = {
            item.id: item
            for item in family_presentations(
                tables,
                plan,
                report=report,
            )
        }

        output: list[WebActionFamily] = []
        for family in grouped:
            family_key = family_id(family)
            presentation = presented[family_key]
            members: list[WebFamilyMember] = []
            for member in family.members:
                action = member.action
                members.append(
                    WebFamilyMember(
                        action_id=action.id,
                        table=action.table,
                        column=action.column,
                        target_label=readable_name(action.column or action.table),
                        affected_count=member.affected_count,
                        decision=action.decision.value,
                        explanation=action_explanation(
                            action,
                            affected_count=member.affected_count,
                            report=report,
                        ),
                        preview=action_sheet_preview(
                            tables,
                            action,
                            sample_limit=4,
                        ),
                    )
                )
            output.append(
                WebActionFamily(
                    id=family_key,
                    action_type=family.action_type,
                    member_count=len(family.members),
                    affected_count=family.affected_count,
                    decision=presentation.decision,
                    explanation=presentation.explanation,
                    members=members,
                )
            )
        return output

    def _save_reviewed_plan(
        self,
        session: GuidedSession,
        plan: CleaningPlan,
    ) -> Path:
        reviewed_path = Path(session.workspace) / "reviewed_plan.json"
        reviewed_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        record_reviewed_plan(session, reviewed_path)
        return reviewed_path

    def decide_action(
        self,
        session_id: str,
        action_id: str,
        decision: ActionDecision,
    ) -> WebSessionState:
        session = self.load(session_id)
        if session.stage != GuidedStage.CLEANING_REVIEW:
            raise ValueError("This session is not in cleaning review.")
        self._assert_review_source(session)

        plan = self._plan(session)
        matches = [item for item in plan.actions if item.id == action_id]
        if len(matches) != 1:
            raise ValueError(f"Unknown cleaning action: {action_id}")
        action: CleaningAction = matches[0]

        if decision == ActionDecision.APPROVED:
            conflicts = approval_mapping_conflicts(plan, [action])
            if conflicts:
                targets = ", ".join(
                    f"{item.source!r} -> {list(item.targets)!r}"
                    for item in conflicts
                )
                raise ValueError(
                    "Approval conflicts with another canonical mapping: "
                    + targets
                )

        action.decision = decision
        tables = load_dataset(session.source_path).tables
        preview = preview_action(tables, action, sample_limit=1)

        if plan.review_session is None:
            plan.review_session = ReviewSession(started_at=datetime.now(UTC))
        elif plan.review_session.outcome == "completed":
            raise ValueError("Cleaning review is already completed.")
        record_review_event(
            plan,
            scope="web_action",
            actions=[action],
            affected_count=preview.affected_count,
            decision=decision.value,
            duration_seconds=0.0,
            targets_preview=[action_target(action)],
        )

        self._save_reviewed_plan(session, plan)
        return self.state(session_id)

    def decide_family(
        self,
        session_id: str,
        requested_family_id: str,
        decision: ActionDecision,
    ) -> WebSessionState:
        session = self.load(session_id)
        if session.stage != GuidedStage.CLEANING_REVIEW:
            raise ValueError("This session is not in cleaning review.")
        self._assert_review_source(session)

        plan = self._plan(session)
        tables = load_dataset(session.source_path).tables
        matches = [
            family
            for family in group_actions(tables, plan)
            if family_id(family) == requested_family_id
        ]
        if len(matches) != 1:
            raise ValueError(f"Unknown cleaning family: {requested_family_id}")
        family = matches[0]
        actions = [member.action for member in family.members]

        if decision == ActionDecision.APPROVED:
            conflicts = approval_mapping_conflicts(plan, actions)
            if conflicts:
                targets = ", ".join(
                    f"{item.source!r} -> {list(item.targets)!r}"
                    for item in conflicts
                )
                raise ValueError(
                    "Bulk approval conflicts with another canonical mapping: "
                    + targets
                )

        if plan.review_session is None:
            plan.review_session = ReviewSession(started_at=datetime.now(UTC))
        elif plan.review_session.outcome == "completed":
            raise ValueError("Cleaning review is already completed.")

        for action in actions:
            action.decision = decision

        record_review_event(
            plan,
            scope="web_family",
            actions=actions,
            affected_count=family.affected_count,
            decision=decision.value,
            duration_seconds=0.0,
            bulk_confirmed=True,
            targets_preview=[action_target(action) for action in actions],
        )
        self._save_reviewed_plan(session, plan)
        return self.state(session_id)

    def _action_by_id(
        self,
        session: GuidedSession,
        action_id: str,
    ) -> CleaningAction:
        plan = self._plan(session)
        matches = [action for action in plan.actions if action.id == action_id]
        if len(matches) != 1:
            raise ValueError(f"Unknown cleaning action: {action_id}")
        return matches[0]

    def narrate_action(
        self,
        session_id: str,
        action_id: str,
    ) -> SemanticNarration:
        if self.narrator is None:
            raise ValueError(
                "Local semantic narration is not enabled for this web session."
            )

        session = self.load(session_id)
        action = self._action_by_id(session, action_id)
        tables = load_dataset(session.source_path).tables
        report = self._report(session)
        preview = preview_action(tables, action, sample_limit=5)
        explanation = action_explanation(
            action,
            affected_count=preview.affected_count,
            report=report,
        )
        payload = action_context_payload(
            dataset_name=Path(session.source_path).name,
            tables=tables,
            action=action,
            explanation=explanation,
            report=report,
        )

        cache_path = Path(session.workspace) / "semantic_narrations.json"
        cache: dict[str, Any] = {}
        if cache_path.exists():
            try:
                raw = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    cache = raw
            except json.JSONDecodeError:
                cache = {}

        cache_key = f"{self.narrator.model}:{action.id}"
        cached = cache.get(cache_key)
        if isinstance(cached, dict):
            return SemanticNarration.model_validate(cached)

        narration = self.narrator.narrate(payload)
        cache[cache_key] = narration.model_dump(mode="json")
        cache_path.write_text(
            json.dumps(cache, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return narration

    def complete_review(self, session_id: str) -> WebSessionState:
        session = self.load(session_id)
        if session.stage != GuidedStage.CLEANING_REVIEW:
            raise ValueError("This session is not in cleaning review.")
        self._assert_review_source(session)

        plan = self._plan(session)
        unresolved = [
            action.id
            for action in plan.actions
            if action.decision == ActionDecision.PROPOSED
        ]
        if unresolved:
            raise ValueError(
                f"{len(unresolved)} cleaning action(s) still need a decision."
            )

        now = datetime.now(UTC)
        if plan.review_session is None:
            plan.review_session = ReviewSession(started_at=now)
        plan.review_session.completed_at = now
        plan.review_session.outcome = "completed"
        plan.review_session.duration_seconds = max(
            0.0,
            (now - plan.review_session.started_at).total_seconds(),
        )

        self._save_reviewed_plan(session, plan)
        return self.state(session_id)

    def apply(self, session_id: str) -> WebSessionState:
        session = self.load(session_id)
        if session.stage != GuidedStage.CLEANING_REVIEW:
            raise ValueError("This session is not ready to apply.")
        verification = finalize_guided_session(session)
        if verification is not None and not verification.passed:
            raise ValueError("Cleaning completed but end-to-end verification failed.")
        return self.state(session_id)

    def cleaned_path(self, session_id: str) -> Path:
        session = self.load(session_id)
        path = Path(session.cleaned_path)
        if not path.exists():
            raise FileNotFoundError("Cleaned output is not available yet.")
        return path
