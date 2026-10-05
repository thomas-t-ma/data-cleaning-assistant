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
from .planner import file_sha256
from .review import action_target, record_review_event
from .schema_draft import (
    SchemaSuggestionDecision,
    load_schema_draft,
    materialize_study_schema,
    write_schema_draft,
    write_study_schema,
)
from .transforms import preview_action


class WebSchemaSuggestion(BaseModel):
    id: str
    kind: str
    table: str
    column: str | None
    confidence: float
    proposed: dict[str, Any]
    rationale: str
    decision: str


class WebActionPreview(BaseModel):
    id: str
    type: str
    table: str
    column: str | None
    title: str
    description: str
    parameters: dict[str, Any]
    confidence: float
    decision: str
    affected_count: int
    sample_changes: list[dict[str, Any]] = Field(default_factory=list)
    sample_rows: list[dict[str, Any]] = Field(default_factory=list)


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


def _safe_upload_name(name: str) -> str:
    cleaned = Path(name).name.strip()
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError("Uploaded file must have a valid filename.")
    suffix = Path(cleaned).suffix.casefold()
    if suffix not in {".csv", ".xlsx", ".parquet"}:
        raise ValueError("Only CSV, XLSX, and Parquet datasets are supported.")
    return cleaned


class LocalWebStore:
    """Filesystem-backed adapter from the web UI to the guided-session engine."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

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
        if session.reviewed_plan_path or session.plan_path:
            plan_path = session.reviewed_plan_path or session.plan_path
            plan = CleaningPlan.model_validate_json(
                Path(plan_path).read_text(encoding="utf-8")
            )
            actions = len(plan.actions)
            unresolved = sum(
                item.decision == ActionDecision.PROPOSED
                for item in plan.actions
            )
            approved = sum(
                item.decision == ActionDecision.APPROVED
                for item in plan.actions
            )

        verification_passed: bool | None = None
        review_completed = False
        if session.reviewed_plan_path or session.plan_path:
            plan_path = session.reviewed_plan_path or session.plan_path
            review_plan = CleaningPlan.model_validate_json(
                Path(plan_path).read_text(encoding="utf-8")
            )
            review_completed = (
                review_plan.review_session is not None
                and review_plan.review_session.outcome == "completed"
            )

        if session.verification_path and Path(session.verification_path).exists():
            payload = json.loads(
                Path(session.verification_path).read_text(encoding="utf-8")
            )
            verification_passed = bool(payload.get("passed"))

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
        )

    def schema_suggestions(self, session_id: str) -> list[WebSchemaSuggestion]:
        session = self.load(session_id)
        draft = load_schema_draft(
            session.reviewed_schema_draft_path or session.schema_draft_path
        )
        return [
            WebSchemaSuggestion(
                id=item.id,
                kind=item.kind.value,
                table=item.table,
                column=item.column,
                confidence=item.confidence,
                proposed=item.proposed,
                rationale=item.rationale,
                decision=item.decision.value,
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
        plan_path = session.reviewed_plan_path or session.plan_path
        if plan_path is None:
            return []
        plan = CleaningPlan.model_validate_json(
            Path(plan_path).read_text(encoding="utf-8")
        )
        tables = load_dataset(session.source_path).tables

        output: list[WebActionPreview] = []
        for action in plan.actions:
            preview = preview_action(tables, action, sample_limit=4)
            output.append(
                WebActionPreview(
                    id=action.id,
                    type=action.type.value,
                    table=action.table,
                    column=action.column,
                    title=action.title,
                    description=action.description,
                    parameters=action.parameters,
                    confidence=action.confidence,
                    decision=action.decision.value,
                    affected_count=preview.affected_count,
                    sample_changes=[
                        item.model_dump(mode="json")
                        for item in preview.sample_changes
                    ],
                    sample_rows=[
                        item.model_dump(mode="json")
                        for item in preview.sample_rows
                    ],
                )
            )
        return output

    def decide_action(
        self,
        session_id: str,
        action_id: str,
        decision: ActionDecision,
    ) -> WebSessionState:
        session = self.load(session_id)
        if session.stage != GuidedStage.CLEANING_REVIEW:
            raise ValueError("This session is not in cleaning review.")
        if file_sha256(session.source_path) != session.source_sha256:
            raise ValueError(
                "The source dataset changed after this session started. "
                "Start a new session before continuing."
            )

        source_plan_path = session.reviewed_plan_path or session.plan_path
        if source_plan_path is None:
            raise ValueError("Cleaning plan is missing.")
        plan = CleaningPlan.model_validate_json(
            Path(source_plan_path).read_text(encoding="utf-8")
        )
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

        reviewed_path = Path(session.workspace) / "reviewed_plan.json"
        reviewed_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        record_reviewed_plan(session, reviewed_path)
        return self.state(session_id)

    def complete_review(self, session_id: str) -> WebSessionState:
        session = self.load(session_id)
        if session.stage != GuidedStage.CLEANING_REVIEW:
            raise ValueError("This session is not in cleaning review.")
        plan_path = session.reviewed_plan_path or session.plan_path
        if plan_path is None:
            raise ValueError("Cleaning plan is missing.")

        plan = CleaningPlan.model_validate_json(
            Path(plan_path).read_text(encoding="utf-8")
        )
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

        reviewed_path = Path(session.workspace) / "reviewed_plan.json"
        reviewed_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        record_reviewed_plan(session, reviewed_path)
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
