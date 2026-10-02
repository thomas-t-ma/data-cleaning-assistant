from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from .actions import ActionDecision, CleaningPlan
from .engine import profile_dataset
from .io import load_dataset
from .planner import file_sha256, propose_actions
from .schema_draft import build_schema_draft, write_schema_draft, write_study_schema
from .study_schema import load_study_schema
from .transforms import apply_plan, write_cleaned_dataset, write_recipe
from .verification import RunVerificationReport, verify_run


class GuidedStage(str, Enum):
    SCHEMA_REVIEW = "schema_review"
    PLANNING = "planning"
    CLEANING_REVIEW = "cleaning_review"
    NO_CHANGES = "no_changes"
    NO_APPROVED_CHANGES = "no_approved_changes"
    NO_EXECUTABLE_CHANGES = "no_executable_changes"
    VERIFIED = "verified"
    VERIFICATION_FAILED = "verification_failed"


class GuidedSession(BaseModel):
    version: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_path: str
    source_sha256: str
    workspace: str
    stage: GuidedStage
    profile_path: str
    schema_draft_path: str
    reviewed_schema_draft_path: str | None = None
    schema_path: str | None = None
    plan_path: str | None = None
    reviewed_plan_path: str | None = None
    cleaned_path: str
    audit_path: str | None = None
    recipe_path: str | None = None
    verification_path: str | None = None
    approved_action_count: int = 0
    executed_action_count: int = 0


def _manifest_path(session: GuidedSession) -> Path:
    return Path(session.workspace) / "session.json"


def save_guided_session(session: GuidedSession) -> Path:
    session.updated_at = datetime.now(UTC)
    path = _manifest_path(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(session.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_guided_session(path: str | Path) -> GuidedSession:
    resolved = Path(path).expanduser().resolve()
    manifest = resolved / "session.json" if resolved.is_dir() else resolved
    if not manifest.exists():
        raise FileNotFoundError(f"Guided session manifest does not exist: {manifest}")
    return GuidedSession.model_validate_json(
        manifest.read_text(encoding="utf-8")
    )


def _check_source(session: GuidedSession) -> Path:
    source = Path(session.source_path)
    if not source.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {source}")
    current = file_sha256(source)
    if current != session.source_sha256:
        raise ValueError(
            "The source dataset changed after this guided session started. "
            "Start a new session before continuing."
        )
    return source


def create_guided_session(
    input_path: str | Path,
    workspace: str | Path,
    *,
    output_path: str | Path | None = None,
    schema_path: str | Path | None = None,
) -> GuidedSession:
    source = Path(input_path).expanduser().resolve()
    loaded = load_dataset(source)

    root = Path(workspace).expanduser().resolve()
    if root.exists():
        raise FileExistsError(
            f"Guided workspace already exists: {root}. "
            "Use guided-resume to continue an existing session."
        )

    cleaned = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else root / f"{source.stem}.cleaned{source.suffix}"
    )
    if cleaned == source:
        raise ValueError("Guided output may not overwrite the source dataset.")
    if cleaned.exists():
        raise FileExistsError(f"Guided output already exists: {cleaned}")

    supplied_schema = None
    if schema_path is not None:
        supplied_schema = load_study_schema(schema_path)
        supplied_schema.validate_dataset(loaded.tables)

    root.mkdir(parents=True)

    profile_path = root / "profile.json"
    profile = profile_dataset(source)
    profile_path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")

    draft_path = root / "schema_draft.yaml"
    draft = build_schema_draft(source)
    write_schema_draft(draft, draft_path)

    confirmed_schema: str | None = None
    stage = (
        GuidedStage.SCHEMA_REVIEW
        if draft.suggestions
        else GuidedStage.PLANNING
    )

    if supplied_schema is not None:
        confirmed_path = root / "study_schema.yaml"
        write_study_schema(supplied_schema, confirmed_path)
        confirmed_schema = str(confirmed_path)
        stage = GuidedStage.PLANNING

    session = GuidedSession(
        source_path=str(source),
        source_sha256=file_sha256(source),
        workspace=str(root),
        stage=stage,
        profile_path=str(profile_path),
        schema_draft_path=str(draft_path),
        schema_path=confirmed_schema,
        cleaned_path=str(cleaned),
    )
    save_guided_session(session)
    return session


def record_confirmed_schema(
    session: GuidedSession,
    *,
    schema_path: str | Path,
    reviewed_draft_path: str | Path,
) -> GuidedSession:
    source = _check_source(session)
    schema = load_study_schema(schema_path)
    schema.validate_dataset(load_dataset(source).tables)

    session.schema_path = str(Path(schema_path).expanduser().resolve())
    session.reviewed_schema_draft_path = str(
        Path(reviewed_draft_path).expanduser().resolve()
    )
    session.stage = GuidedStage.PLANNING
    save_guided_session(session)
    return session


def prepare_guided_plan(session: GuidedSession) -> CleaningPlan:
    source = _check_source(session)
    schema = (
        load_study_schema(session.schema_path)
        if session.schema_path is not None
        else None
    )
    plan = propose_actions(source, schema=schema)

    plan_path = Path(session.workspace) / "cleaning_plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    session.plan_path = str(plan_path)
    session.stage = (
        GuidedStage.NO_CHANGES
        if not plan.actions and not plan.deferred_reviews
        else GuidedStage.CLEANING_REVIEW
    )
    save_guided_session(session)
    return plan


def record_reviewed_plan(
    session: GuidedSession,
    reviewed_plan_path: str | Path,
) -> CleaningPlan:
    source = _check_source(session)
    path = Path(reviewed_plan_path).expanduser().resolve()
    plan = CleaningPlan.model_validate_json(path.read_text(encoding="utf-8"))
    if plan.source_sha256 != session.source_sha256:
        raise ValueError("Reviewed plan does not belong to this guided session.")
    if file_sha256(source) != plan.source_sha256:
        raise ValueError("The source dataset changed after the reviewed plan was created.")

    session.reviewed_plan_path = str(path)
    session.approved_action_count = sum(
        action.decision == ActionDecision.APPROVED
        for action in plan.actions
    )
    session.stage = GuidedStage.CLEANING_REVIEW
    save_guided_session(session)
    return plan


def finalize_guided_session(
    session: GuidedSession,
) -> RunVerificationReport | None:
    source = _check_source(session)
    if session.reviewed_plan_path is None:
        raise ValueError("No reviewed cleaning plan is recorded for this session.")

    plan = CleaningPlan.model_validate_json(
        Path(session.reviewed_plan_path).read_text(encoding="utf-8")
    )
    if plan.review_session is None or plan.review_session.outcome != "completed":
        raise ValueError(
            "Cleaning review is not complete. Resume the guided session before applying."
        )

    session.approved_action_count = sum(
        action.decision == ActionDecision.APPROVED
        for action in plan.actions
    )
    if session.approved_action_count == 0:
        session.stage = GuidedStage.NO_APPROVED_CHANGES
        save_guided_session(session)
        return None

    loaded = load_dataset(source)
    cleaned_tables, changeset = apply_plan(
        source,
        loaded.tables,
        plan,
    )
    if not changeset.executions:
        session.stage = GuidedStage.NO_EXECUTABLE_CHANGES
        save_guided_session(session)
        return None

    cleaned_path = write_cleaned_dataset(
        source,
        loaded.tables,
        cleaned_tables,
        session.cleaned_path,
    )
    changeset.result_path = str(cleaned_path)
    changeset.result_sha256 = file_sha256(cleaned_path)

    audit_path = cleaned_path.with_name(f"{cleaned_path.name}.audit.json")
    recipe_path = cleaned_path.with_name(f"{cleaned_path.name}.recipe.yaml")
    audit_path.write_text(changeset.model_dump_json(indent=2), encoding="utf-8")
    write_recipe(plan, changeset, recipe_path)

    verification = verify_run(
        source,
        cleaned_path,
        audit_path,
        recipe_path,
    )
    verification_path = Path(session.workspace) / "verification.json"
    verification_path.write_text(
        verification.model_dump_json(indent=2),
        encoding="utf-8",
    )

    session.audit_path = str(audit_path)
    session.recipe_path = str(recipe_path)
    session.verification_path = str(verification_path)
    session.executed_action_count = len(changeset.executions)
    session.stage = (
        GuidedStage.VERIFIED
        if verification.passed
        else GuidedStage.VERIFICATION_FAILED
    )
    save_guided_session(session)
    return verification
