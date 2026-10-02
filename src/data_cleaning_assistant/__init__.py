"""Local-first data profiling and cleaning infrastructure."""

from .actions import ChangeSet, CleaningAction, CleaningPlan, DeferredReview
from .ai_context import build_ai_context
from .ai_planner import propose_ai_plan
from .ai_provider import OllamaProvider
from .benchmark import benchmark_cases, research_benchmark_cases, run_ai_benchmark
from .engine import profile_dataset
from .evaluation import OfflineEvaluationReport, run_offline_evaluation
from .models import DatasetReport, Issue
from .planner import propose_actions
from .recipe import CleaningRecipe, load_recipe, recipe_to_plan
from .schema_draft import (
    SchemaDraft,
    SchemaSuggestion,
    build_schema_draft,
    materialize_study_schema,
)
from .study_schema import StudySchema, load_study_schema
from .verification import RunVerificationReport, verify_run

__all__ = [
    "ChangeSet",
    "CleaningAction",
    "CleaningPlan",
    "CleaningRecipe",
    "DatasetReport",
    "DeferredReview",
    "Issue",
    "OfflineEvaluationReport",
    "OllamaProvider",
    "RunVerificationReport",
    "SchemaDraft",
    "SchemaSuggestion",
    "StudySchema",
    "benchmark_cases",
    "build_ai_context",
    "build_schema_draft",
    "load_recipe",
    "load_study_schema",
    "materialize_study_schema",
    "profile_dataset",
    "propose_actions",
    "propose_ai_plan",
    "recipe_to_plan",
    "research_benchmark_cases",
    "run_ai_benchmark",
    "run_offline_evaluation",
    "verify_run",
]
__version__ = "0.12.0"
