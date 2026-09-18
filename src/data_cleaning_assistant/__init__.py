"""Local-first data profiling and cleaning infrastructure."""

from .actions import ChangeSet, CleaningAction, CleaningPlan
from .ai_context import build_ai_context
from .ai_planner import propose_ai_plan
from .ai_provider import OllamaProvider
from .engine import profile_dataset
from .models import DatasetReport, Issue
from .planner import propose_actions
from .recipe import CleaningRecipe, load_recipe, recipe_to_plan

__all__ = [
    "ChangeSet",
    "CleaningAction",
    "CleaningPlan",
    "CleaningRecipe",
    "DatasetReport",
    "Issue",
    "OllamaProvider",
    "build_ai_context",
    "load_recipe",
    "profile_dataset",
    "propose_actions",
    "propose_ai_plan",
    "recipe_to_plan",
]
__version__ = "0.4.0"
