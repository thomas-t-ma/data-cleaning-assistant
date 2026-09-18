"""Local-first data profiling and cleaning infrastructure."""

from .engine import profile_dataset
from .models import DatasetReport, Issue

__all__ = ["DatasetReport", "Issue", "profile_dataset"]
__version__ = "0.2.0"
