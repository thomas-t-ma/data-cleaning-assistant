from __future__ import annotations

from pathlib import Path

from .detectors import detect_basic_issues, detect_fuzzy_categories
from .io import load_dataset
from .models import DatasetReport
from .profiling import profile_table
from .structure import (\n    infer_candidate_keys,\n    infer_grain,\n    infer_relationships,\n    repeated_entity_issues,\n)


def profile_dataset(path: str | Path) -> DatasetReport:
    loaded = load_dataset(path)
    table_profiles = []
    issues = []

    for table_name, frame in loaded.tables.items():
        profile = profile_table(table_name, frame)
        profile.candidate_keys = infer_candidate_keys(frame)
        profile.grain = infer_grain(table_name, frame)
        table_profiles.append(profile)

        issues.extend(detect_basic_issues(table_name, frame))
        issues.extend(detect_fuzzy_categories(table_name, frame))
        issues.extend(repeated_entity_issues(table_name, frame))

    return DatasetReport(
        source_path=str(loaded.path),
        source_type=loaded.source_type,
        tables=table_profiles,
        issues=issues,
        relationships=infer_relationships(loaded.tables),
    )
