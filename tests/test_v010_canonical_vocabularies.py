from __future__ import annotations

from pathlib import Path

import pytest

from data_cleaning_assistant.actions import ActionType
from data_cleaning_assistant.ai_candidates import build_ai_candidates
from data_cleaning_assistant.engine import profile_dataset
from data_cleaning_assistant.io import load_dataset
from data_cleaning_assistant.planner import propose_actions
from data_cleaning_assistant.study_schema import StudySchema


def _license_schema() -> StudySchema:
    return StudySchema.model_validate(
        {
            "tables": {
                "study": {
                    "columns": {
                        "license": {
                            "canonical_values": {
                                "GNU General Public License": {
                                    "aliases": [
                                        "GNU GENERAL PUBLIC LICENSE",
                                        "GNU General Public License (GPL)",
                                    ]
                                },
                                "GNU General Public License v2.0": {
                                    "aliases": []
                                },
                                "GNU General Public License v3": {
                                    "aliases": []
                                },
                            }
                        }
                    }
                }
            }
        }
    )


def test_canonical_vocabulary_compiles_aliases_and_allowed_values() -> None:
    schema = _license_schema()
    spec = schema.tables["study"].columns["license"]

    assert spec.effective_canonical_map() == {
        "GNU GENERAL PUBLIC LICENSE": "GNU General Public License",
        "GNU General Public License (GPL)": "GNU General Public License",
    }
    assert spec.effective_allowed_values() == {
        "GNU General Public License",
        "GNU General Public License v2.0",
        "GNU General Public License v3",
    }
    assert spec.has_category_dictionary() is True


def test_canonical_labels_cannot_be_aliases_of_other_canonical_values() -> None:
    with pytest.raises(ValueError, match="cannot be an alias"):
        StudySchema.model_validate(
            {
                "tables": {
                    "study": {
                        "columns": {
                            "license": {
                                "canonical_values": {
                                    "Generic": {
                                        "aliases": ["Version 2"]
                                    },
                                    "Version 2": {
                                        "aliases": []
                                    },
                                }
                            }
                        }
                    }
                }
            }
        )


def test_legacy_map_cannot_override_protected_canonical_value() -> None:
    with pytest.raises(ValueError, match="cannot be an alias"):
        StudySchema.model_validate(
            {
                "tables": {
                    "study": {
                        "columns": {
                            "license": {
                                "canonical_map": {
                                    "Version 2": "Generic"
                                },
                                "canonical_values": {
                                    "Generic": {"aliases": []},
                                    "Version 2": {"aliases": []},
                                },
                            }
                        }
                    }
                }
            }
        )


def test_alias_cannot_point_to_competing_canonical_values() -> None:
    with pytest.raises(ValueError, match="maps to multiple canonical values"):
        StudySchema.model_validate(
            {
                "tables": {
                    "study": {
                        "columns": {
                            "license": {
                                "canonical_values": {
                                    "Generic": {
                                        "aliases": ["GPL"]
                                    },
                                    "Versioned": {
                                        "aliases": ["GPL"]
                                    },
                                }
                            }
                        }
                    }
                }
            }
        )


def test_vocabulary_normalizes_aliases_but_preserves_distinct_versions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "study.csv"
    source.write_text(
        "id,license\n"
        "1,GNU General Public License\n"
        "2,GNU GENERAL PUBLIC LICENSE\n"
        "3,GNU General Public License (GPL)\n"
        "4,GNU General Public License v2.0\n"
        "5,GNU General Public License v3\n",
        encoding="utf-8",
    )
    schema = _license_schema()

    plan = propose_actions(source, schema=schema)

    actions = [
        action
        for action in plan.actions
        if action.type == ActionType.NORMALIZE_CATEGORIES
        and action.column == "license"
    ]
    assert len(actions) == 1
    assert actions[0].parameters == {
        "mapping": {
            "GNU GENERAL PUBLIC LICENSE": "GNU General Public License",
            "GNU General Public License (GPL)": "GNU General Public License",
        }
    }

    mapping = actions[0].parameters["mapping"]
    assert "GNU General Public License v2.0" not in mapping
    assert "GNU General Public License v3" not in mapping


def test_schema_qa_accepts_aliases_but_flags_unknown_values(
    tmp_path: Path,
) -> None:
    source = tmp_path / "study.csv"
    source.write_text(
        "id,license\n"
        "1,GNU General Public License\n"
        "2,GNU GENERAL PUBLIC LICENSE\n"
        "3,GNU General Public License v2.0\n"
        "4,GNU General Public Licence\n",
        encoding="utf-8",
    )
    schema = _license_schema()

    report = profile_dataset(source, schema=schema)

    unexpected = [
        issue
        for issue in report.issues
        if issue.type == "schema_unexpected_category"
        and issue.column == "license"
    ]
    assert len(unexpected) == 1
    assert unexpected[0].affected_count == 1


def test_schema_vocabulary_blocks_ai_category_merge_candidates(
    tmp_path: Path,
) -> None:
    source = tmp_path / "study.csv"
    source.write_text(
        "id,license\n"
        "1,GNU General Public License\n"
        "2,GNU General Public License\n"
        "3,GNU General Public License v2.0\n"
        "4,GNU General Public License v3\n"
        "5,GNU GENERAL PUBLIC LICENSE\n",
        encoding="utf-8",
    )
    schema = _license_schema()
    report = profile_dataset(source, schema=schema)
    tables = load_dataset(source).tables

    candidates = build_ai_candidates(
        report,
        tables,
        schema=schema,
    )

    assert not any(
        candidate.action.type == ActionType.NORMALIZE_CATEGORIES
        and candidate.action.column == "license"
        for candidate in candidates
    )
