from __future__ import annotations

import re
import tomllib
from pathlib import Path

from data_cleaning_assistant import __version__


ROOT = Path(__file__).resolve().parents[1]


def test_package_version_metadata_and_readme_stay_in_sync() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project_version = pyproject["project"]["version"]

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"Current version: \*\*v([^*]+)\*\*\.", readme)
    assert match is not None

    assert __version__ == project_version == match.group(1)
