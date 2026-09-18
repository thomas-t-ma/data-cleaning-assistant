#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is required but was not found." >&2
  exit 1
fi

if [[ ! -d .git ]]; then
  echo "ERROR: run this from a Git clone of the project." >&2
  exit 1
fi

echo "==> Syncing source from GitHub"
git fetch origin main
git rebase --autostash origin/main

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "ERROR: Python 3.11+ was not found." >&2
  echo "On macOS, try: python3 --version" >&2
  echo "If that also fails, install Python 3.11+ (for example: brew install python)." >&2
  exit 1
fi

PY_VERSION="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "ERROR: Python $PY_VERSION found; Python 3.11+ is required." >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "==> Creating .venv with $PYTHON_BIN ($PY_VERSION)"
  "$PYTHON_BIN" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
fi

if command -v shasum >/dev/null 2>&1; then
  DEPS_HASH="$(shasum -a 256 pyproject.toml | awk '{print $1}')"
else
  DEPS_HASH="$(sha256sum pyproject.toml | awk '{print $1}')"
fi

STAMP_FILE=".venv/.data-cleaning-assistant-pyproject.sha256"
OLD_HASH=""
[[ -f "$STAMP_FILE" ]] && OLD_HASH="$(cat "$STAMP_FILE")"

ENV_OK=1
if ! .venv/bin/python -c 'import data_cleaning_assistant, openpyxl, pandas, pyarrow, pydantic, typer, rich, dateutil, yaml' >/dev/null 2>&1; then
  ENV_OK=0
fi
if ! .venv/bin/python -m pip check >/dev/null 2>&1; then
  ENV_OK=0
fi

if [[ "$DEPS_HASH" != "$OLD_HASH" || "$ENV_OK" -ne 1 ]]; then
  if [[ "$ENV_OK" -ne 1 && "$DEPS_HASH" == "$OLD_HASH" ]]; then
    echo "==> Environment is incomplete; repairing dependencies"
  else
    echo "==> Installing/updating project dependencies"
  fi
  .venv/bin/python -m pip install -e ".[dev]"
  .venv/bin/python -c 'import data_cleaning_assistant, openpyxl, pandas, pyarrow, pydantic, typer, rich, dateutil, yaml'
  .venv/bin/python -m pip check
  printf '%s\n' "$DEPS_HASH" > "$STAMP_FILE"
else
  echo "==> Dependencies verified"
fi

if [[ "${1:-}" != "--no-test" ]]; then
  echo "==> Running tests"
  .venv/bin/python -m pytest -q
fi

echo
printf 'Ready. Python: %s\n' "$ROOT_DIR/.venv/bin/python"
printf 'CLI: %s\n' "$ROOT_DIR/.venv/bin/data-cleaner"
