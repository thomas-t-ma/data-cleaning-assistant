from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class LoadedDataset:
    path: Path
    source_type: str
    tables: dict[str, pd.DataFrame]


def load_dataset(path: str | Path) -> LoadedDataset:
    """Load a supported dataset without mutating or normalizing source values."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        # Keep textual missing-value tokens intact so the profiler can distinguish
        # true nulls from values such as "N/A", "unknown", and ".".
        frame = pd.read_csv(
            path,
            dtype=object,
            keep_default_na=False,
            na_filter=False,
            on_bad_lines="warn",
        )
        frame.columns = [str(column) for column in frame.columns]
        return LoadedDataset(path=path, source_type="csv", tables={path.stem: frame})

    if suffix in {".xlsx", ".xlsm"}:
        tables = pd.read_excel(
            path,
            sheet_name=None,
            dtype=object,
            engine="openpyxl",
            keep_default_na=False,
            na_filter=False,
        )
        for frame in tables.values():
            frame.columns = [str(column) for column in frame.columns]
        return LoadedDataset(path=path, source_type="excel", tables=tables)

    if suffix == ".parquet":
        frame = pd.read_parquet(path)
        frame.columns = [str(column) for column in frame.columns]
        return LoadedDataset(path=path, source_type="parquet", tables={path.stem: frame})

    raise ValueError(
        f"Unsupported file type '{suffix}'. Supported formats: .csv, .xlsx, .xlsm, .parquet"
    )
