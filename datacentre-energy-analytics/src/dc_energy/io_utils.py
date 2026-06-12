"""Small shared I/O helpers used across cleaners."""

import csv
from pathlib import Path

import pandas as pd


def first_existing(paths) -> Path:
    """Return the first path that exists, else raise FileNotFoundError."""
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    raise FileNotFoundError(f"None of the candidate files exist: {list(map(str, paths))}")


def read_csv_robust(path: Path) -> pd.DataFrame:
    """Read a CSV with the fast C engine, falling back to the tolerant
    Python engine (skipping malformed rows) on parser errors.

    The raw 2021 telemetry exports contain occasional corrupt lines, so a
    plain ``read_csv`` can fail mid-file.
    """
    path = Path(path)
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except pd.errors.ParserError:
        return pd.read_csv(
            path,
            encoding="utf-8-sig",
            engine="python",
            quoting=csv.QUOTE_NONE,
            on_bad_lines="skip",
        )


def read_table(path: Path) -> pd.DataFrame:
    """Read parquet/csv/xlsx based on file extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=0)
    return pd.read_csv(path, low_memory=False)
