"""Clean UKPN data-centre demand-profile exports.

Column names vary between UKPN extracts, so the canonical columns are
located by pattern rather than by exact name.
"""

import re
from pathlib import Path

import pandas as pd

from ..io_utils import read_table


def _find(columns, *patterns, also=None):
    for c in columns:
        name = str(c)
        if any(re.search(p, name, re.I) for p in patterns):
            if also is None or also(name):
                return c
    return None


def tidy_ukpn_timeseries(path: Path) -> pd.DataFrame:
    """Return site_id | datetime_local (UTC) | utilisation (0-1)."""
    df = read_table(path)

    datetime_col = _find(df.columns, r"datetime")
    util_col = _find(
        df.columns, r"utilisation",
        also=lambda n: bool(re.search(r"half|hour", n, re.I)) or "%" in n,
    )
    site_col = _find(df.columns, r"anonymised", r"name")

    missing = [n for n, c in
               [("datetime", datetime_col), ("utilisation", util_col), ("site", site_col)]
               if c is None]
    if missing:
        raise ValueError(
            f"Could not locate required column(s) {missing} in {Path(path).name}; "
            f"available: {df.columns.tolist()}"
        )

    tidy = pd.DataFrame({
        "site_id": df[site_col].astype(str),
        "datetime_local": pd.to_datetime(df[datetime_col], errors="coerce", utc=True),
        "utilisation": pd.to_numeric(df[util_col], errors="coerce") / 100.0,
    }).dropna()

    return tidy.reset_index(drop=True)
