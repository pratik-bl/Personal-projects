"""Clean the scraped worldwide data-centre counts (Cloudscene, Aug 2025).

Two raw schemas exist across scrape vintages:
  - ``Country, DataCenters``
  - ``Region, Country, Total Number of data centre``
Both are normalised to country | dc_count (+ region when available).
"""

from pathlib import Path

import pandas as pd

COUNT_CANDIDATES = ["DataCenters", "Total Number of data centre", "dc_count"]


def tidy_global_dc(csv_path: Path) -> pd.DataFrame:
    """Return country | dc_count (and region if present), counts numeric & non-null."""
    df = pd.read_csv(csv_path, low_memory=False)

    count_col = next((c for c in COUNT_CANDIDATES if c in df.columns), None)
    if count_col is None or "Country" not in df.columns:
        raise ValueError(
            f"Unrecognised DC-counts schema in {Path(csv_path).name}: "
            f"{df.columns.tolist()}"
        )

    cols = {"Country": "country", count_col: "dc_count"}
    if "Region" in df.columns:
        cols["Region"] = "region"

    tidy = (
        df[list(cols)]
        .rename(columns=cols)
        .assign(dc_count=lambda x: pd.to_numeric(x["dc_count"], errors="coerce"))
        .dropna(subset=["dc_count"])
        .reset_index(drop=True)
    )
    ordered = [c for c in ("region", "country", "dc_count") if c in tidy.columns]
    return tidy[ordered]
