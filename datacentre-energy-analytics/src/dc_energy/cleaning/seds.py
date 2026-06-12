"""Clean US EIA SEDS extracts (1930-2023) into a single tidy schema.

Handles both layouts the EIA publishes: *long* (a literal ``year`` column)
and *wide* (one column per year), producing
state | msn | year | value | dataset.
"""

from pathlib import Path

import pandas as pd

from ..io_utils import read_table


def _normalize_cols(df):
    return [str(c).strip().lower() for c in df.columns]


def _detect_format(cols):
    return "long" if "year" in cols else "wide"


def tidy_seds(path: Path, tag: str) -> pd.DataFrame:
    """Clean one SEDS file; ``tag`` labels the dataset (e.g. 'co2', 'prices')."""
    df = read_table(path)

    cols_norm = _normalize_cols(df)
    fmt = _detect_format(cols_norm)
    colmap = {norm: orig for norm, orig in zip(cols_norm, df.columns)}

    state_col = colmap.get("state") or colmap.get("statecode")
    msn_col = colmap.get("msn")
    if not msn_col:
        raise ValueError(f"MSN column missing in {Path(path).name}")
    if not state_col:
        # National aggregate file (e.g. production dataset)
        df["state"] = "US"
        state_col = "state"

    if fmt == "long":
        year_col = colmap["year"]
        data_col = next(c for c in df.columns if str(c).lower() == "data")
        tidy = df[[state_col, msn_col, year_col, data_col]].rename(columns={
            state_col: "state", msn_col: "msn", year_col: "year", data_col: "value",
        })
    else:
        year_cols = [c for c in df.columns if str(c).isdigit()]
        tidy = (
            df.melt(id_vars=[state_col, msn_col], value_vars=year_cols,
                    var_name="year", value_name="value")
            .rename(columns={state_col: "state", msn_col: "msn"})
        )
        tidy["year"] = tidy["year"].astype(int)

    tidy["value"] = (
        tidy["value"].replace(["Not Available", "NA", ".", ""], pd.NA).astype("float64")
    )
    tidy["dataset"] = tag
    return tidy.reset_index(drop=True)
