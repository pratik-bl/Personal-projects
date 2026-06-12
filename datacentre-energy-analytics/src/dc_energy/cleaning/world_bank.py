"""Clean the wide World Bank population export into a long tidy table."""

from pathlib import Path

import pandas as pd

AGGREGATE_ROWS = ["World", "High income", "OECD members", "Euro area"]


def tidy_world_bank_pop(csv_path: Path, year_min: int = 1960, year_max: int = 2024) -> pd.DataFrame:
    """Wide (one column per year) -> long: country | iso3 | year | population.

    Drops missing values and aggregate pseudo-countries.
    """
    df = pd.read_csv(csv_path, low_memory=False)

    id_cols = ["Country Name", "Country Code"]
    year_cols = [c for c in df.columns if str(c).isdigit()]

    long = (
        df.melt(id_vars=id_cols, value_vars=year_cols,
                var_name="year", value_name="population")
        .rename(columns={"Country Name": "country", "Country Code": "iso3"})
    )
    long["year"] = long["year"].astype(int)
    long["population"] = pd.to_numeric(long["population"], errors="coerce")
    long = long.query("@year_min <= year <= @year_max and population.notna()")
    long = long[~long["country"].isin(AGGREGATE_ROWS)]
    return long.reset_index(drop=True)
