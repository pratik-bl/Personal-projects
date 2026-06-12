"""Data-centres-per-capita: join scraped DC counts with population."""

import pandas as pd
from unidecode import unidecode


def _norm_country(name):
    if pd.isna(name):
        return None
    return unidecode(str(name)).strip().title()


def dc_per_capita(
    dc_counts: pd.DataFrame,
    population: pd.DataFrame,
    year: int = 2024,
    drop_top_n: int = 0,
) -> pd.DataFrame:
    """Join country DC counts with population and compute density.

    Parameters
    ----------
    dc_counts : tidy frame with ``country`` and ``dc_count``
    population : long frame with ``country``, ``year``, ``population``
    year : population year to use
    drop_top_n : optionally drop the N densest countries (micro-state outliers)

    Returns a frame with ``dc_per_million`` sorted descending.
    """
    pop = population.query("year == @year").copy()

    dc = dc_counts.copy()
    dc["country_norm"] = dc["country"].map(_norm_country)
    pop["country_norm"] = pop["country"].map(_norm_country)

    merged = dc.merge(
        pop[["country_norm", "population"]], on="country_norm", how="inner"
    )
    merged["dc_per_million"] = merged["dc_count"] / merged["population"] * 1_000_000
    merged = merged.sort_values("dc_per_million", ascending=False).reset_index(drop=True)

    if drop_top_n > 0:
        merged = merged.iloc[drop_top_n:].reset_index(drop=True)

    cols = [c for c in ("country", "region", "dc_count", "population", "dc_per_million")
            if c in merged.columns]
    return merged[cols]
