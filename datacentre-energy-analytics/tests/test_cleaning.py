"""Unit tests for the cleaners, run against the bundled sample data."""

import pandas as pd
import pytest

from dc_energy.cleaning import (
    tidy_global_dc,
    tidy_ukpn_timeseries,
    tidy_workstation,
    tidy_world_bank_pop,
)


class TestWorkstation:
    @pytest.fixture(scope="class")
    def tidy(self, samples_dir):
        return tidy_workstation(samples_dir / "workstation_may_aug_2021_raw_SAMPLE.csv")

    def test_headers_translated(self, tidy):
        assert "power_w" in tidy.columns
        assert "voltage_v" in tidy.columns
        assert "potencia" not in tidy.columns  # Spanish header gone

    def test_datetime_parsed_utc_sorted(self, tidy):
        assert pd.api.types.is_datetime64_any_dtype(tidy["datetime"])
        assert str(tidy["datetime"].dt.tz) == "UTC"
        assert tidy["datetime"].is_monotonic_increasing

    def test_metrics_numeric_and_plausible(self, tidy):
        assert pd.api.types.is_numeric_dtype(tidy["power_w"])
        power = tidy["power_w"].dropna()
        assert (power >= 0).all()
        assert power.mean() < 2000  # a workstation, not a rack


class TestWorldBank:
    @pytest.fixture(scope="class")
    def tidy(self, samples_dir):
        return tidy_world_bank_pop(samples_dir / "World_bank_population.csv")

    def test_schema(self, tidy):
        assert list(tidy.columns) == ["country", "iso3", "year", "population"]

    def test_year_bounds_and_no_nulls(self, tidy):
        assert tidy["year"].between(1960, 2024).all()
        assert tidy["population"].notna().all()

    def test_aggregates_dropped(self, tidy):
        assert "World" not in set(tidy["country"])


class TestDcCounts:
    @pytest.fixture(scope="class")
    def tidy(self, samples_dir):
        return tidy_global_dc(samples_dir / "data-centres-worldwide-general-dataset.csv")

    def test_schema_and_positive_counts(self, tidy):
        assert {"country", "dc_count"} <= set(tidy.columns)
        assert (tidy["dc_count"] >= 0).all()  # 71 countries genuinely have zero DCs

    def test_known_country_present(self, tidy):
        assert tidy["country"].str.contains("United Kingdom").any()


class TestUkpn:
    def test_tidy_from_sample(self, samples_dir):
        tidy = tidy_ukpn_timeseries(
            samples_dir / "ukpn-data-centre-demand-profiles_SAMPLE.csv"
        )
        assert list(tidy.columns) == ["site_id", "datetime_local", "utilisation"]
        assert tidy["utilisation"].between(0, 5).all()  # 0-1 plus rare >100% spikes
        assert tidy["datetime_local"].notna().all()
