"""Tests for density analysis and power anchors."""

import pandas as pd
import pytest

from dc_energy.analysis import dc_per_capita, server_power_anchors


@pytest.fixture(scope="module")
def density(processed_dir):
    dc = pd.read_parquet(processed_dir / "global_dc_counts_tidy.parquet")
    pop = pd.read_parquet(processed_dir / "world_pop_long.parquet")
    return dc_per_capita(dc, pop, year=2024)


class TestDensity:
    def test_positive_and_sorted(self, density):
        assert (density["dc_per_million"] >= 0).all()  # zero-DC countries are valid
        assert density["dc_per_million"].is_monotonic_decreasing

    def test_uk_present_with_sane_value(self, density):
        uk = density[density["country"].str.contains("United Kingdom")]
        assert len(uk) == 1
        assert 1 < uk["dc_per_million"].iloc[0] < 50

    def test_drop_top_n(self, processed_dir, density):
        dc = pd.read_parquet(processed_dir / "global_dc_counts_tidy.parquet")
        pop = pd.read_parquet(processed_dir / "world_pop_long.parquet")
        trimmed = dc_per_capita(dc, pop, year=2024, drop_top_n=3)
        assert len(trimmed) == len(density) - 3
        assert trimmed["dc_per_million"].max() <= density["dc_per_million"].iloc[3]


class TestAnchors:
    def test_anchor_values_consistent(self, samples_dir):
        anchors = server_power_anchors(
            [samples_dir / "workstation_2021_may_aug_tidy_SAMPLE.parquet"]
        )
        assert 0 < anchors["server_avg_W"] < 1000
        # peaks must be ordered: p95 <= p99 <= max
        assert (
            anchors["server_peak_kW_p95"]
            <= anchors["server_peak_kW_p99"]
            <= anchors["server_peak_kW_max"]
        )
        # average (in kW) must not exceed the max peak
        assert anchors["server_avg_W"] / 1000 <= anchors["server_peak_kW_max"]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            server_power_anchors([tmp_path / "nope.parquet"])
