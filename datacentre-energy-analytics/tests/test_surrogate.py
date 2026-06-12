"""Smoke test for the LightGBM surrogate on sample telemetry."""

import pytest

lgb = pytest.importorskip("lightgbm")

from dc_energy.modeling import DROP_COLS, TARGET, load_training_frame, train_surrogate


def test_training_frame_excludes_leaky_columns(samples_dir):
    X, y = load_training_frame(
        [samples_dir / "workstation_2021_may_aug_tidy_SAMPLE.parquet"]
    )
    assert TARGET not in X.columns
    for col in DROP_COLS:
        assert col not in X.columns
    assert len(X) == len(y) > 1000


def test_surrogate_beats_mean_baseline(samples_dir):
    paths = [samples_dir / "workstation_2021_may_aug_tidy_SAMPLE.parquet"]
    _, y = load_training_frame(paths)
    baseline_rmse = float(y.std())

    _, rmses = train_surrogate(paths, num_boost_round=50, n_folds=3)
    mean_rmse = sum(rmses) / len(rmses)

    assert mean_rmse < baseline_rmse  # must out-predict "always guess the mean"
