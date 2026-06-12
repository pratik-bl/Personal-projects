"""LightGBM surrogate: predict instantaneous server power (power_w)
from electrical and component telemetry.

Feature/target choices mirror the dissertation notebook: leakage-prone
columns (current_a directly determines power; component power shares) are
dropped before training.
"""

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "power_w"
# current_a leaks the target (P = V*I); *_power_pct are component shares of it.
DROP_COLS = ["weekday", "current_a", "gpu_power_pct", "ram_power_pct", "cpu_temp_c"]

DEFAULT_PARAMS = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.05,
    num_leaves=64,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    seed=42,
    verbosity=-1,
)


def load_training_frame(parquet_paths) -> tuple[pd.DataFrame, pd.Series]:
    """Load telemetry parquets and return (X, y) ready for training."""
    df = pd.concat([pd.read_parquet(p) for p in parquet_paths], ignore_index=True)
    df = df.dropna(subset=[TARGET])
    X = (
        df.select_dtypes(include=["float64", "int64", "float32", "int32"])
        .drop(columns=DROP_COLS + [TARGET], errors="ignore")
    )
    y = df[TARGET]
    return X, y


def train_surrogate(
    parquet_paths,
    out_path: Path | None = None,
    params: dict | None = None,
    num_boost_round: int = 400,
    n_folds: int = 5,
):
    """Train the surrogate with K-fold CV; return (model, cv_rmse_list).

    If ``out_path`` is given the final model (trained on all data) is saved
    in LightGBM text format.
    """
    import lightgbm as lgb
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import KFold

    X, y = load_training_frame(parquet_paths)
    params = {**DEFAULT_PARAMS, **(params or {})}

    rmses = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(X):
        model = lgb.train(
            params,
            lgb.Dataset(X.iloc[train_idx], y.iloc[train_idx]),
            num_boost_round=num_boost_round,
        )
        pred = model.predict(X.iloc[val_idx])
        rmses.append(float(np.sqrt(mean_squared_error(y.iloc[val_idx], pred))))

    final = lgb.train(params, lgb.Dataset(X, y), num_boost_round=num_boost_round)
    if out_path is not None:
        final.save_model(str(out_path))

    return final, rmses
