"""Single-server power anchors from workstation telemetry.

These anchors (average draw, P95/P99/max peaks) are used to convert
site-level utilisation percentages into absolute kW in the UK demand
reconstruction. See the dissertation's methodology chapter for the
caveats on workstation-vs-server representativeness.
"""

from pathlib import Path

import pandas as pd


def server_power_anchors(parquet_paths) -> dict:
    """Compute power anchors from one or more tidy telemetry parquets.

    Returns a dict with ``server_avg_W`` and peak kW anchors at P95/P99/max.
    """
    frames = []
    for p in parquet_paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(p)
        frames.append(pd.read_parquet(p, columns=["power_w"]))
    ws = pd.concat(frames, ignore_index=True)["power_w"].dropna()

    if ws.empty:
        raise ValueError("No power_w readings found in the supplied files")

    return {
        "server_avg_W": float(ws.mean()),
        "server_peak_kW_p95": float(ws.quantile(0.95) / 1000.0),
        "server_peak_kW_p99": float(ws.quantile(0.99) / 1000.0),
        "server_peak_kW_max": float(ws.max() / 1000.0),
        "n_readings": int(ws.size),
    }
