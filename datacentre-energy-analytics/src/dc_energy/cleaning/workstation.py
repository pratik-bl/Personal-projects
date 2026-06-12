"""Clean the 2021 workstation power-telemetry exports.

The raw CSVs use Spanish headers, mixed quoting, day-first timestamps and
contain occasional malformed rows. Output: one row per reading with English
snake_case columns and UTC timestamps.
"""

from pathlib import Path

import pandas as pd
from unidecode import unidecode

from ..io_utils import read_csv_robust

HEADER_MAP = {
    "mac": "mac",
    "weekday": "weekday",
    "fecha_servidor": "datetime",
    "voltaje": "voltage_v",
    "corriente": "current_a",
    "potencia": "power_w",
    "frecuencia": "frequency_hz",
    "energia": "active_energy_kwh",
    "fp": "power_factor",
    "esp32_temp": "esp32_temp_c",
    "workstation_cpu": "cpu_util_pct",
    "workstation_cpu_power": "cpu_power_pct",
    "workstation_cpu_temp": "cpu_temp_c",
    "workstation_gpu": "gpu_util_pct",
    "workstation_gpu_power": "gpu_power_pct",
    "workstation_gpu_temp": "gpu_temp_c",
    "workstation_ram": "ram_util_pct",
    "workstation_ram_power": "ram_power_pct",
}

NON_NUMERIC = ("datetime", "mac", "weekday")


def tidy_workstation(csv_path: Path) -> pd.DataFrame:
    """Clean one raw telemetry CSV into a tidy frame.

    Steps: robust read -> normalise/translate headers -> drop duplicate
    ESP32 timestamp -> parse day-first datetimes as UTC -> sort -> coerce
    all metric columns to numeric.
    """
    df = read_csv_robust(csv_path)

    df.columns = [unidecode(str(c)).lower().strip('" ') for c in df.columns]
    df = df.rename(columns={k: HEADER_MAP.get(k, k) for k in df.columns})
    df = df.drop(columns=["fecha_esp32"], errors="ignore")

    if "datetime" not in df.columns:
        raise KeyError(
            f"'fecha_servidor' column not found in {Path(csv_path).name}; "
            f"headers={df.columns.tolist()}"
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"].astype(str).str.strip('" '),
        dayfirst=True,
        utc=True,
        errors="coerce",
    )
    df = df.dropna(subset=["datetime"]).sort_values("datetime")

    for col in df.columns.difference(list(NON_NUMERIC)):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)
