"""Central path configuration.

All paths resolve relative to the repository root by default and can be
overridden with the ``DC_ENERGY_DATA`` environment variable, so the same
code runs locally, in CI, and on a server.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("DC_ENERGY_DATA", REPO_ROOT / "data"))
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"
OUTPUTS_DIR = REPO_ROOT / "outputs"
MODELS_DIR = REPO_ROOT / "models"
