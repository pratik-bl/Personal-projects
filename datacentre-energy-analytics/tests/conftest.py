from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SAMPLES = REPO / "data" / "samples"
PROCESSED = REPO / "data" / "processed"


@pytest.fixture(scope="session")
def samples_dir() -> Path:
    return SAMPLES


@pytest.fixture(scope="session")
def processed_dir() -> Path:
    return PROCESSED
