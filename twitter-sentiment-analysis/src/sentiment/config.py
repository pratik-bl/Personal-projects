"""Central configuration: paths, label maps, and default hyperparameters."""

from pathlib import Path

# Repo root (two levels up from this file's package dir)
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
TRAIN_FILE = DATA_DIR / "twitter-training-data.txt"
DEV_FILE = DATA_DIR / "twitter-dev-data.txt"
TEST_FILES = [
    DATA_DIR / "twitter-test1.txt",
    DATA_DIR / "twitter-test2.txt",
    DATA_DIR / "twitter-test3.txt",
]
GLOVE_FILE = DATA_DIR / "glove.6B.100d.txt"

MODELS_DIR = ROOT / "models"

LABELS = ["negative", "neutral", "positive"]
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

SEED = 42

# Neural defaults (best config found on the dev set)
EMBED_DIM = 100
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MAX_LEN = 30
MAX_VOCAB_SIZE = 5000
NUM_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3
