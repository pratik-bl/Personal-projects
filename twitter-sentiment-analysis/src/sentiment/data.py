"""Loading the SemEval-2017 Task 4A tab-separated tweet files."""

import pandas as pd


def load_data(file_path) -> pd.DataFrame:
    """Load a `tweet_id \\t label \\t text` file into a DataFrame."""
    rows = []
    with open(file_path, mode="r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 3:
                rows.append(parts)
    return pd.DataFrame(rows, columns=["tweet_id", "label", "text"])


def read_gold(file_path) -> dict:
    """Return {tweet_id: gold_label} for a test file."""
    id_gts = {}
    with open(file_path, "r", encoding="utf8") as fh:
        for line in fh:
            fields = line.strip().split("\t")
            if len(fields) >= 2:
                id_gts[fields[0]] = fields[1]
    return id_gts
