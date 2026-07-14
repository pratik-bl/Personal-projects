"""SemEval-2017 Task 4A evaluation: macro-F1 over positive & negative classes."""

import pandas as pd

from .config import LABELS
from .data import read_gold


def _per_class_counts(id_preds: dict, id_gts: dict) -> dict:
    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in LABELS}
    for tweet_id, gt in id_gts.items():
        pred = id_preds.get(tweet_id, "neutral")
        if gt == pred:
            counts[gt]["tp"] += 1
        else:
            counts[gt]["fn"] += 1
            counts[pred]["fp"] += 1
    return counts


def semeval_macro_f1(id_preds: dict, test_file) -> float:
    """Official SemEval-2017 4A metric: mean F1 of the positive and negative classes."""
    counts = _per_class_counts(id_preds, read_gold(test_file))

    f1_sum = 0.0
    for cat in ("positive", "negative"):
        tp, fp, fn = counts[cat]["tp"], counts[cat]["fp"], counts[cat]["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_sum += 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return f1_sum / 2


def confusion_matrix(id_preds: dict, test_file) -> pd.DataFrame:
    """Row-normalised confusion matrix (rows = predicted, cols = gold)."""
    id_gts = read_gold(test_file)
    conf = {c1: {c2: 0 for c2 in LABELS} for c1 in LABELS}
    for tweet_id, gt in id_gts.items():
        pred = id_preds.get(tweet_id, "neutral")
        conf[pred][gt] += 1

    df = pd.DataFrame(conf).T[LABELS]
    row_sums = df.sum(axis=1).replace(0, 1)
    return df.div(row_sums, axis=0).round(3)


def evaluate(id_preds: dict, test_file, classifier: str, verbose: bool = True) -> float:
    """Print and return the SemEval macro-F1, plus the confusion matrix."""
    f1 = semeval_macro_f1(id_preds, test_file)
    if verbose:
        print(f"{test_file} ({classifier}): macro-F1 = {f1:.3f}")
        print(confusion_matrix(id_preds, test_file), "\n")
    return f1


def get_misclassifications(id_preds: dict, df) -> list:
    """Return (tweet_id, gold, pred, text) for every wrongly classified tweet."""
    out = []
    for _, row in df.iterrows():
        pred = id_preds.get(row["tweet_id"], "neutral")
        if row["label"] != pred:
            out.append((row["tweet_id"], row["label"], pred, row["text"]))
    return out
