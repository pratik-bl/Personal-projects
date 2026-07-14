"""Training utilities: seeding, epoch loops, early stopping, prediction."""

import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import INV_LABEL_MAP, LABEL_MAP
from .datasets import TwitterDataset


def set_seed(seed: int = 42):
    """Seed python, numpy and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(labels) -> torch.Tensor:
    """Inverse-frequency class weights (index-aligned with LABEL_MAP)."""
    counts = np.zeros(len(LABEL_MAP))
    for label in labels:
        counts[LABEL_MAP[label]] += 1
    weights = counts.sum() / (len(counts) * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total += y_batch.size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        logits = model(x_batch)
        total_loss += criterion(logits, y_batch).item()
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total += y_batch.size(0)
    return total_loss / len(loader), correct / total


def fit(model, train_loader, dev_loader, optimizer, criterion, device,
        num_epochs: int = 10, patience: int = 3, verbose: bool = True):
    """Train with early stopping on dev loss; restores the best weights."""
    best_dev_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        dev_loss, dev_acc = eval_epoch(model, dev_loader, criterion, device)

        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs} | "
                  f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
                  f"dev loss {dev_loss:.4f} acc {dev_acc:.3f}")

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print("Early stopping triggered.")
                break

    model.load_state_dict(best_weights)
    return model, best_dev_loss


@torch.no_grad()
def predict(model, df, word2idx, max_len: int, device, batch_size: int = 32) -> dict:
    """Return {tweet_id: predicted_label_str} for every tweet in ``df``."""
    model.eval()
    loader = DataLoader(TwitterDataset(df, word2idx, max_len),
                        batch_size=batch_size, shuffle=False)
    preds = []
    for x_batch, _ in loader:
        logits = model(x_batch.to(device))
        preds.extend(logits.argmax(dim=1).cpu().numpy())

    return {tweet_id: INV_LABEL_MAP[int(p)]
            for tweet_id, p in zip(df["tweet_id"], preds)}
