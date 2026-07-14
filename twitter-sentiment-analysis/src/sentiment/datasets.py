"""PyTorch Dataset for tokenised tweets."""

import torch
from torch.utils.data import Dataset

from .config import LABEL_MAP
from .preprocessing import clean_tweet
from .vocab import tokens_to_indices


class TwitterDataset(Dataset):
    """Tokenises, indexes and pads tweets from a DataFrame with (tweet_id, label, text)."""

    def __init__(self, df, word2idx, max_len: int = 30):
        self.samples = []
        for _, row in df.iterrows():
            tokens = clean_tweet(row["text"])
            x_idxs = tokens_to_indices(tokens, word2idx, max_len)
            self.samples.append((x_idxs, LABEL_MAP[row["label"]]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_idxs, y_label = self.samples[idx]
        return torch.LongTensor(x_idxs), torch.tensor(y_label, dtype=torch.long)
