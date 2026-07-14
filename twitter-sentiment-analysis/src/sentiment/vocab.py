"""Vocabulary construction and pre-trained GloVe embedding loading."""

import collections

import numpy as np

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_vocab(all_token_lists, max_vocab_size: int = 5000):
    """Build (word2idx, idx2word) from tokenised texts, keeping the most frequent words."""
    freq = collections.Counter()
    for tokens in all_token_lists:
        freq.update(tokens)

    most_common = freq.most_common(max_vocab_size - 2)
    idx2word = [PAD_TOKEN, UNK_TOKEN] + [w for w, _ in most_common]
    word2idx = {w: i for i, w in enumerate(idx2word)}
    return word2idx, idx2word


def tokens_to_indices(tokens, word2idx, max_len: int = 30):
    """Map tokens to vocab indices, truncating/padding to ``max_len``."""
    unk = word2idx[UNK_TOKEN]
    idxs = [word2idx.get(w, unk) for w in tokens][:max_len]
    idxs += [word2idx[PAD_TOKEN]] * (max_len - len(idxs))
    return idxs


def load_glove_embeddings(glove_path, word2idx, embedding_dim: int = 100, seed: int = 42):
    """Build an embedding matrix from a GloVe text file.

    In-vocabulary words get their GloVe vector; out-of-vocabulary words
    (including <PAD>/<UNK>) get small random-uniform vectors.
    """
    rng = np.random.default_rng(seed)
    embedding_matrix = rng.uniform(-0.05, 0.05, (len(word2idx), embedding_dim))

    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            vals = line.rstrip().split(" ")
            word, vector = vals[0], vals[1:]
            if len(vector) != embedding_dim:
                continue
            if word in word2idx:
                embedding_matrix[word2idx[word]] = np.asarray(vector, dtype=float)
    return embedding_matrix
