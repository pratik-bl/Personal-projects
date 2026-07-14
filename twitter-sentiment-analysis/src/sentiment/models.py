"""Neural sentiment classifiers: LSTM, BiLSTM, and BiLSTM with attention pooling."""

import torch
import torch.nn as nn


def _make_embedding(vocab_size, embed_dim, embedding_matrix=None, freeze=True):
    embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
    if embedding_matrix is not None:
        embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
    embedding.weight.requires_grad = not freeze
    return embedding


class LSTMSentiment(nn.Module):
    """Unidirectional LSTM over frozen GloVe embeddings; last hidden state -> linear."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 embedding_matrix=None, freeze_emb=True):
        super().__init__()
        self.embedding = _make_embedding(vocab_size, embed_dim, embedding_matrix, freeze_emb)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (h_n, _) = self.lstm(embedded)
        return self.fc(h_n[-1])


class BiLSTMSentiment(nn.Module):
    """Stacked bidirectional LSTM; concatenated final fwd/bwd hidden states -> linear."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 embedding_matrix=None, freeze_emb=True, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = _make_embedding(vocab_size, embed_dim, embedding_matrix, freeze_emb)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (h_n, _) = self.lstm(embedded)
        final_feat = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return self.fc(self.dropout(final_feat))


class BiLSTMAttention(nn.Module):
    """BiLSTM with additive attention pooling over all timesteps.

    Instead of using only the final hidden states, a learned attention
    layer scores every timestep's output and the classifier consumes the
    attention-weighted sum. Padding positions are masked out so they
    receive zero attention weight.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 embedding_matrix=None, freeze_emb=True, num_layers=1,
                 dropout=0.3, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = _make_embedding(vocab_size, embed_dim, embedding_matrix, freeze_emb)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
            bidirectional=True,
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        mask = (x != self.pad_idx)                      # (batch, seq)
        # Guard against fully-padded rows (empty tweets): attend to position 0
        empty_rows = ~mask.any(dim=1)
        if empty_rows.any():
            mask[empty_rows, 0] = True
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)               # (batch, seq, 2*hidden)

        scores = self.attention(lstm_out).squeeze(-1)   # (batch, seq)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (batch, seq, 1)

        context = (lstm_out * weights).sum(dim=1)    