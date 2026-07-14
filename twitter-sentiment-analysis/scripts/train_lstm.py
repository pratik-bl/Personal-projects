"""Train and evaluate a neural sentiment classifier (LSTM / BiLSTM / BiLSTM+attention).

Usage:
    python scripts/train_lstm.py                              # BiLSTM, best-found hyperparams
    python scripts/train_lstm.py --arch bilstm-attn
    python scripts/train_lstm.py --tune                       # small grid search on dev macro-F1
    python scripts/train_lstm.py --class-weight --save models/bilstm.pt
"""

import argparse
import copy
import itertools

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sentiment import config as cfg
from sentiment.data import load_data
from sentiment.datasets import TwitterDataset
from sentiment.evaluation import evaluate, semeval_macro_f1
from sentiment.models import ARCHITECTURES
from sentiment.preprocessing import clean_tweet
from sentiment.training import (class_weights, fit, predict, set_seed)
from sentiment.vocab import build_vocab, load_glove_embeddings


def build_model(arch, word2idx, embedding_matrix, hidden_dim, dropout, num_layers):
    model_cls = ARCHITECTURES[arch]
    kwargs = dict(
        vocab_size=len(word2idx),
        embed_dim=cfg.EMBED_DIM,
        hidden_dim=hidden_dim,
        num_classes=len(cfg.LABELS),
        embedding_matrix=embedding_matrix,
        freeze_emb=True,
    )
    if arch != "lstm":
        kwargs.update(num_layers=num_layers, dropout=dropout)
    return model_cls(**kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=list(ARCHITECTURES), default="bilstm")
    parser.add_argument("--hidden-dim", type=int, default=cfg.HIDDEN_DIM)
    parser.add_argument("--num-layers", type=int, default=cfg.NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=cfg.DROPOUT)
    parser.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS)
    parser.add_argument("--max-len", type=int, default=cfg.MAX_LEN)
    parser.add_argument("--glove", default=str(cfg.GLOVE_FILE))
    parser.add_argument("--class-weight", action="store_true",
                        help="weight the loss by inverse class frequency")
    parser.add_argument("--tune", action="store_true",
                        help="grid-search hidden_dim/dropout/lr on dev macro-F1")
    parser.add_argument("--save", default=None, help="path to save best model state_dict")
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data...")
    train_df = load_data(cfg.TRAIN_FILE)
    dev_df = load_data(cfg.DEV_FILE)
    test_dfs = [load_data(f) for f in cfg.TEST_FILES]

    print("Building vocabulary and loading GloVe embeddings...")
    all_tokens = [clean_tweet(t) for t in train_df["text"]] + \
                 [clean_tweet(t) for t in dev_df["text"]]
    word2idx, _ = build_vocab(all_tokens, max_vocab_size=cfg.MAX_VOCAB_SIZE)
    embedding_matrix = load_glove_embeddings(args.glove, word2idx, cfg.EMBED_DIM, args.seed)

    train_loader = DataLoader(TwitterDataset(train_df, word2idx, args.max_len),
                              batch_size=cfg.BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(TwitterDataset(dev_df, word2idx, args.max_len),
                            batch_size=cfg.BATCH_SIZE, shuffle=False)

    weights = class_weights(train_df["label"]).to(device) if args.class_weight else None
    criterion = nn.CrossEntropyLoss(weight=weights)

    if args.tune:
        best_f1, best_config, best_state, best_model = -1.0, None, None, None
        grid = itertools.product([64, 128], [0.2, 0.3], [5e-4, 1e-3])
        for hidden_dim, dropout, lr in grid:
            print(f"\n--- config: hidden={hidden_dim} dropout={dropout} lr={lr} ---")
            set_seed(args.seed)
            model = build_model(args.arch, word2idx, embedding_matrix,
                                hidden_dim, dropout, args.num_layers).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            fit(model, train_loader, dev_loader, optimizer, criterion, device,
                num_epochs=5, patience=cfg.EARLY_STOPPING_PATIENCE, verbose=False)
            dev_preds = predict(model, dev_df, word2idx, args.max_len, device)
            dev_f1 = semeval_macro_f1(dev_preds, cfg.DEV_FILE)
            print(f"dev macro-F1: {dev_f1:.3f}")
            if dev_f1 > best_f1:
                best_f1, best_config = dev_f1, (hidden_dim, dropout, lr)
                best_state = copy.deepcopy(model.state_dict())
                best_model = model
        print(f"\nBest config: hidden={best_config[0]} dropout={best_config[1]} "
              f"lr={best_config[2]} (dev macro-F1 {best_f1:.3f})")
        best_model.load_state_dict(best_state)
        model = best_model
    else:
        model = build_model(args.arch, word2idx, embedding_matrix,
                            args.hidden_dim, args.dropout, args.num_layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        fit(model, train_loader, dev_loader, optimizer, criterion, device,
            num_epochs=args.epochs, patience=cfg.EARLY_STOPPING_PATIENCE)

    print(f"\n=== {args.arch} test results ===")
    for test_file, df_test in zip(cfg.TEST_FILES, test_dfs):
        id_preds = predict(model, df_test, word2idx, args.max_len, device)
        evaluate(id_preds, test_file, classifier=args.arch)

    if args.save:
        torch.save(model.state_dict(), args.save)
        print(f"Saved model to {args.save}")


if __name__ == "__main__":
    main()
