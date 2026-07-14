# Twitter Sentiment Analysis — SemEval-2017 Task 4A

Three-way sentiment classification of tweets (positive / negative / neutral) on the
[SemEval-2017 Task 4, Subtask A](https://alt.qcri.org/semeval2017/task4/) benchmark,
comparing classical machine learning baselines against recurrent neural models with
pre-trained word embeddings.

Originally developed as coursework for **CS918 Natural Language Processing** (MSc,
University of Warwick), refactored from a single research notebook into a reusable
Python package.

## Highlights

- **Classical baselines**: TF-IDF (uni+bigrams) features with Multinomial Naive Bayes, linear SVM, and grid-search-tuned Logistic Regression (MaxEnt)
- **Neural models**: LSTM and stacked bidirectional LSTM over frozen 100-d GloVe embeddings, trained with Adam, dropout, and early stopping on dev loss
- **Twitter-aware preprocessing**: URL/mention placeholder tokens, hashtag stripping, elongation reduction, emoticon-preserving tokenisation
- **Official evaluation**: SemEval macro-F1 over the positive and negative classes, with per-model confusion matrices and misclassification analysis
- **Reproducible**: seeded runs, pinned dependencies, config in one place, CLI scripts for every experiment

## Results

Macro-averaged F1 over the positive and negative classes (official SemEval-2017 4A
metric), on the three held-out test sets:

| Model | Features | Test 1 | Test 2 | Test 3 |
|---|---|---|---|---|
| Multinomial Naive Bayes | TF-IDF (1–2 grams) | 0.416 | 0.463 | 0.410 |
| Linear SVM | TF-IDF (1–2 grams) | 0.562 | 0.593 | 0.532 |
| Logistic Regression (tuned, C=10) | TF-IDF (1–2 grams) | 0.565 | 0.589 | 0.535 |
| LSTM (1 layer) | GloVe 100d (frozen) | 0.540 | 0.545 | 0.534 |
| BiLSTM (2 layers, dropout 0.3) | GloVe 100d (frozen) | 0.603 | 0.612 | 0.582 |
| **BiLSTM + attention pooling** (post-submission) | GloVe 100d (frozen), class-weighted loss | **0.656** | **0.657** | **0.602** |

Key observations from the error analysis:

- The single-layer LSTM underperformed the tuned linear baselines — final-hidden-state pooling over short noisy tweets loses too much signal.
- Of the coursework models, the 2-layer BiLSTM was strongest, mainly through better separation of *negative* vs *neutral* tweets (visible in the confusion matrices).
- Replacing final-hidden-state pooling with masked attention over all timesteps (added post-submission, with class-weighted loss and Twitter-aware tokenisation) gained a further ~5 points of macro-F1 on every test set — on short noisy tweets, sentiment-bearing tokens can appear anywhere in the sequence.
- All models struggled with sarcasm and mixed-sentiment tweets, where contradictory cues ("crying" in a positive fan tweet) defeat both lexical features and sequence models.

Full write-up in [`report/lab-report.docx`](report/lab-report.docx); original
experiment notebook in [`notebooks/original-experiments.ipynb`](notebooks/original-experiments.ipynb).

## Project structure

```
├── src/sentiment/          # installable package
│   ├── config.py            # paths, label maps, default hyperparameters
│   ├── data.py              # dataset file loading
│   ├── preprocessing.py     # tweet normalisation + TweetTokenizer
│   ├── vocab.py             # vocabulary + GloVe embedding matrix
│   ├── datasets.py          # PyTorch Dataset
│   ├── models.py            # LSTM, BiLSTM, BiLSTM+attention
│   ├── training.py          # seeding, train/eval loops, early stopping, prediction
│   └── evaluation.py        # SemEval macro-F1, confusion matrices, error analysis
├── scripts/
│   ├── train_baselines.py   # NB / SVM / LogReg experiments
│   └── train_lstm.py        # neural experiments (+ grid search)
├── notebooks/               # original research notebook
├── report/                  # lab report
└── data/                    # datasets & embeddings (see data/README.md)
```

## Setup

```bash
git clone https://github.com/pratik-bl/Personal-projects.git
cd Personal-projects/twitter-sentiment-analysis

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
python -c "import nltk; nltk.download('punkt')"
```

Then place the SemEval data files and `glove.6B.100d.txt` in `data/` —
see [`data/README.md`](data/README.md) for download instructions.

## Usage

```bash
# Classical baselines (all three models)
python scripts/train_baselines.py

# Tuned Logistic Regression with error analysis
python scripts/train_baselines.py --model logreg --tune --show-errors 10

# BiLSTM with the best-found hyperparameters
python scripts/train_lstm.py

# Grid search over hidden size / dropout / learning rate on dev macro-F1
python scripts/train_lstm.py --tune

# BiLSTM with attention pooling and class-weighted loss
python scripts/train_lstm.py --arch bilstm-attn --class-weight --save models/bilstm_attn.pt
```

## Improvements over the original coursework

The refactor also adds three model-side upgrades that were not part of the marked
submission. Re-benchmarked on Kaggle (GPU), together they lift macro-F1 from
0.603/0.612/0.582 to **0.656/0.657/0.602** across the three test sets:

1. **Attention pooling** (`--arch bilstm-attn`): a masked additive-attention layer over all BiLSTM timesteps replaces final-hidden-state pooling, which helps on short texts where sentiment-bearing tokens can appear anywhere.
2. **TweetTokenizer**: NLTK's Twitter-aware tokenizer replaces the generic `word_tokenize`, keeping emoticons intact and reducing word elongation natively.
3. **Class-weighted loss** (`--class-weight`): inverse-frequency weights counter the neutral-heavy label distribution in both the sklearn baselines and the neural loss.

## Possible future work

- Fine-tune (unfreeze) embeddings after initial convergence
- Twitter-specific embeddings (`glove.twitter.27B`) instead of Wikipedia-trained ones
- Transformer baselines (BERTweet / RoBERTa) for comparison

## License

MIT — see [LICENSE](LICENSE). Dataset © SemEval-2017 Task 4 organisers;
GloVe embeddings © Stanford NLP (Public Domain Dedication and License).
