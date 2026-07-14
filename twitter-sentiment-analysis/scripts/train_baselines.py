"""Train and evaluate the classical baselines (Naive Bayes, SVM, Logistic Regression).

Usage:
    python scripts/train_baselines.py                     # all models, default settings
    python scripts/train_baselines.py --model logreg --tune
    python scripts/train_baselines.py --class-weight --show-errors 5
"""

import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sentiment.config import DEV_FILE, SEED, TEST_FILES, TRAIN_FILE
from sentiment.data import load_data
from sentiment.evaluation import evaluate, get_misclassifications
from sentiment.preprocessing import text_to_clean_str


def build_models(class_weight, tune, X_train, y_train):
    cw = "balanced" if class_weight else None
    models = {
        "NaiveBayes": MultinomialNB(),
        "SVM": LinearSVC(class_weight=cw, random_state=SEED),
        "LogReg": LogisticRegression(max_iter=1000, solver="liblinear",
                                     class_weight=cw, random_state=SEED),
    }

    if tune:
        grid = GridSearchCV(
            estimator=LogisticRegression(max_iter=1000, class_weight=cw, random_state=SEED),
            param_grid={"C": [0.01, 0.1, 1.0, 10.0], "penalty": ["l2"], "solver": ["liblinear"]},
            scoring=make_scorer(f1_score, average="macro"),
            cv=3, n_jobs=-1, verbose=1,
        )
        grid.fit(X_train, y_train)
        print(f"Best params: {grid.best_params_} | best CV macro-F1: {grid.best_score_:.4f}")
        models["LogReg"] = grid.best_estimator_

    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["nb", "svm", "logreg", "all"], default="all")
    parser.add_argument("--tune", action="store_true", help="grid-search LogReg C")
    parser.add_argument("--class-weight", action="store_true", help="use balanced class weights")
    parser.add_argument("--show-errors", type=int, default=0,
                        help="print N misclassified test1 tweets for error analysis")
    args = parser.parse_args()

    print("Loading data...")
    train_df = load_data(TRAIN_FILE)
    _ = load_data(DEV_FILE)  # dev set is used by the neural pipeline; loaded here as a sanity check
    test_dfs = [load_data(f) for f in TEST_FILES]

    print("Extracting TF-IDF features (uni+bigrams, 10k features)...")
    train_texts = [text_to_clean_str(t) for t in train_df["text"]]
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    X_train = tfidf.fit_transform(train_texts)
    y_train = train_df["label"]

    models = build_models(args.class_weight, args.tune, X_train, y_train)
    selected = {"nb": ["NaiveBayes"], "svm": ["SVM"], "logreg": ["LogReg"]}.get(
        args.model, list(models)
    )

    for name in selected:
        model = models[name]
        print(f"\n=== {name} ===")
        model.fit(X_train, y_train)
        for test_file, df_test in zip(TEST_FILES, test_dfs):
            X_test = tfidf.transform([text_to_clean_str(t) for t in df_test["text"]])
            preds = model.predict(X_test)
            id_preds = dict(zip(df_test["tweet_id"], preds))
            evaluate(id_preds, test_file, classifier=name)

        if args.show_errors:
            X_test = tfidf.transform([text_to_clean_str(t) for t in test_dfs[0]["text"]])
            id_preds = dict(zip(test_dfs[0]["tweet_id"], model.predict(X_test)))
            errors = get_misclassifications(id_preds, test_dfs[0])
            print(f"{len(errors)} misclassified tweets on test1; showing {args.show_errors}:")
            for tweet_id, gold, pred, text in errors[: args.show_errors]:
                print(f"  [{tweet_id}] gold={gold} pred={pred} :: {text[:100]}")


if __name__ == "__main__":
    main()
