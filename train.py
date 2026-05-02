import argparse

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from config import (
    CV_FOLDS,
    LABEL_COLUMN,
    MODEL_DIR,
    MODEL_FILE,
    RANDOM_STATE,
    TEST_FILE,
    TEST_SIZE,
    TEXT_COLUMN,
    TRAIN_FILE,
)
from detector import classify_sentence
from preprocess import normalize_for_model


def load_data(file_path=TRAIN_FILE):
    """Load and clean training data from CSV."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    missing = {TEXT_COLUMN, LABEL_COLUMN} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)}. Available: {list(df.columns)}")

    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip().str.lower()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].replace({"mad": "bad", "negative": "bad", "positive": "good"})
    df = df[df[TEXT_COLUMN] != ""]
    df = df.drop_duplicates(subset=[TEXT_COLUMN, LABEL_COLUMN])

    if df[LABEL_COLUMN].nunique() < 2:
        raise ValueError("Training needs at least two label classes.")

    print(f"Loaded {len(df)} clean samples")
    print(f"Label distribution:\n{df[LABEL_COLUMN].value_counts()}")
    return df[TEXT_COLUMN], df[LABEL_COLUMN]


def create_model():
    """Create a strong text-classification pipeline for short sentences."""
    features = FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    preprocessor=normalize_for_model,
                    token_pattern=r"(?u)\b\w+\b",
                    lowercase=False,
                    ngram_range=(1, 4),
                    min_df=1,
                    max_df=0.95,
                    max_features=20000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    preprocessor=normalize_for_model,
                    lowercase=False,
                    ngram_range=(3, 6),
                    min_df=1,
                    max_features=30000,
                    sublinear_tf=True,
                ),
            ),
        ],
        transformer_weights={"word_tfidf": 1.0, "char_tfidf": 0.6},
    )

    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train(show_test_output=False):
    X, y = load_data(TRAIN_FILE)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    model = create_model()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        model,
        param_grid={
            "classifier__C": [0.1, 0.25, 0.5, 1.0, 2.0],
        },
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    print("\nTraining and selecting best model...")
    search.fit(X_train, y_train)
    model = search.best_estimator_

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'=' * 50}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    print(f"Best parameters: {search.best_params_}")
    print(f"Holdout accuracy: {accuracy:.4f}")
    print(f"{'=' * 50}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=sorted(y.unique())))

    print("\nTraining final model on all available data...")
    final_model = create_model()
    final_model.set_params(**search.best_params_)
    final_model.fit(X, y)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final_model, MODEL_FILE)
    print(f"\nModel saved to {MODEL_FILE}")
    evaluate_external_test(final_model, show_output=show_test_output)
    return final_model


def evaluate_external_test(model, file_path=TEST_FILE, show_output=False):
    try:
        df = pd.read_csv(file_path).fillna("")
    except FileNotFoundError:
        print(f"\nExternal test skipped: {file_path} not found")
        return None

    y_true = df[LABEL_COLUMN].astype(str).str.strip().str.lower()
    results = [classify_sentence(text, model=model) for text in df[TEXT_COLUMN]]
    y_pred = [result["label"] for result in results]
    accuracy = accuracy_score(y_true, y_pred)

    if show_output:
        print("External test output:")
        for text, expected, result in zip(df[TEXT_COLUMN], y_true, results):
            status = "PASS" if expected == result["label"] else "FAIL"
            print(
                f"[{status}] text={text!r} expected={expected!r} "
                f"predicted={result['label']!r} confidence={result['confidence']}"
            )
    return accuracy


def load_model(model_path=MODEL_FILE):
    return joblib.load(model_path)


def predict(texts, model_path=MODEL_FILE, model=None):
    if isinstance(texts, str):
        texts = [texts]
    if model is None:
        model = load_model(model_path)
    return model.predict(texts)


def predict_with_confidence(text, model_path=MODEL_FILE, model=None):
    if model is None:
        model = load_model(model_path)
    probabilities = model.predict_proba([text])[0]
    probability_by_label = dict(zip(model.classes_, probabilities))
    label = max(probability_by_label, key=probability_by_label.get)
    return label, round(float(probability_by_label[label]), 4), {
        key: round(float(value), 4) for key, value in probability_by_label.items()
    }


def interactive_loop(model):
    print("\nInteractive prediction mode.")
    print("Type a sentence and I will classify it, then show bad words or phrases found.")
    print("Type 'quit' to exit.\n")
    while True:
        try:
            text = input("Enter a sentence: ")
        except EOFError:
            break
        if text.lower() == "quit":
            break
        result = classify_sentence(text, model=model)
        label = result["label"]
        print(f"Prediction: {label}")
        print(f"Confidence: {result['confidence']}")
        if label == "uncertain":
            print("  \u26a0 Low confidence \u2014 prediction may not be reliable.")
        elif label == "unknown":
            print("  \u2139 Input not recognized \u2014 try rephrasing.")
        if result["bad_terms"]:
            print(f"Bad word/phrase found: {', '.join(result['bad_terms'])}")
        else:
            print("Bad word/phrase found: none")
        print(f"Decision source: {result['source']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a good/bad word classifier.")
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Only train the model; do not start the sentence input prompt.",
    )
    parser.add_argument(
        "--show-test-output",
        action="store_true",
        help="Print every row from test.csv with PASS/FAIL details.",
    )
    args = parser.parse_args()

    trained_model = train(show_test_output=args.show_test_output)
    if not args.no_interactive:
        interactive_loop(trained_model)
