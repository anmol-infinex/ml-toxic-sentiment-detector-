import csv
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from config import TRAIN_FILE, LABEL_COLUMN, TEXT_COLUMN, TEST_SIZE, RANDOM_STATE
from preprocess import normalize_for_model


def load_data(file_path=TRAIN_FILE):
    rows = []
    with Path(file_path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or TEXT_COLUMN not in reader.fieldnames or LABEL_COLUMN not in reader.fieldnames:
            raise ValueError(f"CSV must contain '{TEXT_COLUMN}' and '{LABEL_COLUMN}' columns.")
        for row in reader:
            text = str(row.get(TEXT_COLUMN, "") or "").strip()
            label = str(row.get(LABEL_COLUMN, "") or "").strip().lower()
            label = {"mad": "bad", "negative": "bad", "positive": "good"}.get(label, label)
            if text and label in {"bad", "good"}:
                rows.append((text, label))

    dedup = {}
    for text, label in rows:
        dedup[(text, label)] = (text, label)
    rows = list(dedup.values())
    labels = [label for _, label in rows]
    if len(set(labels)) < 2:
        raise ValueError("Training needs at least two label classes.")
    return [text for text, _ in rows], labels


def create_model():
    features = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            analyzer="word",
            preprocessor=normalize_for_model,
            token_pattern=r"(?u)\b\w+\b",
            lowercase=False,
            ngram_range=(1, 4),
            min_df=1,
            max_df=1.0,
            max_features=12000,
            sublinear_tf=True,
        )),
        ("char_tfidf", TfidfVectorizer(
            analyzer="char_wb",
            preprocessor=normalize_for_model,
            lowercase=False,
            ngram_range=(3, 6),
            min_df=1,
            max_features=16000,
            sublinear_tf=True,
        )),
    ], transformer_weights={"word_tfidf": 1.0, "char_tfidf": 0.6})

    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
            C=1.0,
        )),
    ])


def preprocess_texts(texts):
    return [str(text) for text in texts]


def train(show_test_output=False):
    raw_x, y = load_data()
    if len(raw_x) >= 10 and len(set(y)) >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            raw_x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        model = create_model()
        model.fit(X_train, y_train)
        if show_test_output:
            print(f"Holdout accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")
    else:
        model = create_model()

    model.fit(raw_x, y)
    return model


def load_model(model_path=None):
    return train()


def predict(texts, model_path=None, model=None):
    if isinstance(texts, str):
        texts = [texts]
    if model is None:
        model = train()
    return model.predict(texts)
