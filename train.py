import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from config import TRAIN_FILE, LABEL_COLUMN, TEXT_COLUMN, TEST_SIZE, RANDOM_STATE
from preprocess import normalize_for_model


def load_data(file_path=TRAIN_FILE):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
    df[LABEL_COLUMN] = (
        df[LABEL_COLUMN].astype(str).str.strip().str.lower()
        .replace({"mad": "bad", "negative": "bad", "positive": "good"})
    )
    df = df[df[TEXT_COLUMN] != ""]
    df = df.drop_duplicates(subset=[TEXT_COLUMN, LABEL_COLUMN])
    if df[LABEL_COLUMN].nunique() < 2:
        raise ValueError("Training needs at least two label classes.")
    return df[TEXT_COLUMN].tolist(), df[LABEL_COLUMN]


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
            max_features=20000,
            sublinear_tf=True,
        )),
        ("char_tfidf", TfidfVectorizer(
            analyzer="char_wb",
            preprocessor=normalize_for_model,
            lowercase=False,
            ngram_range=(3, 6),
            min_df=1,
            max_features=30000,
            sublinear_tf=True,
        )),
    ], transformer_weights={"word_tfidf": 1.0, "char_tfidf": 0.6})

    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(
            max_iter=2000,
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
    X_train, X_test, y_train, y_test = train_test_split(
        raw_x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    model = create_model()
    model.fit(X_train, y_train)
    holdout = accuracy_score(y_test, model.predict(X_test))
    print(f"Holdout accuracy: {holdout:.4f}")
    return model


def load_model(model_path=None):
    return train()


def predict(texts, model_path=None, model=None):
    if isinstance(texts, str):
        texts = [texts]
    if model is None:
        model = train()
    return model.predict(texts)
