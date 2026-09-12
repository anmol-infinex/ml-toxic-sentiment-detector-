import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import accuracy_score
from config import TRAIN_FILE, MODEL_DIR, MODEL_FILE, TEXT_COLUMN, LABEL_COLUMN, TEST_SIZE, RANDOM_STATE
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
    return df[TEXT_COLUMN], df[LABEL_COLUMN]


def create_model():
    # normalize_for_model is applied explicitly before vectorization. This avoids
    # an sklearn preprocessor/analyzer incompatibility during deployment builds.
    return Pipeline([
        ("features", FeatureUnion([
            ("word_tfidf", TfidfVectorizer(
                analyzer="word", token_pattern=r"(?u)\b\w+\b",
                lowercase=False, ngram_range=(1, 4), min_df=1,
                max_df=1.0, max_features=20000, sublinear_tf=True)),
            ("char_tfidf", TfidfVectorizer(
                analyzer="char_wb", lowercase=False, ngram_range=(3, 6),
                min_df=1, max_features=30000, sublinear_tf=True)),
        ], transformer_weights={"word_tfidf": 1.0, "char_tfidf": 0.6})),
        ("classifier", LogisticRegression(
            max_iter=2000, class_weight="balanced", solver="liblinear",
            random_state=RANDOM_STATE, C=1.0)),
    ])


def preprocess_texts(texts):
    return [normalize_for_model(str(text)) for text in texts]


def train(show_test_output=False):
    X, y = load_data()
    X = preprocess_texts(X.tolist())
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    model = create_model()
    model.fit(X_train, y_train)
    print(f"Holdout accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")
    model.fit(X, y)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")
    return model


def load_model(model_path=MODEL_FILE):
    return joblib.load(model_path)


def predict(texts, model_path=MODEL_FILE, model=None):
    if isinstance(texts, str):
        texts = [texts]
    if model is None:
        model = load_model(model_path)
    return model.predict(preprocess_texts(texts))
