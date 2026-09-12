import csv
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'models'
MODEL_FILE = MODEL_DIR / 'sentiment_model.joblib'


def read_data():
    rows=[]
    with open(ROOT/'train.csv', 'r', encoding='utf-8-sig', newline='') as f:
        reader=csv.DictReader(f)
        for row in reader:
            text=str(row.get('text','')).strip()
            label=str(row.get('label','')).strip().lower()
            if text and label:
                label={'positive':'good','negative':'bad','mad':'bad'}.get(label,label)
                rows.append((text,label))
    if len(rows)<2 or len({label for _,label in rows})<2:
        raise RuntimeError('Training data must contain both good and bad labels.')
    return rows


def normalize(text):
    from preprocess import normalize_for_model
    value=normalize_for_model(text)
    return value if value.strip() else 'emptytoken'

rows=read_data()
texts=[normalize(t) for t,_ in rows]
labels=[y for _,y in rows]

features=FeatureUnion([
    ('word', TfidfVectorizer(token_pattern=r'(?u)\\b\\w+\\b', lowercase=False, ngram_range=(1,4), min_df=1, max_df=1.0, sublinear_tf=True, max_features=20000)),
    ('char', TfidfVectorizer(analyzer='char_wb', lowercase=False, ngram_range=(3,6), min_df=1, sublinear_tf=True, max_features=30000)),
])
model=Pipeline([
    ('features',features),
    ('classifier',LogisticRegression(max_iter=2000,class_weight='balanced',solver='liblinear',random_state=42,C=1.0)),
])
model.fit(texts, labels)
MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(model, MODEL_FILE)
print(f'Model saved: {MODEL_FILE}')
