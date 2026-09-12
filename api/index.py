from pathlib import Path
import sys

# Make repository-root modules importable from /api/index.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from detector import classify_sentence
from train import load_model
from config import MODEL_FILE

app = FastAPI(title="ML Toxic Sentiment Detector API")

MODEL = None
MODEL_LOAD_ERROR = None

try:
    MODEL = load_model(MODEL_FILE)
except Exception as exc:
    MODEL_LOAD_ERROR = f"Model could not be loaded: {type(exc).__name__}: {exc}"


class PredictionRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {
        "status": "ok" if MODEL is not None else "model_error",
        "service": "ML Toxic Sentiment Detector",
    }


@app.get("/health")
def health():
    if MODEL is None:
        return {"status": "error", "error": MODEL_LOAD_ERROR}
    return {"status": "healthy", "model": str(MODEL_FILE.name)}


@app.post("/predict")
def predict(request: PredictionRequest):
    if MODEL is None:
        raise HTTPException(status_code=500, detail=MODEL_LOAD_ERROR)

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Please enter some text.")

    result = classify_sentence(text, model=MODEL)

    return {
        "text": text,
        "prediction": result["label"],
        "confidence": result["confidence"],
        "model_probability": result.get("model_probability", {}),
        "bad_terms": result.get("bad_terms", []),
        "source": result.get("source", "ml_plus_rules"),
    }
