from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="ML Toxic Sentiment Detector API")

MODEL = None
MODEL_ERROR = None

try:
    from train import create_model, load_data
    from detector import classify_sentence
    raw_x, y = load_data()
    MODEL = create_model()
    MODEL.fit(raw_x, y)
except Exception as exc:
    MODEL_ERROR = {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc().splitlines()[-12:],
    }


class PredictionRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {
        "service": "ML Toxic Sentiment Detector",
        "status": "healthy" if MODEL is not None else "model_error",
        "model": "in-memory classifier" if MODEL is not None else None,
        "error": MODEL_ERROR,
        "message": "POST JSON {\"text\":\"...\"} to /predict",
    }


@app.get("/health")
def health():
    if MODEL is None:
        return {"status": "model_error", "error": MODEL_ERROR}
    return {"status": "healthy", "model": "in-memory classifier"}


@app.post("/predict")
def predict(request: PredictionRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail=MODEL_ERROR or "Model unavailable.")
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
