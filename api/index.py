from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from train import create_model, load_data
from detector import classify_sentence

app = FastAPI(title="ML Toxic Sentiment Detector API")

MODEL = None
MODEL_LOAD_ERROR = None

try:
    # Train once when the serverless instance initializes.
    # The model is kept in memory for subsequent requests in the same instance.
    raw_x, y = load_data()
    MODEL = create_model()
    MODEL.fit(raw_x, y)
except Exception as exc:
    MODEL_LOAD_ERROR = f"Model initialization failed: {type(exc).__name__}: {exc}"

class PredictionRequest(BaseModel):
    text: str

@app.get("/")
def root():
    if MODEL is None:
        return {"service":"ML Toxic Sentiment Detector","status":"model_error","error":MODEL_LOAD_ERROR}
    return {"service":"ML Toxic Sentiment Detector","status":"healthy","message":"POST {text: string} to /predict"}

@app.get("/health")
def health():
    if MODEL is None:
        raise HTTPException(status_code=503, detail=MODEL_LOAD_ERROR)
    return {"status":"healthy","model":"in-memory classifier"}

@app.post("/predict")
def predict(request: PredictionRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail=MODEL_LOAD_ERROR)
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
