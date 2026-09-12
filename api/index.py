from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

from detector import classify_sentence
from config import MODEL_FILE

app = FastAPI(title="ML Toxic Sentiment Detector API")

MODEL = None
MODEL_LOAD_ERROR = None
try:
    if not MODEL_FILE.exists():
        MODEL_LOAD_ERROR = f"Model file not found: {MODEL_FILE}"
    else:
        MODEL = joblib.load(MODEL_FILE)
except Exception as exc:
    MODEL_LOAD_ERROR = f"Model load failed: {type(exc).__name__}: {exc}"

class PredictionRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {
        "service":"ML Toxic Sentiment Detector",
        "status":"healthy" if MODEL is not None else "model_unavailable",
        "message":"POST text to /predict" if MODEL is not None else MODEL_LOAD_ERROR,
    }

@app.get("/health")
def health():
    if MODEL is None:
        return {"status":"error","error":MODEL_LOAD_ERROR}
    return {"status":"healthy","model":MODEL_FILE.name}

@app.post("/predict")
def predict(request: PredictionRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail=MODEL_LOAD_ERROR)
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Please enter some text.")
    result = classify_sentence(text, model=MODEL)
    return {
        "text":text,
        "prediction":result["label"],
        "confidence":result["confidence"],
        "model_probability":result.get("model_probability",{}),
        "bad_terms":result.get("bad_terms",[]),
        "source":result.get("source","ml_plus_rules"),
    }
