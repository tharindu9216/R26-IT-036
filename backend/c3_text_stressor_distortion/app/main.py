from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import db
from .diary import router as diary_router
from .fusion.router import router as fusion_router
from .predictors import cbt_predictor, stress_predictor, topic_predictor


app = FastAPI(title="Stress Assistant API")
db.init_db()
app.include_router(diary_router)
app.include_router(fusion_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "stress_model_loaded": stress_predictor.is_loaded,
        "stress_checkpoints": stress_predictor.checkpoint_paths,
        "cbt_model_loaded": cbt_predictor.is_loaded,
        "topic_model_loaded": topic_predictor.is_loaded,
    }

@app.post("/api/predict")
def predict(payload: PredictRequest):
    try:
        result = stress_predictor.predict(payload.text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    return {
        "label": result.label,
        "is_stressed": result.is_stressed,
        "confidence": result.confidence,
        "probabilities": result.probabilities,
    }


@app.post("/api/predict/distortion")
def predict_distortion(payload: PredictRequest):
    try:
        result = cbt_predictor.predict(payload.text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    return {
        "label": result.label,
        "has_distortion": result.has_distortion,
        "confidence": result.confidence,
        "probabilities": result.probabilities,
    }
