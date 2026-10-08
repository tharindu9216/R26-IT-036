"""FastAPI routes for combined prediction and on-demand fusion XAI."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi import APIRouter, HTTPException

from .. import db
from ..predictors import fusion_explanation_service, fusion_service
from .schemas import (
    FusionAnalysisResponse,
    FusionExplanationResponse,
    FusionPredictRequest,
)


router = APIRouter(prefix="/api", tags=["fusion"])


@router.post("/fusion/predict", response_model=FusionAnalysisResponse)
def predict_fusion(payload: FusionPredictRequest):
    try:
        return fusion_service.analyze(payload.text).as_dict()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Combined prediction failed: {exc}",
        ) from exc


@router.post(
    "/diary/{entry_id}/explain",
    response_model=FusionExplanationResponse,
)
def explain_diary_entry(entry_id: int):
    entry = db.get_entry(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Diary entry not found")

    stress_result = SimpleNamespace(
        is_stressed=entry["stress"]["is_stressed"],
    )
    cbt_result = SimpleNamespace(
        has_distortion=entry["distortion"]["has_distortion"],
    )
    return fusion_explanation_service.explain(
        entry["content"],
        stress_result,
        cbt_result,
    )
