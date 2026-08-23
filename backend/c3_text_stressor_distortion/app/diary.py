from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import db
from .predictors import fusion_service, topic_predictor

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/diary", tags=["diary"])


class DiaryEntryCreate(BaseModel):
    content: str = Field(..., min_length=1, max_length=8000)
    title: str = Field("", max_length=200)


@router.post("")
def create_entry(payload: DiaryEntryCreate):
    try:
        analysis = fusion_service.analyze(payload.content)
        stress_result = analysis.stress
        distortion_result = analysis.distortion
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    # Topic tagging is a best-effort enrichment, not a core signal like the
    # two ensembles above — a BERTopic/embedding hiccup (or a "weak match" on
    # short casual text) should never block saving the entry.
    topic_theme = topic_similarity = topic_terms = None
    try:
        topic_result = topic_predictor.predict(payload.content)
        if topic_result.is_clear:
            topic_theme = topic_result.theme
            topic_similarity = topic_result.similarity
            topic_terms = topic_result.terms
    except Exception:
        logger.warning("Topic prediction failed", exc_info=True)

    entry = db.insert_entry(
        content=payload.content,
        title=payload.title,
        stress_label=stress_result.label,
        stress_is_stressed=stress_result.is_stressed,
        stress_confidence=stress_result.confidence,
        stress_probabilities=stress_result.probabilities,
        distortion_label=distortion_result.label,
        distortion_has_distortion=distortion_result.has_distortion,
        distortion_confidence=distortion_result.confidence,
        distortion_probabilities=distortion_result.probabilities,
        topic_theme=topic_theme,
        topic_similarity=topic_similarity,
        topic_terms=topic_terms,
    )
    return entry


@router.get("")
def list_entries(limit: int = 50, offset: int = 0):
    return db.list_entries(limit=limit, offset=offset)


@router.get("/summary")
def get_summary():
    return db.summary()


@router.get("/{entry_id}")
def get_entry(entry_id: int):
    entry = db.get_entry(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Diary entry not found")
    return entry


@router.delete("/{entry_id}")
def delete_entry(entry_id: int):
    if not db.delete_entry(entry_id):
        raise HTTPException(status_code=404, detail="Diary entry not found")
    return {"deleted": True}
