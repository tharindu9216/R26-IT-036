"""API schemas for explainable decision fusion."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


FusionStateName = Literal[
    "no_signal",
    "stress_only",
    "distortion_only",
    "stress_with_distortion",
]


class FusionPredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)


class HeadPredictionResponse(BaseModel):
    label: str
    detected: bool
    confidence: float
    probabilities: dict[str, float]


class FusionDecisionResponse(BaseModel):
    state: FusionStateName
    label: str
    summary: str
    stress_detected: bool
    distortion_detected: bool
    method: str


class FusionAnalysisResponse(BaseModel):
    stress: HeadPredictionResponse
    distortion: HeadPredictionResponse
    fusion: FusionDecisionResponse


class TokenEvidenceResponse(BaseModel):
    text: str
    score: float
    source: Literal["stress", "cbt"]
    model: str


class CombinedFeatureEvidenceResponse(BaseModel):
    word: str
    integrated_gradients: float
    shap: float
    lime: float
    consensus: float
    agreement: float
    source: Literal["stress", "cbt"]
    model: str


class FusionExplanationResponse(BaseModel):
    fusion: FusionDecisionResponse
    method: str
    methods_used: list[Literal["Integrated Gradients", "SHAP", "LIME"]]
    xai_status: Literal["complete", "partial", "rule_only"]
    stress_evidence: list[TokenEvidenceResponse]
    cbt_evidence: list[TokenEvidenceResponse]
    stress_feature_evidence: list[CombinedFeatureEvidenceResponse]
    cbt_feature_evidence: list[CombinedFeatureEvidenceResponse]
    stress_lime_r2: float | None
    cbt_lime_r2: float | None
    overlap_evidence: list[TokenEvidenceResponse]
    fusion_reason: str
    errors: list[str]
    disclaimer: str
