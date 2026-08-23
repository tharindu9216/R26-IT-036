"""Orchestrate the independent heads and their transparent joint decision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .decision import FusionDecision, decide_fusion


@dataclass(frozen=True)
class FusionAnalysis:
    stress: Any
    distortion: Any
    fusion: FusionDecision

    def as_dict(self) -> dict:
        return {
            "stress": {
                "label": self.stress.label,
                "detected": bool(self.stress.is_stressed),
                "confidence": float(self.stress.confidence),
                "probabilities": self.stress.probabilities,
            },
            "distortion": {
                "label": self.distortion.label,
                "detected": bool(self.distortion.has_distortion),
                "confidence": float(self.distortion.confidence),
                "probabilities": self.distortion.probabilities,
            },
            "fusion": self.fusion.as_dict(),
        }


class FusionService:
    def __init__(self, stress_predictor, cbt_predictor):
        self._stress_predictor = stress_predictor
        self._cbt_predictor = cbt_predictor

    def analyze(self, text: str) -> FusionAnalysis:
        stress = self._stress_predictor.predict(text)
        distortion = self._cbt_predictor.predict(text)
        decision = decide_fusion(
            stress.is_stressed,
            distortion.has_distortion,
        )
        return FusionAnalysis(stress, distortion, decision)
