"""Shared predictor singletons, imported by both main.py and diary.py so the
underlying transformer checkpoints and BERTopic resources are only ever
loaded once."""
from __future__ import annotations

from .cbt_model import CBTPredictor
from .fusion.service import FusionService
from .fusion.xai_service import FusionExplanationService
from .stress_model import StressPredictor
from .topic_model import TopicPredictor

stress_predictor = StressPredictor()
cbt_predictor = CBTPredictor()
topic_predictor = TopicPredictor()
fusion_service = FusionService(stress_predictor, cbt_predictor)
fusion_explanation_service = FusionExplanationService(
    stress_predictor,
    cbt_predictor,
)
