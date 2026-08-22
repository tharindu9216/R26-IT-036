"""On-demand component XAI composed into a fusion explanation.

SHAP, LIME, and Integrated Gradients are aligned and averaged for one
representative DeBERTa-v3 member per head. This reuses the same consensus
implementation as the Streamlit XAI panels without implying that one
attribution map exactly explains every member of both deployed ensembles.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from threading import Lock

import torch.nn as nn

from ..config import PROJECT_ROOT
from .decision import decide_fusion
from .explainer import compose_explanation


logger = logging.getLogger(__name__)
STRESS_XAI_DIR = (
    PROJECT_ROOT
    / "ai_components"
    / "c3_text_stressor_distortion"
    / "Stress header"
    / "xai"
)
DEFAULT_XAI_MEMBER = "DeBERTa-v3"
DEFAULT_IG_STEPS = 50
DEFAULT_LIME_SAMPLES = 1000


class _DualOutputAdapter(nn.Module):
    """Adapt the CBT binary model to the dual-output StressIG interface."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @property
    def encoder(self):
        return self.model.encoder

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask), None


def _stress_combined_xai_class():
    module_name = "backend_fusion_combined_xai"
    if module_name in sys.modules:
        return sys.modules[module_name].StressCombinedXAI
    module_path = STRESS_XAI_DIR / "combined_explainer.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Missing combined XAI module: {module_path}")
    if str(STRESS_XAI_DIR) not in sys.path:
        sys.path.insert(0, str(STRESS_XAI_DIR))
    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load the combined XAI module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.StressCombinedXAI


class FusionExplanationService:
    def __init__(self, stress_predictor, cbt_predictor):
        self._stress_predictor = stress_predictor
        self._cbt_predictor = cbt_predictor
        self._lock = Lock()

    @staticmethod
    def _explain(
        model,
        tokenizer,
        labels: dict[str, str],
        text: str,
        target_class: int,
        device,
        max_length: int,
        decision_threshold: float,
    ) -> dict:
        StressCombinedXAI = _stress_combined_xai_class()
        explainer = StressCombinedXAI(
            model,
            tokenizer,
            labels,
            device=str(device),
            max_length=max_length,
            decision_threshold=decision_threshold,
        )
        return explainer.explain(
            text,
            target_class=target_class,
            lime_samples=DEFAULT_LIME_SAMPLES,
            ig_steps=DEFAULT_IG_STEPS,
        )

    def explain(self, text: str, stress_result, cbt_result) -> dict:
        decision = decide_fusion(
            bool(stress_result.is_stressed),
            bool(cbt_result.has_distortion),
        )
        stress_rationales = []
        cbt_rationales = []
        stress_features = []
        cbt_features = []
        stress_lime_r2 = None
        cbt_lime_r2 = None
        errors: list[str] = []

        # Captum performs gradient work against shared singleton models. Keep
        # explanation requests serial so concurrent API calls cannot interfere.
        with self._lock:
            try:
                resources = self._stress_predictor.xai_resources(
                    DEFAULT_XAI_MEMBER
                )
                result = self._explain(
                    resources.model,
                    resources.tokenizer,
                    {"0": "Not Stressed", "1": "Stressed"},
                    text,
                    int(bool(stress_result.is_stressed)),
                    resources.device,
                    resources.max_len,
                    resources.decision_threshold,
                )
                stress_rationales = result.get("rationales", [])
                stress_features = result.get("ranked_features", [])
                stress_lime_r2 = result.get("lime_result", {}).get("local_r2")
            except Exception:  # Rule trace remains useful if XAI fails.
                logger.warning("Stress fusion XAI failed", exc_info=True)
                errors.append("Stress token evidence is temporarily unavailable.")

            try:
                resources = self._cbt_predictor.xai_resources(DEFAULT_XAI_MEMBER)
                result = self._explain(
                    _DualOutputAdapter(resources.model),
                    resources.tokenizer,
                    {"0": "No Distortion", "1": "Distortion"},
                    text,
                    int(bool(cbt_result.has_distortion)),
                    resources.device,
                    resources.max_len,
                    resources.decision_threshold,
                )
                cbt_rationales = result.get("rationales", [])
                cbt_features = result.get("ranked_features", [])
                cbt_lime_r2 = result.get("lime_result", {}).get("local_r2")
            except Exception:  # Rule trace remains useful if XAI fails.
                logger.warning("CBT fusion XAI failed", exc_info=True)
                errors.append("CBT token evidence is temporarily unavailable.")

        return compose_explanation(
            decision,
            stress_rationales=stress_rationales,
            cbt_rationales=cbt_rationales,
            stress_features=stress_features,
            cbt_features=cbt_features,
            stress_lime_r2=stress_lime_r2,
            cbt_lime_r2=cbt_lime_r2,
            stress_model=DEFAULT_XAI_MEMBER,
            cbt_model=DEFAULT_XAI_MEMBER,
            errors=errors,
        )
