"""On-demand XAI for the two deployed C3 probability ensembles.

Every transformer member is explained independently with SHAP, LIME, and
Integrated Gradients.  Word-level attributions are normalized per member,
aligned by occurrence, and combined with the exact weights used by the
deployed soft-voting ensemble.
"""

from __future__ import annotations

import importlib.util
import logging
import re
import sys
from collections import defaultdict, deque
from pathlib import Path
from threading import Lock

import numpy as np
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
DEFAULT_IG_STEPS = 50
DEFAULT_LIME_SAMPLES = 1000
ATTRIBUTION_METHODS = ("integrated_gradients", "shap", "lime")


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

    @staticmethod
    def _word_key(word: str) -> str:
        return re.sub(r"^\W+|\W+$", "", word.casefold())

    @staticmethod
    def _normalise(values: list[float]) -> list[float]:
        array = np.asarray(values, dtype=float)
        if not array.size:
            return []
        scale = float(np.max(np.abs(array)))
        if scale <= 1e-12:
            return [0.0] * len(array)
        return (array / scale).tolist()

    @classmethod
    def _align_method(
        cls,
        canonical_words: list[str],
        feature_rows: list[dict],
        method: str,
    ) -> list[float]:
        available: dict[str, deque[float]] = defaultdict(deque)
        raw_scores = [float(row.get(method, 0.0)) for row in feature_rows]
        normalized = cls._normalise(raw_scores)
        for row, score in zip(feature_rows, normalized, strict=True):
            key = cls._word_key(str(row.get("word", "")))
            if key:
                available[key].append(score)
        aligned = []
        for word in canonical_words:
            key = cls._word_key(word)
            aligned.append(available[key].popleft() if available[key] else 0.0)
        return aligned

    @classmethod
    def _aggregate_member_results(
        cls,
        member_results: list[tuple[object, dict]],
    ) -> dict:
        """Combine normalized member explanations with deployment weights."""
        if not member_results:
            raise ValueError("At least one member explanation is required")

        canonical_rows = member_results[0][1].get("feature_rows", [])
        canonical_words = [str(row.get("word", "")) for row in canonical_rows]
        if not canonical_words:
            raise ValueError("Member XAI returned no word-level features")

        available_weight = sum(float(resource.weight) for resource, _ in member_results)
        if available_weight <= 0:
            raise ValueError("Available ensemble XAI weights must be positive")

        aggregated = {
            method: np.zeros(len(canonical_words), dtype=float)
            for method in ATTRIBUTION_METHODS
        }
        member_weights = {}
        lime_r2 = 0.0
        lime_r2_weight = 0.0
        for resource, result in member_results:
            normalized_weight = float(resource.weight) / available_weight
            member_weights[str(resource.name)] = float(resource.weight)
            rows = list(result.get("feature_rows", []))
            for method in ATTRIBUTION_METHODS:
                aggregated[method] += normalized_weight * np.asarray(
                    cls._align_method(canonical_words, rows, method),
                    dtype=float,
                )
            local_r2 = result.get("lime_result", {}).get("local_r2")
            if local_r2 is not None:
                lime_r2 += normalized_weight * float(local_r2)
                lime_r2_weight += normalized_weight

        feature_rows = []
        consensus_scores = []
        for index, word in enumerate(canonical_words):
            method_scores = [
                float(aggregated[method][index])
                for method in ATTRIBUTION_METHODS
            ]
            consensus = float(np.mean(method_scores))
            consensus_scores.append(consensus)
            nonzero = [score for score in method_scores if abs(score) > 1e-9]
            if nonzero and abs(consensus) > 1e-9:
                direction = 1 if consensus > 0 else -1
                agreement = sum(
                    1 for score in nonzero
                    if (1 if score > 0 else -1) == direction
                ) / len(nonzero)
            else:
                agreement = 0.0
            feature_rows.append({
                "word": word,
                "integrated_gradients": method_scores[0],
                "shap": method_scores[1],
                "lime": method_scores[2],
                "consensus": consensus,
                "agreement": agreement,
            })

        ranked_features = sorted(
            feature_rows,
            key=lambda row: abs(float(row["consensus"])),
            reverse=True,
        )
        # Keep rationales auditable and tokenizer-independent: the strongest
        # aggregate words are emitted directly with their signed scores.
        rationales = [
            {"text": row["word"], "score": float(row["consensus"])}
            for row in ranked_features[:8]
            if row["word"] and abs(float(row["consensus"])) > 1e-9
        ]
        return {
            "rationales": rationales,
            "ranked_features": ranked_features,
            "lime_r2": (
                lime_r2 / lime_r2_weight if lime_r2_weight > 0 else None
            ),
            "member_weights": member_weights,
            "model_label": "weighted_ensemble",
        }

    def _explain_ensemble(
        self,
        predictor,
        labels: dict[str, str],
        text: str,
        target_class: int,
        *,
        adapt_cbt: bool,
        error_prefix: str,
        errors: list[str],
    ) -> dict:
        member_results = []
        expected_members = predictor.xai_ensemble_resources()
        for resources in expected_members:
            try:
                model = (
                    _DualOutputAdapter(resources.model)
                    if adapt_cbt
                    else resources.model
                )
                model_text = predictor.preprocess_for_member(
                    text,
                    resources.name,
                )
                result = self._explain(
                    model,
                    resources.tokenizer,
                    labels,
                    model_text,
                    target_class,
                    resources.device,
                    resources.max_len,
                    resources.decision_threshold,
                )
                member_results.append((resources, result))
            except Exception:
                logger.warning(
                    "%s ensemble XAI failed for %s",
                    error_prefix,
                    resources.name,
                    exc_info=True,
                )
                errors.append(
                    f"{error_prefix} token evidence is unavailable for "
                    f"{resources.name}."
                )
        return self._aggregate_member_results(member_results)

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
                result = self._explain_ensemble(
                    self._stress_predictor,
                    {"0": "Not Stressed", "1": "Stressed"},
                    text,
                    int(bool(stress_result.is_stressed)),
                    adapt_cbt=False,
                    error_prefix="Stress",
                    errors=errors,
                )
                stress_rationales = result.get("rationales", [])
                stress_features = result.get("ranked_features", [])
                stress_lime_r2 = result.get("lime_r2")
                stress_members = result.get("member_weights", {})
            except Exception:  # Rule trace remains useful if XAI fails.
                logger.warning("Stress fusion XAI failed", exc_info=True)
                errors.append("Stress token evidence is temporarily unavailable.")
                stress_members = {}

            try:
                result = self._explain_ensemble(
                    self._cbt_predictor,
                    {"0": "No Distortion", "1": "Distortion"},
                    text,
                    int(bool(cbt_result.has_distortion)),
                    adapt_cbt=True,
                    error_prefix="CBT",
                    errors=errors,
                )
                cbt_rationales = result.get("rationales", [])
                cbt_features = result.get("ranked_features", [])
                cbt_lime_r2 = result.get("lime_r2")
                cbt_members = result.get("member_weights", {})
            except Exception:  # Rule trace remains useful if XAI fails.
                logger.warning("CBT fusion XAI failed", exc_info=True)
                errors.append("CBT token evidence is temporarily unavailable.")
                cbt_members = {}

        return compose_explanation(
            decision,
            stress_rationales=stress_rationales,
            cbt_rationales=cbt_rationales,
            stress_features=stress_features,
            cbt_features=cbt_features,
            stress_lime_r2=stress_lime_r2,
            cbt_lime_r2=cbt_lime_r2,
            stress_model="weighted_ensemble",
            cbt_model="weighted_ensemble",
            stress_ensemble_members=stress_members,
            cbt_ensemble_members=cbt_members,
            errors=errors,
        )
