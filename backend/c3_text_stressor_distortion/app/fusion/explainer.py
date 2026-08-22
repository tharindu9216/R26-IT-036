"""Compose component token evidence with a deterministic fusion rule trace."""

from __future__ import annotations

import re
from typing import Iterable

from .decision import FusionDecision, FusionState


DISCLAIMER = (
    "The SHAP, LIME, and Integrated Gradients consensus describes model "
    "behaviour, not psychological facts or a diagnosis. The combined state "
    "is produced by a transparent rule."
)
CONSENSUS_METHODS = ["Integrated Gradients", "SHAP", "LIME"]


def _normalise_span(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip(" .,!?;:")


def _serialise_evidence(
    rationales: Iterable[dict] | None,
    source: str,
    model_name: str,
) -> list[dict]:
    evidence = []
    for item in rationales or []:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        evidence.append({
            "text": text,
            "score": float(item.get("score", 0.0)),
            "source": source,
            "model": model_name,
        })
    return evidence


def _serialise_features(
    ranked_features: Iterable[dict] | None,
    source: str,
    model_name: str,
    limit: int = 15,
) -> list[dict]:
    features = []
    for item in ranked_features or []:
        word = str(item.get("word", "")).strip()
        if not word:
            continue
        features.append({
            "word": word,
            "integrated_gradients": float(
                item.get("integrated_gradients", 0.0)
            ),
            "shap": float(item.get("shap", 0.0)),
            "lime": float(item.get("lime", 0.0)),
            "consensus": float(item.get("consensus", 0.0)),
            "agreement": float(item.get("agreement", 0.0)),
            "source": source,
            "model": model_name,
        })
        if len(features) >= limit:
            break
    return features


def rule_trace(decision: FusionDecision) -> str:
    traces = {
        FusionState.STRESS_WITH_DISTORTION: (
            "The Stress head detected a stress signal and the CBT head detected "
            "a cognitive-distortion signal, so the combined state is Stress "
            "with thinking-pattern signal."
        ),
        FusionState.STRESS_ONLY: (
            "The Stress head detected a stress signal and the CBT head did not "
            "detect a cognitive-distortion signal, so the combined state is "
            "Stress signal only."
        ),
        FusionState.DISTORTION_ONLY: (
            "The CBT head detected a cognitive-distortion signal and the Stress "
            "head did not detect a stress signal, so the combined state is "
            "Thinking-pattern signal only."
        ),
        FusionState.NO_SIGNAL: (
            "Neither component head detected its corresponding signal, so the "
            "combined state is No combined signal."
        ),
    }
    return traces[decision.state]


def compose_explanation(
    decision: FusionDecision,
    stress_rationales: Iterable[dict] | None = None,
    cbt_rationales: Iterable[dict] | None = None,
    stress_features: Iterable[dict] | None = None,
    cbt_features: Iterable[dict] | None = None,
    stress_lime_r2: float | None = None,
    cbt_lime_r2: float | None = None,
    stress_model: str = "DeBERTa-v3",
    cbt_model: str = "DeBERTa-v3",
    errors: Iterable[str] | None = None,
) -> dict:
    stress_evidence = _serialise_evidence(
        stress_rationales, "stress", stress_model
    )
    cbt_evidence = _serialise_evidence(
        cbt_rationales, "cbt", cbt_model
    )
    stress_feature_evidence = _serialise_features(
        stress_features, "stress", stress_model
    )
    cbt_feature_evidence = _serialise_features(
        cbt_features, "cbt", cbt_model
    )
    stress_keys = {_normalise_span(item["text"]) for item in stress_evidence}
    overlap = [
        item
        for item in cbt_evidence
        if _normalise_span(item["text"]) in stress_keys
    ]
    error_list = [str(error) for error in errors or [] if str(error).strip()]
    evidence_count = int(bool(stress_evidence)) + int(bool(cbt_evidence))
    if not error_list:
        status = "complete"
    elif evidence_count:
        status = "partial"
    else:
        status = "rule_only"

    return {
        "fusion": decision.as_dict(),
        "method": "hierarchical_rule_trace_with_ig_shap_lime_consensus",
        "methods_used": CONSENSUS_METHODS,
        "xai_status": status,
        "stress_evidence": stress_evidence,
        "cbt_evidence": cbt_evidence,
        "stress_feature_evidence": stress_feature_evidence,
        "cbt_feature_evidence": cbt_feature_evidence,
        "stress_lime_r2": (
            float(stress_lime_r2) if stress_lime_r2 is not None else None
        ),
        "cbt_lime_r2": (
            float(cbt_lime_r2) if cbt_lime_r2 is not None else None
        ),
        "overlap_evidence": overlap,
        "fusion_reason": rule_trace(decision),
        "errors": error_list,
        "disclaimer": DISCLAIMER,
    }
