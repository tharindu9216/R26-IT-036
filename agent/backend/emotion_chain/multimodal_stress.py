"""Self-report scoring and late fusion with the wearable stress result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


QUESTION_DIRECTIONS = {
    "manageable": False,
    "tense": True,
    "overwhelmed": True,
    "relaxed": False,
    "worried": True,
    "in_control": False,
}


@dataclass(frozen=True)
class QuestionnaireResult:
    probability: float
    is_stressed: bool
    answers: dict[str, int]
    item_stress_scores: dict[str, float]


@dataclass(frozen=True)
class MultimodalStressResult:
    probability: float | None
    is_stressed: bool
    available: bool
    normalized_weights: dict[str, float]
    probabilities: dict[str, float]


def score_questionnaire(
    answers: Mapping[str, int], *, threshold: float = 0.5
) -> QuestionnaireResult:
    """Convert six 0-3 momentary check-in answers to P(stress)."""

    if set(answers) != set(QUESTION_DIRECTIONS):
        missing = sorted(set(QUESTION_DIRECTIONS) - set(answers))
        extra = sorted(set(answers) - set(QUESTION_DIRECTIONS))
        raise ValueError(f"Questionnaire answers do not match; missing={missing}, extra={extra}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Questionnaire threshold must be between 0 and 1")

    normalized_answers: dict[str, int] = {}
    item_scores: dict[str, float] = {}
    for key, stress_worded in QUESTION_DIRECTIONS.items():
        value = answers[key]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 3:
            raise ValueError(f"Answer {key!r} must be an integer from 0 to 3")
        normalized_answers[key] = value
        item_scores[key] = value / 3.0 if stress_worded else (3 - value) / 3.0

    probability = sum(item_scores.values()) / len(item_scores)
    return QuestionnaireResult(
        probability=probability,
        is_stressed=probability >= threshold,
        answers=normalized_answers,
        item_stress_scores=item_scores,
    )


def fuse_multimodal_stress(
    *,
    questionnaire: float | None = None,
    sensor: float | None = None,
    voice: float | None = None,
    questionnaire_weight: float = 0.6,
    sensor_weight: float = 0.4,
    voice_weight: float = 0.0,
    threshold: float = 0.5,
) -> MultimodalStressResult:
    """Weighted average over available inputs, renormalizing missing ones."""

    configured = {
        "questionnaire": questionnaire_weight,
        "sensor": sensor_weight,
        "voice": voice_weight,
    }
    provided = {
        key: float(value)
        for key, value in {
            "questionnaire": questionnaire,
            "sensor": sensor,
            "voice": voice,
        }.items()
        if value is not None
    }
    for key, value in configured.items():
        if value < 0:
            raise ValueError(f"{key} weight cannot be negative")
    for key, value in provided.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} probability must be between 0 and 1")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Multimodal threshold must be between 0 and 1")

    # A zero-weight source is kept as an independent signal elsewhere in the
    # graph, but it must not appear in this combined score or its XAI weights.
    probabilities = {
        key: value for key, value in provided.items() if configured[key] > 0
    }
    active_total = sum(configured[key] for key in probabilities)
    if not probabilities or active_total <= 0:
        return MultimodalStressResult(None, False, False, {}, probabilities)

    normalized = {key: configured[key] / active_total for key in probabilities}
    probability = sum(normalized[key] * value for key, value in probabilities.items())
    return MultimodalStressResult(
        probability=probability,
        is_stressed=probability >= threshold,
        available=True,
        normalized_weights=normalized,
        probabilities=probabilities,
    )
