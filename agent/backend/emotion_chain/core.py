"""Dependency-free labels, constraints, tokenization and reply routing."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence


CURRENT_EMOTIONS = ("neutral", "anger", "fear", "joy", "sadness")

STATE_LABELS = (
    "anger",
    "fear",
    "high_anger",
    "high_fear",
    "high_joy",
    "high_sadness",
    "joy",
    "low_anger",
    "low_fear",
    "low_joy",
    "low_sadness",
    "neutral",
    "sadness",
)

STATE_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "neutral": ("neutral", "joy", "sadness", "anger", "fear"),
    "anger": ("neutral", "low_anger", "high_anger"),
    "fear": ("neutral", "low_fear", "high_fear"),
    "joy": ("neutral", "low_joy", "high_joy"),
    "sadness": ("neutral", "low_sadness", "high_sadness"),
}

NEGATIVE_EMOTION_FAMILIES = {"anger", "fear", "sadness"}

INTENSITY_LABELS = ("low", "high")

# The state each (current emotion, forecast intensity) pair names. The
# forecaster predicts the intensity of the *next* turn, not its emotion, so the
# family is carried over from the current emotion -- which is exactly the
# constraint ``STATE_TRANSITIONS`` already encodes: from ``sadness`` the only
# reachable states are ``neutral``, ``low_sadness`` and ``high_sadness``, and
# the forecaster chooses between the last two.
#
# ``neutral`` is the one current emotion with no low/high state to name: the
# 13-label vocabulary has no ``low_neutral``/``high_neutral``, and an intensity
# alone cannot say which family a neutral turn moves into. Those turns keep
# ``neutral`` as the state and carry the intensity in its own field.
INTENSITY_STATES: dict[str, tuple[str, str]] = {
    "anger": ("low_anger", "high_anger"),
    "fear": ("low_fear", "high_fear"),
    "joy": ("low_joy", "high_joy"),
    "sadness": ("low_sadness", "high_sadness"),
}

TOKEN_PATTERN = re.compile(
    r"[a-z0-9]+(?:'[a-z0-9]+)?|[^\w\s]",
    flags=re.IGNORECASE,
)


def normalize_current_emotion(label: str) -> str:
    normalized = label.strip().lower()
    if normalized not in CURRENT_EMOTIONS:
        expected = ", ".join(CURRENT_EMOTIONS)
        raise ValueError(f"Unknown current emotion {label!r}; expected: {expected}")
    return normalized


def emotion_family(label: str) -> str:
    """Collapse low/high state labels to their current-emotion family."""

    normalized = label.strip().lower()
    for prefix in ("low_", "high_"):
        if normalized.startswith(prefix):
            return normalized[len(prefix) :]
    return normalized


def tokenize_state_text(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text cannot be empty")
    return TOKEN_PATTERN.findall(text.lower())


def allowed_next_states(current_emotion: str) -> tuple[str, ...]:
    return STATE_TRANSITIONS[normalize_current_emotion(current_emotion)]


def constrain_probability_map(
    probabilities: Mapping[str, float],
    current_emotion: str,
) -> dict[str, float]:
    """Mask unreachable states and renormalize the remaining probabilities."""

    allowed = set(allowed_next_states(current_emotion))
    constrained = {
        label: max(0.0, float(probabilities.get(label, 0.0)))
        if label in allowed
        else 0.0
        for label in STATE_LABELS
    }
    total = sum(constrained.values())
    if total <= 0.0:
        fallback = 1.0 / len(allowed)
        return {
            label: fallback if label in allowed else 0.0 for label in STATE_LABELS
        }
    return {label: value / total for label, value in constrained.items()}


def normalize_intensity(label: str) -> str:
    normalized = label.strip().lower()
    if normalized not in INTENSITY_LABELS:
        expected = ", ".join(INTENSITY_LABELS)
        raise ValueError(f"Unknown intensity {label!r}; expected: {expected}")
    return normalized


def intensity_state(current_emotion: str, intensity: str) -> str:
    """Name the next state a (current emotion, forecast intensity) pair implies.

    The result is always one of ``allowed_next_states(current_emotion)``, so
    everything downstream still reads the same 13-label vocabulary the earlier
    forecaster produced.
    """

    current = normalize_current_emotion(current_emotion)
    level = normalize_intensity(intensity)
    states = INTENSITY_STATES.get(current)
    if states is None:  # neutral -- see INTENSITY_STATES.
        return current
    return states[INTENSITY_LABELS.index(level)]


def intensity_probability_map(
    current_emotion: str, probability_high: float
) -> dict[str, float]:
    """Spread one high-intensity probability over the reachable state labels.

    Every unreachable state stays at zero, exactly as
    ``constrain_probability_map`` leaves them, so the UI charts and the
    ``next_emotion_probabilities`` field keep their previous shape. A neutral
    current emotion collapses to a single state for the reason given on
    ``INTENSITY_STATES``.
    """

    if not 0.0 <= probability_high <= 1.0:
        raise ValueError("probability_high must be between 0 and 1")
    current = normalize_current_emotion(current_emotion)
    states = INTENSITY_STATES.get(current)
    if states is None:
        reachable = {current: 1.0}
    else:
        low_state, high_state = states
        reachable = {
            low_state: 1.0 - float(probability_high),
            high_state: float(probability_high),
        }
    return {label: reachable.get(label, 0.0) for label in STATE_LABELS}


def reply_route_reasons(
    current_emotion: str,
    next_emotion: str,
    *,
    sensor_stress: bool = False,
    text_signal_flagged: bool = False,
    voice_stress: bool = False,
    multimodal_stress: bool = False,
    next_emotion_confidence: float | None = None,
    next_negative_min_confidence: float = 0.0,
) -> tuple[str, ...]:
    """Name every condition asking for the supportive adapter over plain Qwen.

    Returns the reasons rather than the verdict so the choice can be explained
    -- the same shape as ``escalation_signals`` in ``support_contacts.py``, and
    for the same reason: "why did this turn get the supportive strategy" is a
    question worth being able to answer. Any one reason is enough; an empty
    tuple means the turn stays on the base model.

    ``text_signal_flagged`` is the text-mode C3 Stress/CBT late-fusion signal.
    ``voice_stress`` is the voice-mode C2 appraisal signal. They are mutually
    exclusive at the application boundary; C1 sensor stress is shared.
    """

    if not 0.0 <= next_negative_min_confidence <= 1.0:
        raise ValueError("next_negative_min_confidence must be between 0 and 1")
    if next_emotion_confidence is not None and not 0.0 <= next_emotion_confidence <= 1.0:
        raise ValueError("next_emotion_confidence must be between 0 and 1")

    current_family = emotion_family(normalize_current_emotion(current_emotion))
    next_family = emotion_family(next_emotion)
    next_is_confident = (
        next_emotion_confidence is None
        or next_emotion_confidence >= next_negative_min_confidence
    )

    reasons = []
    if current_family in NEGATIVE_EMOTION_FAMILIES:
        reasons.append("negative_current_emotion")
    if next_family in NEGATIVE_EMOTION_FAMILIES and next_is_confident:
        reasons.append("negative_forecast")
    if sensor_stress:
        reasons.append("sensor_stress")
    if text_signal_flagged:
        reasons.append("text_signal")
    if voice_stress:
        reasons.append("voice_stress")
    if multimodal_stress:
        reasons.append("multimodal_stress")
    return tuple(reasons)


def select_reply_route(
    current_emotion: str,
    next_emotion: str,
    *,
    sensor_stress: bool = False,
    text_signal_flagged: bool = False,
    voice_stress: bool = False,
    multimodal_stress: bool = False,
    next_emotion_confidence: float | None = None,
    next_negative_min_confidence: float = 0.0,
) -> str:
    """Choose between plain Qwen and the ESConv supportive LoRA adapter."""

    return (
        "supportive_adapter"
        if reply_route_reasons(
            current_emotion,
            next_emotion,
            sensor_stress=sensor_stress,
            text_signal_flagged=text_signal_flagged,
            voice_stress=voice_stress,
            multimodal_stress=multimodal_stress,
            next_emotion_confidence=next_emotion_confidence,
            next_negative_min_confidence=next_negative_min_confidence,
        )
        else "base"
    )


def validate_forecaster_metadata(metadata: Mapping[str, object]) -> None:
    """Reject a next-intensity artifact this code cannot read correctly.

    The costly mistake is a silent one: both branches take the current-emotion
    probabilities as a positional vector, so an artifact trained against a
    different label order would score every turn without raising anything. The
    order is therefore checked here rather than trusted.
    """

    task = str(metadata.get("task", ""))
    if task != "next_intensity":
        raise ValueError(f"Expected a next_intensity artifact, found task={task!r}")

    labels = [str(label).lower() for label in metadata.get("labels", ())]
    if tuple(labels) != INTENSITY_LABELS:
        raise ValueError(
            f"Forecaster labels {labels} do not match {list(INTENSITY_LABELS)}"
        )

    emotions = [str(label).lower() for label in metadata.get("current_emotions", ())]
    if tuple(emotions) != CURRENT_EMOTIONS:
        raise ValueError(
            "Forecaster emotion order does not match the current-emotion model: "
            f"{emotions} != {list(CURRENT_EMOTIONS)}"
        )

    blend = metadata.get("blend")
    if not isinstance(blend, Mapping):
        raise ValueError("Forecaster metadata has no blend section")
    weights = float(blend["baseline_weight"]) + float(blend["fusion_weight"])
    if abs(weights - 1.0) > 1e-6:
        raise ValueError(f"Blend weights must sum to 1, found {weights}")
    threshold = float(blend["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Blend threshold must be between 0 and 1, found {threshold}")


def validate_checkpoint_labels(
    checkpoint_label2id: Mapping[str, int],
    metadata_label2id: Mapping[str, int] | None = None,
) -> None:
    expected = {label: index for index, label in enumerate(STATE_LABELS)}
    actual = {str(label): int(index) for label, index in checkpoint_label2id.items()}
    if actual != expected:
        raise ValueError("Next-emotion checkpoint label order is incompatible")
    if metadata_label2id is not None:
        metadata = {str(label): int(index) for label, index in metadata_label2id.items()}
        if metadata != actual:
            raise ValueError("Checkpoint and meta_forecast.json labels do not match")
