"""Decision-level (late) fusion of the C3 Stress and CBT headers.

Each header is already its own independent ensemble decision (see
``c3/stress`` and ``c3/CBT``). Rather than feeding the two headers into
routing/reply generation separately, this module fuses their probabilities
into one weighted decision -- the same "weighted probability average +
threshold" shape the ensembles themselves already use one level down, just
applied across headers instead of across model members.

C1's sensor stress is a different modality (physiological, not text) and is
not part of this fusion -- it stays an independent signal all the way through
routing and reply generation.

``InfluenceWeights``/``weighted_influence`` at the bottom of this module sit
one level above that and do something different: they do not combine the
modalities into a decision at all. They only say how much each one should sway
the *wording* of a reply, because a wristband window and a voice-appraisal
frame are not as trustworthy as the sentence the user actually wrote. Nothing
is gated or thresholded by them -- see ``reply_generator._signal_influence``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FusedTextSignal:
    is_flagged: bool
    probability: float
    stress_probability: float
    cbt_probability: float
    method: str


def fuse_stress_and_cbt(
    stress_probability: float,
    cbt_probability: float,
    *,
    stress_weight: float = 0.5,
    cbt_weight: float = 0.5,
    threshold: float = 0.5,
    method: str = "weighted_average",
    stress_flagged: bool | None = None,
    cbt_flagged: bool | None = None,
    stress_threshold: float = 0.5,
    cbt_threshold: float = 0.37,
) -> FusedTextSignal:
    """Apply one configurable late-fusion ablation policy.

    ``weighted_average`` is the selected deployment policy. ``or``,
    ``stress_only`` and ``cbt_only`` are explicit ablations. When the caller
    already has each header's validated decision, it should pass those flags;
    otherwise the per-header thresholds are used.
    """

    supported = {"weighted_average", "or", "stress_only", "cbt_only"}
    if method not in supported:
        choices = ", ".join(sorted(supported))
        raise ValueError(f"Unknown fusion method {method!r}; choose from: {choices}")
    total_weight = stress_weight + cbt_weight
    if total_weight <= 0:
        raise ValueError("Fusion weights must sum to a positive value")
    for name, value in (
        ("fusion", threshold),
        ("stress", stress_threshold),
        ("CBT", cbt_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} threshold must be between 0 and 1")

    weighted_probability = (
        stress_weight * stress_probability + cbt_weight * cbt_probability
    ) / total_weight
    stress_decision = (
        bool(stress_flagged)
        if stress_flagged is not None
        else stress_probability >= stress_threshold
    )
    cbt_decision = (
        bool(cbt_flagged)
        if cbt_flagged is not None
        else cbt_probability >= cbt_threshold
    )

    if method == "weighted_average":
        probability = weighted_probability
        is_flagged = probability >= threshold
    elif method == "or":
        probability = max(stress_probability, cbt_probability)
        is_flagged = stress_decision or cbt_decision
    elif method == "stress_only":
        probability = stress_probability
        is_flagged = stress_decision
    else:  # cbt_only
        probability = cbt_probability
        is_flagged = cbt_decision

    return FusedTextSignal(
        is_flagged=is_flagged,
        probability=probability,
        stress_probability=stress_probability,
        cbt_probability=cbt_probability,
        method=method,
    )


# --------------------------------------------------------------------------
# How much each modality is allowed to sway the reply's wording
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InfluenceWeights:
    """Trust multipliers for the three stress modalities, in [0, 1].

    ``1.0`` marks primary evidence; a lower value marks a source the model
    should treat as a weaker hint that may be wrong. These are deliberately
    not a gate: a low weight never switches a signal off, it only reaches the
    prompt as weaker evidence, so there is no threshold to tune. ``0.0`` is
    the one exception and drops the source from the prompt entirely.
    """

    text: float = 1.0
    sensor: float = 1.0
    voice: float = 1.0

    def __post_init__(self) -> None:
        for name in ("text", "sensor", "voice"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} influence weight must be between 0 and 1, got {value!r}"
                )


DEFAULT_INFLUENCE_WEIGHTS = InfluenceWeights()


def weighted_influence(
    weight: float,
    *,
    flagged: bool,
    probability: float | None,
) -> float | None:
    """Scale one modality's own confidence by how much it is trusted.

    ``None`` means the source should not be shown to the model at all: either
    it is switched off (weight 0) or it did not run this turn -- C3 in voice
    mode, C2 in text mode, C1 when no sensor is connected. A modality that ran
    and found nothing still reports, at a low influence, because "the wristband
    looked and saw calm" is not the same as "nobody looked".
    """

    if weight <= 0.0:
        return None
    if probability is None and not flagged:
        return None
    confidence = float(probability) if probability is not None else float(flagged)
    return weight * min(1.0, max(0.0, confidence))
