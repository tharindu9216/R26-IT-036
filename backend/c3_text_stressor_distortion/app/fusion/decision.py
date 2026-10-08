"""Transparent four-state decision fusion.

This module intentionally contains no learned parameters. It preserves the
two component predictions and names their joint state for the diary UI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class FusionState(str, Enum):
    NO_SIGNAL = "no_signal"
    STRESS_ONLY = "stress_only"
    DISTORTION_ONLY = "distortion_only"
    STRESS_WITH_DISTORTION = "stress_with_distortion"


@dataclass(frozen=True)
class FusionDecision:
    state: FusionState
    label: str
    summary: str
    stress_detected: bool
    distortion_detected: bool
    method: str = "decision_level_fusion"

    def as_dict(self) -> dict:
        result = asdict(self)
        result["state"] = self.state.value
        return result


def decide_fusion(
    is_stressed: bool,
    has_distortion: bool,
) -> FusionDecision:
    """Map the two independent binary decisions to one transparent state."""
    if is_stressed and has_distortion:
        return FusionDecision(
            state=FusionState.STRESS_WITH_DISTORTION,
            label="Stress with thinking-pattern signal",
            summary=(
                "The entry contains both a stress signal and a possible "
                "cognitive-distortion signal."
            ),
            stress_detected=True,
            distortion_detected=True,
        )
    if is_stressed:
        return FusionDecision(
            state=FusionState.STRESS_ONLY,
            label="Stress signal only",
            summary=(
                "The entry contains a stress signal without a detected "
                "cognitive-distortion signal."
            ),
            stress_detected=True,
            distortion_detected=False,
        )
    if has_distortion:
        return FusionDecision(
            state=FusionState.DISTORTION_ONLY,
            label="Thinking-pattern signal only",
            summary=(
                "The entry contains a possible cognitive-distortion signal "
                "without a detected stress signal."
            ),
            stress_detected=False,
            distortion_detected=True,
        )
    return FusionDecision(
        state=FusionState.NO_SIGNAL,
        label="No combined signal",
        summary=(
            "Neither the Stress head nor the CBT head detected its "
            "corresponding signal in this entry."
        ),
        stress_detected=False,
        distortion_detected=False,
    )
