"""Turn-to-turn emotional deviation, ported from the C4 deviation tracker.

C4 measures how far the user's emotion moved between two consecutive turns and
feeds the resulting level into its strategy selection. The agent needs the same
signal for two things: prompt context ("this turn is a sharp drop, not more of
the same") and the extreme-negative escalation rule in
``support_contacts.py``.

The scores and level names are deliberately identical to
``R26-IT-036/ai_components/c4_emotion_support/c4_pipeline/deviation_tracker.py``
so the two components can never disagree about the same pair of turns. The one
difference is the label space: C4 tracks deviation over its 5 classifier
labels, while the agent also has the 13 intensity states, so the valence groups
below cover both vocabularies and ``high_sadness`` groups with ``sadness``.

Deviation is a comparison between turns, not a claim about one message. With no
previous turn it is "None" -- the first message of a conversation has nothing to
deviate from, and that is not the same as a calm one.
"""

from __future__ import annotations

from dataclasses import dataclass


POSITIVE_STATES = frozenset({"joy", "low_joy", "high_joy"})
NEGATIVE_STATES = frozenset(
    {
        "sadness",
        "anger",
        "fear",
        "low_sadness",
        "high_sadness",
        "low_anger",
        "high_anger",
        "low_fear",
        "high_fear",
    }
)
NEUTRAL_STATES = frozenset({"neutral"})

SAME_EMOTION = 0.0
LOW_DEVIATION = 0.25
MODERATE_DEVIATION = 0.60
HIGH_DEVIATION = 0.90

DEVIATION_LEVELS = ("None", "Low", "Moderate", "High")


@dataclass(frozen=True)
class EmotionDeviation:
    """How far this turn's emotion moved from the previous turn's."""

    previous_emotion: str | None
    current_emotion: str
    score: float
    level: str

    @property
    def is_high(self) -> bool:
        return self.level == "High"

    def as_dict(self) -> dict[str, object]:
        return {
            "previous_emotion": self.previous_emotion,
            "current_emotion": self.current_emotion,
            "deviation_score": self.score,
            "deviation_level": self.level,
        }


def valence_group(emotion: str) -> str:
    """Collapse a classifier label or a 13-state label to its valence group."""

    normalized = emotion.strip().lower()
    if normalized in POSITIVE_STATES:
        return "positive"
    if normalized in NEGATIVE_STATES:
        return "negative"
    if normalized in NEUTRAL_STATES:
        return "neutral"
    return "other"


def compute_deviation(
    previous_emotion: str | None, current_emotion: str
) -> EmotionDeviation:
    """Score the move from ``previous_emotion`` to ``current_emotion``.

    Nothing to compare against, or no change at all, is "None". A move that
    involves neutral is "Low" (drifting into or out of neutral is ordinary).
    Staying inside the negative group while changing emotion -- sadness to
    anger -- is "Moderate". Crossing between positive and negative is "High":
    that is the reversal the strategy rules and the escalation check care
    about.
    """

    current = current_emotion.strip().lower()
    if previous_emotion is None:
        return EmotionDeviation(None, current, SAME_EMOTION, "None")

    previous = previous_emotion.strip().lower()
    if previous == current:
        return EmotionDeviation(previous, current, SAME_EMOTION, "None")

    previous_group = valence_group(previous)
    current_group = valence_group(current)

    if previous_group == "neutral" or current_group == "neutral":
        score, level = LOW_DEVIATION, "Low"
    elif previous_group == "negative" and current_group == "negative":
        score, level = MODERATE_DEVIATION, "Moderate"
    elif {previous_group, current_group} == {"positive", "negative"}:
        score, level = HIGH_DEVIATION, "High"
    else:
        score, level = MODERATE_DEVIATION, "Moderate"

    return EmotionDeviation(previous, current, score, level)
