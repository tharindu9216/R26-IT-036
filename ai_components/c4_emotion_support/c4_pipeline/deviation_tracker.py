from typing import Dict, Optional

from config import (
    HIGH_DEVIATION,
    LOW_DEVIATION,
    MODERATE_DEVIATION,
    NEGATIVE,
    NEUTRAL,
    OTHER,
    POSITIVE,
    SAME_EMOTION,
)


def compute_deviation(previous_emotion: Optional[str], current_emotion: str) -> Dict[str, object]:
    if previous_emotion is None:
        return {
            "previous_emotion": None,
            "current_emotion": current_emotion,
            "deviation_score": SAME_EMOTION,
            "deviation_level": "None",
        }

    if previous_emotion == current_emotion:
        return {
            "previous_emotion": previous_emotion,
            "current_emotion": current_emotion,
            "deviation_score": SAME_EMOTION,
            "deviation_level": "None",
        }

    prev_group = _emotion_group(previous_emotion)
    curr_group = _emotion_group(current_emotion)

    if prev_group == NEUTRAL or curr_group == NEUTRAL:
        score = LOW_DEVIATION
        level = "Low"
    elif prev_group == NEGATIVE and curr_group == NEGATIVE:
        score = MODERATE_DEVIATION
        level = "Moderate"
    elif (prev_group == POSITIVE and curr_group == NEGATIVE) or (
        prev_group == NEGATIVE and curr_group == POSITIVE
    ):
        score = HIGH_DEVIATION
        level = "High"
    else:
        score = MODERATE_DEVIATION
        level = "Moderate"

    return {
        "previous_emotion": previous_emotion,
        "current_emotion": current_emotion,
        "deviation_score": score,
        "deviation_level": level,
    }


def _emotion_group(emotion: str):
    if emotion in POSITIVE:
        return POSITIVE
    if emotion in NEGATIVE:
        return NEGATIVE
    if emotion in NEUTRAL:
        return NEUTRAL
    return OTHER
