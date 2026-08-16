"""Bridge between the classifier's 5-label space and the forecaster's 8-label space.

The two models were trained on different corpora and their label sets only
partially overlap:

    classifier  : neutral, anger, fear, joy, sadness
    forecaster  : angry, anxious, calm, excited, happy, neutral, sad, stressed

The forecaster takes the speaker's *current* emotion as an auxiliary embedded
feature (`prev_emotion_id` in the training CSVs, which preprocess.py sets to
`cur_emotion_id` for the forecast task). To feed it the classifier's prediction
we have to project 5 -> 8. To then run the forecast through the deviation
tracker and the strategy rules, we project 8 -> 5 back.

Both projections are lossy and that is stated in the UI rather than hidden:
`fear` covers both `anxious` and `stressed`, and `joy` covers both `happy` and
`excited`.
"""

from typing import Dict, List, Optional

from config import EMOTION_LABELS, FORECAST_LABELS

# classifier label -> forecaster label
CLASSIFIER_TO_FORECAST: Dict[str, str] = {
    "neutral": "neutral",
    "anger": "angry",
    "fear": "anxious",     # the forecaster corpus has no "fear"; anxious is nearest
    "joy": "happy",
    "sadness": "sad",
}

# forecaster label -> classifier label
FORECAST_TO_CLASSIFIER: Dict[str, str] = {
    "angry": "anger",
    "anxious": "fear",
    "calm": "neutral",
    "excited": "joy",
    "happy": "joy",
    "neutral": "neutral",
    "sad": "sadness",
    "stressed": "fear",    # stress collapses onto fear in the 5-label space
}

# Pairs where the projection loses information, surfaced in the UI.
LOSSY_PROJECTIONS = {
    ("fear", "anxious"),
    ("fear", "stressed"),
    ("joy", "happy"),
    ("joy", "excited"),
}

FORECAST_LABEL_TO_ID: Dict[str, int] = {
    label: index for index, label in enumerate(FORECAST_LABELS)
}
CLASSIFIER_LABEL_TO_ID: Dict[str, int] = {
    label: index for index, label in enumerate(EMOTION_LABELS)
}

# The aux embedding in the trained checkpoints has `n_emo + 1` rows; the extra
# final row is the "unknown / first turn" bucket (preprocess.py fills NaN with
# n_emo). Anything we cannot map goes there rather than to a wrong emotion.
UNKNOWN_AUX_ID = len(FORECAST_LABELS)


def classifier_to_forecast_label(label: Optional[str]) -> Optional[str]:
    """Project a classifier label into the forecaster's vocabulary."""
    if label is None:
        return None
    if label in FORECAST_LABEL_TO_ID:
        return label
    return CLASSIFIER_TO_FORECAST.get(label)


def forecast_to_classifier_label(label: Optional[str]) -> Optional[str]:
    """Project a forecaster label into the classifier's vocabulary."""
    if label is None:
        return None
    if label in CLASSIFIER_LABEL_TO_ID:
        return label
    return FORECAST_TO_CLASSIFIER.get(label)


def aux_emotion_id(label: Optional[str]) -> int:
    """Aux feature id the forecaster expects for a given current emotion.

    Returns the unknown bucket when the emotion is missing or unmappable, which
    is exactly how the first turn of a conversation was encoded at training time.
    """
    projected = classifier_to_forecast_label(label)
    if projected is None:
        return UNKNOWN_AUX_ID
    return FORECAST_LABEL_TO_ID.get(projected, UNKNOWN_AUX_ID)


def project_probabilities(forecast_probs: Dict[str, float]) -> Dict[str, float]:
    """Collapse an 8-label forecast distribution onto the 5 classifier labels.

    Probability mass is summed, not averaged: P(fear) = P(anxious) + P(stressed).
    """
    collapsed = {label: 0.0 for label in EMOTION_LABELS}
    for label, prob in forecast_probs.items():
        target = forecast_to_classifier_label(label)
        if target in collapsed:
            collapsed[target] += float(prob)
    return collapsed


def is_lossy(classifier_label: Optional[str], forecast_label: Optional[str]) -> bool:
    """True when the 8 -> 5 projection merged distinguishable states."""
    return (classifier_label, forecast_label) in LOSSY_PROJECTIONS


def describe_mapping(forecast_label: str) -> str:
    """Human-readable note for the trace/UI."""
    projected = forecast_to_classifier_label(forecast_label)
    if projected == forecast_label:
        return f"'{forecast_label}' exists in both label spaces."
    siblings: List[str] = sorted(
        source
        for source, target in FORECAST_TO_CLASSIFIER.items()
        if target == projected
    )
    if len(siblings) > 1:
        return (
            f"Forecaster label '{forecast_label}' maps to classifier label "
            f"'{projected}', which also covers {', '.join(s for s in siblings if s != forecast_label)}."
        )
    return f"Forecaster label '{forecast_label}' maps to classifier label '{projected}'."
