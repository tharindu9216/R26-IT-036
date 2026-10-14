"""Bridge between the classifier's 5 emotions and the forecaster's 13 next-STATES.

What changed and why
--------------------
The previous forecaster predicted a bare next *emotion* in a private 8-label
vocabulary (`angry / anxious / calm / excited / happy / neutral / sad /
stressed`). That forced a lossy projection in both directions -- `fear` had to
choose between `anxious` and `stressed`, `joy` between `happy` and `excited` --
and, more importantly, it answered the wrong question. "The user will be sad
next turn" is nearly the same sentence as "the user is sad now"; emotion has
strong inertia, so a persistence baseline matched it.

The retrained forecaster predicts the next emotional *state*: the emotion
together with where its intensity is heading.

    neutral
    joy | sadness | anger | fear          onset  (only reachable from neutral)
    low_X | high_X                        the emotion persists, intensity moves

Two consequences for this module:

1. The **input** projection is gone. The forecaster conditions on the current
   emotion in the classifier's own five labels, so 5 -> 5 is the identity.
   `assert_label_spaces_aligned()` checks that at import rather than trusting it.

2. The **output** projection is no longer lossy. Every state decomposes exactly
   into (base emotion, intensity), and the base emotion is already a classifier
   label. `low_sadness -> sadness` discards the intensity, but the intensity is
   kept and reported alongside rather than silently dropped -- see
   `describe_state`. `is_lossy` therefore returns False for every state, and is
   kept only so callers written against the old contract keep working.

The trajectory is the part downstream stages actually gained: `high_sadness`
and `low_sadness` project onto the same emotion but call for opposite responses.
"""

from typing import Dict, List, Optional

from config import (
    EMOTION_LABELS,
    FORECAST_CURRENT_EMOTIONS,
    FORECAST_LABELS,
    FORECAST_TRANSITIONS,
)

# ----------------------------------------------------------------- id spaces
FORECAST_LABEL_TO_ID: Dict[str, int] = {
    label: index for index, label in enumerate(FORECAST_LABELS)
}
CLASSIFIER_LABEL_TO_ID: Dict[str, int] = {
    label: index for index, label in enumerate(EMOTION_LABELS)
}
CURRENT_EMOTION_TO_ID: Dict[str, int] = {
    label: index for index, label in enumerate(FORECAST_CURRENT_EMOTIONS)
}

# The aux embedding has `n_emo + 1` rows; the final row is the "unknown current
# emotion" bucket. Anything unmappable goes there rather than to a wrong emotion.
UNKNOWN_AUX_ID = len(FORECAST_CURRENT_EMOTIONS)

BASE_EMOTIONS = ["joy", "sadness", "anger", "fear"]
INTENSITIES = ["low", "high"]

# Trajectory vocabulary, ordered from best to worst for a support system.
# `persisting` is the out-of-distribution bucket -- see `trajectory`.
TRAJECTORIES = [
    "steady", "resolving", "easing", "persisting", "onset", "escalating",
]


def assert_label_spaces_aligned() -> None:
    """Fail loudly if the classifier and forecaster stop sharing a vocabulary.

    The whole point of the retrain is that they do. A silent mismatch would
    feed the forecaster an aux id meaning a different emotion than the
    classifier predicted, which produces confident nonsense rather than an error.
    """
    if set(EMOTION_LABELS) != set(FORECAST_CURRENT_EMOTIONS):
        raise ValueError(
            "classifier and forecaster current-emotion spaces diverged: "
            f"{sorted(EMOTION_LABELS)} vs {sorted(FORECAST_CURRENT_EMOTIONS)}"
        )
    unknown = set(FORECAST_LABELS) - set(_STATE_DECOMPOSITION)
    if unknown:
        raise ValueError(f"next-state labels with no decomposition: {sorted(unknown)}")


def _decompose_all() -> Dict[str, Dict[str, Optional[str]]]:
    table: Dict[str, Dict[str, Optional[str]]] = {
        "neutral": {"base": "neutral", "intensity": None}
    }
    for emotion in BASE_EMOTIONS:
        table[emotion] = {"base": emotion, "intensity": None}
        for intensity in INTENSITIES:
            table[f"{intensity}_{emotion}"] = {"base": emotion, "intensity": intensity}
    return table


_STATE_DECOMPOSITION = _decompose_all()

assert_label_spaces_aligned()


# ----------------------------------------------------------------- decomposition
def state_base_emotion(state: Optional[str]) -> Optional[str]:
    """The classifier-space emotion a next-state belongs to."""
    if state is None:
        return None
    entry = _STATE_DECOMPOSITION.get(state)
    return entry["base"] if entry else None


def state_intensity(state: Optional[str]) -> Optional[str]:
    """'low' / 'high', or None for neutral and for onset states."""
    if state is None:
        return None
    entry = _STATE_DECOMPOSITION.get(state)
    return entry["intensity"] if entry else None


def reachable_states(current_emotion: Optional[str]) -> List[str]:
    """Next-states the corpus transition rules allow from `current_emotion`.

    An unknown current emotion lifts the constraint entirely rather than
    guessing -- returning a wrong 3-state set would be worse than no mask.
    """
    if current_emotion is None:
        return list(FORECAST_LABELS)
    return list(FORECAST_TRANSITIONS.get(current_emotion, FORECAST_LABELS))


def trajectory(current_emotion: Optional[str], state: Optional[str]) -> str:
    """Where the emotion is heading, which is what the state buys over a label.

        steady       neutral now, neutral next
        onset        neutral now, an emotion appears
        resolving    an emotion now, neutral next
        easing       the emotion persists at low intensity
        escalating   the emotion persists at high intensity
        persisting   the emotion is still here, with no intensity given

    `persisting` covers pairs the corpus transition rules cannot produce -- most
    often a caller passing a bare emotion label (`sadness -> sadness`) where a
    state was expected. Calling that `onset` would be wrong, since the emotion
    is already present, and calling it `unknown` would hide a still-negative
    forecast from the distress rules. It is reported for what it is and treated
    on the cautious side by `is_deteriorating`.
    """
    if state is None or current_emotion is None:
        return "unknown"
    base = state_base_emotion(state)
    intensity = state_intensity(state)
    if base is None:
        return "unknown"
    if current_emotion == "neutral":
        return "steady" if base == "neutral" else "onset"
    if base == "neutral":
        return "resolving"
    if intensity == "high":
        return "escalating"
    if intensity == "low":
        return "easing"
    return "persisting"


def is_deteriorating(current_emotion: Optional[str], state: Optional[str]) -> bool:
    """True when the forecast says a negative emotion is arriving or intensifying.

    This is the single predicate the strategy rules and the reply router care
    about most, so it lives here rather than being re-derived at each call site.
    """
    base = state_base_emotion(state)
    if base not in ("sadness", "anger", "fear"):
        return False
    return trajectory(current_emotion, state) in (
        "onset", "escalating", "persisting",
    )


# ----------------------------------------------------------------- projections
def classifier_to_forecast_label(label: Optional[str]) -> Optional[str]:
    """Identity on the shared current-emotion vocabulary.

    Retained so callers written against the old 5 -> 8 mapping keep working;
    there is nothing left to translate.
    """
    if label is None:
        return None
    return label if label in CURRENT_EMOTION_TO_ID else None


def forecast_to_classifier_label(label: Optional[str]) -> Optional[str]:
    """Project a next-STATE onto the classifier's 5-label emotion space."""
    if label is None:
        return None
    if label in CLASSIFIER_LABEL_TO_ID and label not in _STATE_DECOMPOSITION:
        return label
    return state_base_emotion(label) or (
        label if label in CLASSIFIER_LABEL_TO_ID else None
    )


def aux_emotion_id(label: Optional[str]) -> int:
    """Aux feature id for the current emotion the forecaster conditions on."""
    if label is None:
        return UNKNOWN_AUX_ID
    return CURRENT_EMOTION_TO_ID.get(label, UNKNOWN_AUX_ID)


def project_probabilities(forecast_probs: Dict[str, float]) -> Dict[str, float]:
    """Collapse a 13-state distribution onto the 5 classifier emotions.

    Mass is summed, not averaged: P(sadness) = P(sadness) + P(low_sadness) +
    P(high_sadness).
    """
    collapsed = {label: 0.0 for label in EMOTION_LABELS}
    for label, probability in forecast_probs.items():
        target = forecast_to_classifier_label(label)
        if target in collapsed:
            collapsed[target] += float(probability)
    return collapsed


def intensity_probabilities(forecast_probs: Dict[str, float]) -> Dict[str, float]:
    """Marginal over intensity: how much mass says escalate / ease / neither."""
    marginal = {"high": 0.0, "low": 0.0, "none": 0.0}
    for label, probability in forecast_probs.items():
        marginal[state_intensity(label) or "none"] += float(probability)
    return marginal


def trajectory_probabilities(
    current_emotion: Optional[str], forecast_probs: Dict[str, float]
) -> Dict[str, float]:
    """Marginal over trajectory, for the UI's 'where is this heading' readout."""
    marginal = {name: 0.0 for name in TRAJECTORIES}
    for label, probability in forecast_probs.items():
        name = trajectory(current_emotion, label)
        if name in marginal:
            marginal[name] += float(probability)
    return marginal


# ----------------------------------------------------------------- description
def is_lossy(classifier_label: Optional[str], forecast_label: Optional[str]) -> bool:
    """Always False now; kept so old callers do not have to change.

    The 8-label forecaster merged distinguishable emotions on the way down
    (`anxious` and `stressed` both became `fear`). The state space does not:
    the projection drops intensity, and the intensity is reported separately
    instead of being lost.
    """
    return False


def describe_state(state: str, current_emotion: Optional[str] = None) -> str:
    """One plain sentence for the trace and the UI."""
    base = state_base_emotion(state)
    intensity = state_intensity(state)
    if base is None:
        return f"'{state}' is not a known next-emotion state."
    if base == "neutral":
        if current_emotion in (None, "neutral"):
            return "The user is expected to stay emotionally neutral."
        return f"The user's {current_emotion} is expected to settle back to neutral."
    if intensity is None:
        return f"{base.capitalize()} is expected to appear from a neutral state."
    direction = "intensify" if intensity == "high" else "persist but soften"
    return f"The user's {base} is expected to {direction} ({intensity} intensity)."


def describe_mapping(forecast_label: str) -> str:
    """Human-readable note for the trace/UI, replacing the old lossy-mapping note."""
    base = state_base_emotion(forecast_label)
    intensity = state_intensity(forecast_label)
    if base is None:
        return f"'{forecast_label}' is not a known next-emotion state."
    if intensity is None:
        return (
            f"Next-state '{forecast_label}' carries no intensity and maps "
            f"directly onto classifier label '{base}'."
        )
    return (
        f"Next-state '{forecast_label}' maps onto classifier label '{base}' "
        f"with intensity '{intensity}'. The intensity is dropped by the "
        f"projection and reported separately, not lost."
    )
