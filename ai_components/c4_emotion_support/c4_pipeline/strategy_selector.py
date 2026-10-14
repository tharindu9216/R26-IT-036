"""Rule-based support strategy selection.

Selection returns the full evaluation trace (every rule, whether it fired, and
why) rather than just the winner. That trace is the explanation for this stage:
unlike the neural models, this component is transparent by construction, and
the UI shows it as such instead of pretending it needs attribution.

Rules are evaluated top to bottom; the first match wins.

What the next-STATE forecaster changed here
-------------------------------------------
The old rules read a bare forecast emotion, so `Comfort` fired on
`current in {sadness, disgust} and forecast in {sadness, fear}` -- which, given
that the old forecaster was barely distinguishable from persistence, was
close to "the user is sad, and will probably still be sad". It could not tell
deepening distress from distress that was already lifting, so it answered both
the same way.

The forecaster now predicts a *state*, and `label_mapping.trajectory` turns
(current emotion, next state) into one of:

    steady · onset · escalating · easing · resolving

`escalating` and `easing` project onto the same emotion and call for opposite
responses, so the rules are written against the trajectory:

    escalating / onset of distress  -> Comfort      (stay with the feeling)
    easing / resolving              -> Encourage    (name the improvement)

The strategy vocabulary itself is unchanged -- these six names are what
`strategy_mapping.py` knows how to translate into ESConv's annotation scheme,
and inventing a seventh would put the adapter out of distribution.
"""

from typing import Callable, Dict, List, Optional, Tuple

from .label_mapping import (
    forecast_to_classifier_label,
    is_deteriorating,
    state_intensity,
    trajectory,
)

STRATEGIES = [
    "Listen",
    "Comfort",
    "Reassure",
    "Encourage",
    "Maintain Tone",
    "Safe Fallback",
]

NEGATIVE_EMOTIONS = ("sadness", "anger", "fear", "disgust")

# Trajectories that mean the situation is getting better rather than worse.
IMPROVING = ("easing", "resolving")


def _rules(
    current: str,
    forecast: str,
    path: str,
    deviation_level: str,
    safety_risk: bool,
    deteriorating: bool,
) -> List[Tuple[str, str, Callable[[], bool], str]]:
    """(strategy, condition text, predicate, reason) in priority order."""
    return [
        (
            "Safe Fallback",
            "safety_risk_detected",
            lambda: safety_risk,
            "Safety risk keywords were detected, so a safe fallback response is selected.",
        ),
        (
            "Comfort",
            "negative emotion with trajectory in {escalating, onset, persisting}",
            lambda: deteriorating,
            "The forecast says a negative emotion is intensifying or about to "
            "appear, so the response stays with the feeling rather than trying "
            "to move past it.",
        ),
        (
            "Reassure",
            "current == fear or deviation_level == High",
            lambda: current == "fear" or deviation_level == "High",
            "Fear or high emotional deviation was detected, so reassurance is selected.",
        ),
        (
            "Encourage",
            "current in negative and trajectory in {easing, resolving}",
            lambda: current in NEGATIVE_EMOTIONS and path in IMPROVING,
            "The forecast has the negative emotion easing or settling back to "
            "neutral, so the response names that movement and offers a small "
            "next step.",
        ),
        (
            # Reached only after the three distress rules above have declined,
            # so "not negative" is enough here. Escalation is deliberately NOT
            # excluded: `high_joy` is the user getting happier, and answering
            # that with comfort would be a bug.
            "Maintain Tone",
            "forecast base emotion in {neutral, joy}",
            lambda: forecast in ("neutral", "joy"),
            "The forecast is neutral or positive, so maintaining the tone is suitable.",
        ),
    ]


def select_strategy(
    current_emotion: str,
    forecasted_emotion: str,
    deviation_level: str,
    safety_risk_detected: bool,
) -> Dict[str, object]:
    """Pick a support strategy and return the rule trace that produced it.

    `forecasted_emotion` is a next-emotion STATE (`high_sadness`, `low_joy`,
    `neutral`, ...). It is decomposed here into the base emotion the older
    rules were written against plus the trajectory the new rules use, and all
    three forms are reported in `inputs` so the trace stays readable.

    A bare emotion label is still accepted. Its trajectory comes back as
    `persisting` rather than a real direction, which routes a still-negative
    forecast to Comfort and a neutral or positive one to Maintain Tone. That is
    the cautious reading, and it is what the rule fallback and any caller
    written against the old contract will hit.
    """
    projected_forecast = (
        forecast_to_classifier_label(forecasted_emotion) or forecasted_emotion
    )
    path = trajectory(current_emotion, forecasted_emotion)
    deteriorating = is_deteriorating(current_emotion, forecasted_emotion)

    trace: List[Dict[str, object]] = []
    selected: Optional[Dict[str, str]] = None
    for strategy, condition, predicate, reason in _rules(
        current_emotion,
        projected_forecast,
        path,
        deviation_level,
        safety_risk_detected,
        deteriorating,
    ):
        fired = bool(predicate())
        trace.append(
            {
                "strategy": strategy,
                "condition": condition,
                "fired": fired,
                "reached": selected is None,
            }
        )
        if fired and selected is None:
            selected = {"strategy": strategy, "reason": reason}

    if selected is None:
        selected = {
            "strategy": "Listen",
            "reason": "No specific rule matched, so listening is used as the safe default.",
        }
        trace.append(
            {
                "strategy": "Listen",
                "condition": "default",
                "fired": True,
                "reached": True,
            }
        )

    return {
        "strategy": selected["strategy"],
        "reason": selected["reason"],
        "inputs": {
            "current_emotion": current_emotion,
            "forecasted_state": forecasted_emotion,
            "forecasted_emotion_projected": projected_forecast,
            "forecast_intensity": state_intensity(forecasted_emotion),
            "forecast_trajectory": path,
            "deteriorating": deteriorating,
            "deviation_level": deviation_level,
            "safety_risk_detected": safety_risk_detected,
        },
        "rule_trace": trace,
    }
