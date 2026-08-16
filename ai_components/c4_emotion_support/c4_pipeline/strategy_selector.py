"""Rule-based support strategy selection.

The rules are unchanged, but selection now returns the full evaluation trace
(every rule, whether it fired, and why) rather than just the winner. That trace
is the explanation for this stage of the pipeline: unlike the neural models,
this component is transparent by construction, and the UI shows it as such
instead of pretending it needs attribution.

Rules are evaluated top to bottom; the first match wins.
"""

from typing import Callable, Dict, List, Tuple

from .label_mapping import forecast_to_classifier_label

STRATEGIES = [
    "Listen",
    "Comfort",
    "Reassure",
    "Encourage",
    "Maintain Tone",
    "Safe Fallback",
]


def _rules(
    current: str, forecast: str, deviation_level: str, safety_risk: bool
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
            "current in {sadness, disgust} and forecast in {sadness, fear}",
            lambda: current in ("sadness", "disgust") and forecast in ("sadness", "fear"),
            "Low mood with a likely negative next emotion suggests a comforting response.",
        ),
        (
            "Reassure",
            "current == fear or deviation_level == High",
            lambda: current == "fear" or deviation_level == "High",
            "Fear or high emotional deviation was detected, so reassurance is selected.",
        ),
        (
            "Maintain Tone",
            "current == neutral and forecast in {joy, neutral}",
            lambda: current == "neutral" and forecast in ("joy", "neutral"),
            "The tone appears steady or positive, so maintaining the tone is suitable.",
        ),
        (
            "Encourage",
            "current in {sadness, fear} and forecast in {neutral, joy}",
            lambda: current in ("sadness", "fear") and forecast in ("neutral", "joy"),
            "A shift toward neutral or positive emotion suggests an encouraging response.",
        ),
    ]


def select_strategy(
    current_emotion: str,
    forecasted_emotion: str,
    deviation_level: str,
    safety_risk_detected: bool,
) -> Dict[str, object]:
    """Pick a support strategy and return the rule trace that produced it.

    `forecasted_emotion` may arrive in the forecaster's 8-label vocabulary; the
    rules are written against the classifier's 5 labels, so it is projected
    first and both forms are reported.
    """
    projected_forecast = forecast_to_classifier_label(forecasted_emotion) or forecasted_emotion

    trace: List[Dict[str, object]] = []
    selected = None
    for strategy, condition, predicate, reason in _rules(
        current_emotion, projected_forecast, deviation_level, safety_risk_detected
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
            "forecasted_emotion": forecasted_emotion,
            "forecasted_emotion_projected": projected_forecast,
            "deviation_level": deviation_level,
            "safety_risk_detected": safety_risk_detected,
        },
        "rule_trace": trace,
    }
