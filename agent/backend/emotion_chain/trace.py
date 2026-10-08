"""Turn one finished graph run into an ordered record of what each part did.

The UI history drawer answers "how did this reply happen": which components
ran, what each returned, and -- the part that is not obvious from the reply
itself -- why the supportive strategy was or was not selected. That is a
presentation of state the graph already produced, so nothing here runs a model
or re-decides anything; ``build_trace`` only reads the finished state.

Two deliberate choices:

* A component that did not run this turn is kept as a ``skipped`` step with the
  reason, rather than dropped. "C3 does not exist in voice mode" and "C3 ran and
  found nothing" are very different things, and a trace that silently omitted
  the first would read as the second.
* Decisions carry their reasons, not just their outcome. ``reply_route_reasons``
  and ``escalation_signals`` both return the conditions that fired, so the
  drawer can show the rule that applied instead of an unexplained label.
"""

from __future__ import annotations

from typing import Any

from .core import reply_route_reasons
from .support_contacts import MIN_ESCALATION_SIGNALS, escalation_signals

TRACE_VERSION = 2

ROUTE_REASON_TEXT = {
    "negative_current_emotion": "current emotion is in a negative family",
    # The next-intensity forecaster keeps the current emotion's family, so this
    # reason only fires alongside the one above and never decides a route by
    # itself. It is still listed: with the confidence floor applied it means
    # "the forecaster expects this to intensify", which is worth showing.
    "negative_forecast": "forecast state is negative, at or above the confidence floor",
    "sensor_stress": "the wearable reported stress",
    "text_signal": "the fused text stress/CBT signal was flagged",
    "voice_stress": "the voice appraisal reported stress",
}

ESCALATION_REASON_TEXT = {
    "high_intensity_forecast": "the next turn is forecast to intensify",
    "high_deviation": "emotion crossed valence since the previous turn",
    "sensor_stress": "the wearable reported stress",
    "text_signal": "the fused text stress/CBT signal was flagged",
    "voice_stress": "the voice appraisal reported stress",
}

STRATEGIES = {
    "supportive_adapter": (
        "ESConv supportive LoRA, with the acknowledge / reflect / suggest / ask "
        "system prompt"
    ),
    "base": "Plain Qwen, with the short conversational system prompt",
    "crisis": "Fixed safety template; no model is called at all",
}


def _percent(value: Any) -> str | None:
    if value is None:
        return None
    return f"{float(value):.1%}"


def _output(label: str, value: Any) -> dict[str, str] | None:
    """Drop an output whose value the turn never produced."""

    if value is None or value == "":
        return None
    return {"label": label, "value": str(value)}


def _step(
    key: str,
    component: str,
    title: str,
    *,
    status: str = "ran",
    summary: str | None = None,
    detail: str | None = None,
    outputs: list[dict[str, str] | None] | None = None,
    reasons: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    step: dict[str, Any] = {
        "key": key,
        "component": component,
        "title": title,
        "status": status,
    }
    if summary is not None:
        step["summary"] = summary
    if detail:
        step["detail"] = detail
    kept = [item for item in (outputs or []) if item]
    if kept:
        step["outputs"] = kept
    if reasons:
        step["reasons"] = reasons
    return step


def _input_step(state: dict, mode: str) -> dict[str, Any]:
    title = (
        "Speech transcribed by Faster-Whisper"
        if mode == "voice"
        else "The message as typed"
    )
    return _step("input", "Input", title, summary=str(state.get("user_text", "")))


def _safety_step(state: dict) -> dict[str, Any]:
    crisis = state.get("safety_status") == "crisis"
    return _step(
        "safety",
        "Crisis screen",
        "Regex patterns for self-harm language, checked before any model runs",
        summary="crisis" if crisis else "normal",
        detail=(
            "Matched a crisis pattern, so the rest of the pipeline is skipped and "
            "a fixed template answers instead. This is a conservative routing "
            "aid, not a validated risk classifier."
            if crisis
            else None
        ),
    )


def _sensor_step(state: dict) -> dict[str, Any]:
    # Present the wearable and questionnaire fusion as the shared stress row.
    # The raw wearable value remains visible as an input, while the row's
    # headline is the final value used by routing.
    if state.get("multimodal_stress_available"):
        weights = state.get("multimodal_stress_weights") or {}
        inputs = state.get("multimodal_stress_inputs") or {}
        source_labels = {
            "questionnaire": "check-in",
            "sensor": "wearable",
        }
        contributions = [
            _output(
                source_labels.get(source, source),
                f"{_percent(inputs.get(source))} at {float(weight):.0%} weight",
            )
            for source, weight in weights.items()
        ]
        return _step(
            "sensor",
            "Combined stress",
            "Questionnaire (60%) and wearable (40%) combined",
            summary=(
                f"{'Stressed' if state.get('multimodal_stress') else 'Calm'} · "
                f"{_percent(state.get('multimodal_stress_probability'))}"
            ),
            detail=(
                "This is the header score captured when this message was sent and "
                "used by both the written and voice paths. The live header can change "
                "as new sensor windows arrive. C2 voice appraisal remains separate."
            ),
            outputs=[
                _output(
                    "combined stress score",
                    _percent(state.get("multimodal_stress_probability")),
                ),
                *contributions,
            ],
        )
    if not state.get("sensor_stress_available"):
        return _step(
            "sensor",
            "C1 - wearable",
            "Stress from BVP + EDA on the wrist sensor",
            status="skipped",
            summary="no reading",
            detail="No sensor is connected, or it has not finished calibrating.",
        )
    flagged = bool(state.get("sensor_stress"))
    return _step(
        "sensor",
        "C1 - wearable",
        "Stress from BVP + EDA on the wrist sensor",
        summary="stressed" if flagged else "calm",
        outputs=[
            _output("probability", _percent(state.get("sensor_stress_probability")))
        ],
    )


def _voice_steps(state: dict, mode: str) -> list[dict[str, Any]]:
    if mode != "voice":
        return []
    uncertain = bool(state.get("voice_appraisal_uncertain"))
    flagged = bool(state.get("voice_stress"))
    summary = "uncertain" if uncertain else ("stressed" if flagged else "calm")
    return [
        _step(
            "voice",
            "C2 - voice appraisal",
            "Arousal/dominance/valence from tone, mapped by fuzzy appraisal rules",
            summary=summary,
            outputs=[
                _output("appraisal state", state.get("voice_appraisal_state")),
                _output("stress score", _percent(state.get("voice_stress_score"))),
            ],
        )
    ]


def _c3_steps(state: dict, mode: str) -> list[dict[str, Any]]:
    if mode == "voice":
        return [
            _step(
                "c3",
                "C3 - stress + CBT",
                "Two DeBERTa heads over the message text",
                status="skipped",
                summary="not in the voice graph",
                detail=(
                    "Voice turns use the C2 appraisal instead; the C3 nodes are "
                    "not part of that graph and its models are never loaded."
                ),
            )
        ]
    if state.get("current_emotion") is None and state.get("safety_status") == "crisis":
        return [
            _step(
                "c3",
                "C3 - stress + CBT",
                "Two DeBERTa heads over the message text",
                status="skipped",
                summary="not reached",
                detail="The crisis branch answers before C3 runs.",
            )
        ]

    stress = bool(state.get("text_stress"))
    distortion = bool(state.get("has_cbt_distortion"))
    flagged = bool(state.get("text_signal_flagged"))
    return [
        _step(
            "c3",
            "C3 - stress + CBT",
            "Two DeBERTa heads over the message text",
            summary=(
                f"stress {'yes' if stress else 'no'} - "
                f"distortion {'yes' if distortion else 'no'}"
            ),
            outputs=[
                _output(
                    "stress probability",
                    _percent(state.get("text_stress_probability")),
                ),
                _output(
                    "distortion probability", _percent(state.get("cbt_probability"))
                ),
            ],
        ),
        _step(
            "fusion",
            "Late fusion",
            "Decision-level fusion of the two C3 heads into one text signal",
            summary="flagged" if flagged else "not flagged",
            detail="Routing sees only this fused signal, never the two heads separately.",
            outputs=[
                _output(
                    "fused probability",
                    _percent(state.get("text_signal_probability")),
                ),
                _output("method", state.get("text_signal_method")),
            ],
        ),
    ]


def _emotion_steps(state: dict) -> list[dict[str, Any]]:
    current = state.get("current_emotion")
    if not current:
        return [
            _step(
                "emotion",
                "Emotion chain",
                "Current emotion, then the next-turn intensity forecast",
                status="skipped",
                summary="not reached",
                detail="The crisis branch answers before the classifiers run.",
            )
        ]
    intensity = str(state.get("next_intensity", ""))
    branches = state.get("next_intensity_branches") or {}
    next_state = str(state.get("next_emotion", ""))
    detail = (
        "The forecaster predicts intensity, not the next emotion, so the "
        f"family carries over from the current one and the pair names "
        f"{next_state or 'the next state'}."
    )
    if str(current).strip().lower() == "neutral":
        detail += (
            " A neutral turn has no low/high state to name, so it stays "
            "neutral and only the intensity above is the model's own output."
        )
    return [
        _step(
            "emotion",
            "Current emotion",
            "Classifier over the message, into one of five families",
            summary=str(current),
            outputs=[
                _output("confidence", _percent(state.get("current_emotion_confidence")))
            ],
        ),
        _step(
            "next",
            "Next intensity",
            "Will the next turn be more or less intense: a sparse TF-IDF model "
            "and a frozen-RoBERTa ensemble, blended and cut at a fixed threshold",
            summary=intensity or next_state,
            detail=detail,
            outputs=[
                _output("state", next_state),
                _output(
                    "P(high)", _percent(state.get("next_intensity_probability"))
                ),
                _output("sparse branch", _percent(branches.get("baseline"))),
                _output("neural branch", _percent(branches.get("fusion"))),
            ],
        ),
    ]


def _deviation_step(state: dict) -> dict[str, Any]:
    if not state.get("current_emotion"):
        return _step(
            "deviation",
            "Deviation",
            "How far the emotion moved since the previous turn",
            status="skipped",
            summary="not reached",
        )
    previous = state.get("previous_emotion")
    return _step(
        "deviation",
        "Deviation",
        "How far the emotion moved since the previous turn",
        summary=str(state.get("deviation_level", "None")),
        detail=(
            None
            if previous
            else "First turn of the conversation, so there is nothing to compare against."
        ),
        outputs=[
            _output("previous emotion", previous),
            _output("score", f"{float(state.get('deviation_score', 0.0)):.2f}"),
        ],
    )


def _strategy_step(
    state: dict, route: str, next_negative_min_confidence: float
) -> dict[str, Any]:
    """The routing decision, with the conditions that actually produced it."""

    title = "Which reply strategy this turn gets"
    if route == "crisis":
        return _step(
            "strategy",
            "Strategy selection",
            title,
            summary="crisis",
            detail=STRATEGIES["crisis"],
            reasons=[
                {
                    "code": "crisis",
                    "text": "the crisis screen matched, which overrides routing",
                }
            ],
        )

    current = state.get("current_emotion")
    if not current:
        return _step(
            "strategy", "Strategy selection", title, status="skipped", summary=route
        )

    has_multimodal = bool(state.get("multimodal_stress_available", False))
    fired = reply_route_reasons(
        str(current),
        str(state.get("next_emotion", "")),
        sensor_stress=bool(state.get("sensor_stress", False)) if not has_multimodal else False,
        text_signal_flagged=bool(state.get("text_signal_flagged", False)),
        voice_stress=bool(state.get("voice_stress", False)) if not has_multimodal else False,
        multimodal_stress=(
            bool(state.get("multimodal_stress", False)) if has_multimodal else False
        ),
        next_emotion_confidence=state.get("next_emotion_confidence"),
        next_negative_min_confidence=next_negative_min_confidence,
    )
    reasons = [
        {"code": code, "text": ROUTE_REASON_TEXT.get(code, code)}
        for code in fired
        if code != "multimodal_stress"
    ]
    if not fired:
        reasons = [
            {
                "code": "no_trigger",
                "text": "no distress condition fired, so the turn stays on the base model",
            }
        ]
    return _step(
        "strategy",
        "Strategy selection",
        title,
        summary=route,
        detail=(
            STRATEGIES.get(route, route)
            + ". Any one condition below is enough to select the supportive strategy."
        ),
        reasons=reasons,
    )


def _generation_step(state: dict, route: str) -> dict[str, Any]:
    source = str(state.get("reply_source", ""))
    labels = {
        "qwen": "generated locally",
        "qwen_remote": "generated on the support node",
        "template_fallback": "template fallback - the model could not be reached",
        "safety_template": "fixed safety template, no model call",
    }
    detail = None
    if source == "template_fallback":
        detail = (
            "Generation failed, so the routed template answered instead. Check "
            "/api/deployment if that was not expected."
        )
    return _step(
        "generate",
        "Reply generation",
        "Qwen 4B, with the selected strategy's system prompt and the recent "
        "turns replayed",
        summary=labels.get(source, source or "unknown"),
        detail=detail,
        outputs=[
            _output("adapter active", "yes" if route == "supportive_adapter" else "no")
        ],
    )


def _escalation_step(state: dict) -> dict[str, Any]:
    title = "Whether a verified human contact is offered"
    contacts = state.get("support_contacts") or []
    named = ", ".join(str(contact.get("name", "")) for contact in contacts)

    if state.get("safety_status") == "crisis":
        return _step(
            "escalation",
            "Escalation",
            title,
            summary="contacts offered",
            reasons=[
                {"code": "crisis", "text": "every crisis turn offers contacts"}
            ],
            outputs=[_output("contacts", named)],
        )

    current = state.get("current_emotion")
    if not current:
        return _step(
            "escalation", "Escalation", title, status="skipped", summary="not reached"
        )

    has_multimodal = bool(state.get("multimodal_stress_available", False))
    fired = escalation_signals(
        str(current),
        str(state.get("next_emotion", "")),
        deviation_level=str(state.get("deviation_level", "None")),
        sensor_stress=bool(state.get("sensor_stress", False)) if not has_multimodal else False,
        text_signal_flagged=bool(state.get("text_signal_flagged", False)),
        voice_stress=bool(state.get("voice_stress", False)) if not has_multimodal else False,
        multimodal_stress=(
            bool(state.get("multimodal_stress", False)) if has_multimodal else False
        ),
    )
    return _step(
        "escalation",
        "Escalation",
        title,
        summary="contacts offered" if contacts else "not escalated",
        detail=(
            f"{len(fired)} of the {MIN_ESCALATION_SIGNALS} agreeing signals needed. "
            "The contact sentence is appended after generation, so the number is "
            "never something the model wrote."
        ),
        reasons=[
            {"code": code, "text": ESCALATION_REASON_TEXT.get(code, code)}
            for code in fired
            if code != "multimodal_stress"
        ],
        outputs=[_output("contacts", named)],
    )


def build_trace(
    state: dict,
    *,
    mode: str = "text",
    next_negative_min_confidence: float = 0.0,
) -> dict[str, Any]:
    """Describe one finished turn as an ordered list of what each part returned.

    ``state`` is the dict LangGraph returns from ``invoke``, which already
    carries what the caller put in (the sensor snapshot, the transcript).
    Nothing is recomputed except the two decision explanations, and those are
    pure functions of values the state already holds.
    """

    route = str(state.get("reply_route", ""))
    steps = [
        _input_step(state, mode),
        _safety_step(state),
        _sensor_step(state),
        *_voice_steps(state, mode),
        *_c3_steps(state, mode),
        *_emotion_steps(state),
        _deviation_step(state),
        _strategy_step(state, route, next_negative_min_confidence),
        _generation_step(state, route),
        _escalation_step(state),
        _step(
            "output",
            "Reply",
            "What the user sees",
            summary=str(state.get("reply", "")),
        ),
    ]
    return {"version": TRACE_VERSION, "mode": mode, "route": route, "steps": steps}
