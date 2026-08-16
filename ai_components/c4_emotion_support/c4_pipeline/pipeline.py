"""End-to-end C4 turn: classify -> track deviation -> forecast -> select -> respond.

Adds two things over the original prototype:

* the forecaster now receives the real dialogue context it was trained on, and
  the classifier's label is projected into the forecaster's vocabulary before
  being used as the auxiliary current-emotion feature;
* every stage attaches an explanation, collected under `trace["explanations"]`.

XAI is opt-in per call (`explain=`) because Integrated Gradients costs ~32
forward passes per model and the chat should stay responsive when the panel is
collapsed.
"""

from typing import Dict, List, Optional, Sequence, Tuple

from config import (
    CURRENT_EMOTION_MODEL_PATH,
    EMOTION_LABELS,
    FORECAST_LABELS,
    NEXT_EMOTION_MODEL_PATH,
)
from . import xai
from .deviation_tracker import compute_deviation
from .emotion_classifier import EmotionClassifier
from .emotion_forecaster import EmotionForecaster
from .label_mapping import (
    describe_mapping,
    forecast_to_classifier_label,
    is_lossy,
    project_probabilities,
)
from .reply_graph import generate_supportive_reply
from .response_generator import generate_response
from .safety import check_safety
from .strategy_selector import select_strategy

_ROLE_TO_SPEAKER = {"user": "user", "assistant": "supporter", "supporter": "supporter"}


def _dialogue_history(
    conversation_state: Dict[str, object], user_message: str
) -> List[Tuple[str, str]]:
    """(speaker, text) turns preceding the current message.

    Tolerates being called either before or after the caller has appended the
    incoming user message to `messages`: a trailing user turn identical to
    `user_message` is dropped so the current utterance is never double-counted
    into its own context.
    """
    messages = list(conversation_state.get("messages") or [])
    if messages:
        last = messages[-1]
        if last.get("role") == "user" and last.get("content") == user_message:
            messages = messages[:-1]
    return [
        (_ROLE_TO_SPEAKER.get(message.get("role", "user"), "user"), message.get("content", ""))
        for message in messages
    ]


def run_c4_pipeline(
    user_message: str,
    conversation_state: Dict[str, object],
    classifier: Optional[EmotionClassifier] = None,
    forecaster: Optional[EmotionForecaster] = None,
    force_fallback_forecaster: bool = False,
    explain: bool = True,
    dialogue_history: Optional[Sequence[Tuple[str, str]]] = None,
    use_llm: bool = True,
) -> Dict[str, object]:
    conversation_state.setdefault("emotion_history", [])
    conversation_state.setdefault("pipeline_traces", [])

    classifier = classifier or EmotionClassifier(CURRENT_EMOTION_MODEL_PATH)
    forecaster = forecaster or EmotionForecaster(
        NEXT_EMOTION_MODEL_PATH,
        labels=list(FORECAST_LABELS),
        force_fallback=force_fallback_forecaster,
    )
    if dialogue_history is None:
        dialogue_history = _dialogue_history(conversation_state, user_message)

    # 1. safety -------------------------------------------------------------
    safety = check_safety(user_message)

    # 2. current emotion ----------------------------------------------------
    current_output = classifier.predict(user_message)

    # 3. deviation from the previous user turn ------------------------------
    previous_emotion = (
        conversation_state["emotion_history"][-1]["emotion"]
        if conversation_state["emotion_history"]
        else None
    )
    deviation = compute_deviation(previous_emotion, current_output.label)

    # 4. next-turn forecast -------------------------------------------------
    history_emotions = [item["emotion"] for item in conversation_state["emotion_history"]]
    forecast_output = forecaster.predict(
        current_emotion=current_output.label,
        history=history_emotions,
        deviation_level=deviation["deviation_level"],
        previous_emotion=previous_emotion,
        current_message=user_message,
        dialogue_history=dialogue_history,
    )
    forecast_label = forecast_output["label"]
    forecast_projected = forecast_to_classifier_label(forecast_label) or forecast_label

    # 5. strategy -----------------------------------------------------------
    strategy = select_strategy(
        current_emotion=current_output.label,
        forecasted_emotion=forecast_label,
        deviation_level=deviation["deviation_level"],
        safety_risk_detected=bool(safety["risk_detected"]),
    )

    # 6. response -----------------------------------------------------------
    # The LangGraph stage routes the turn (crisis / adapter / base), prompts
    # Qwen3-4B, and guards the output. `use_llm=False` keeps the original
    # template responder, which is what runs when no GPU is present.
    if use_llm:
        reply_state = generate_supportive_reply(
            user_message=user_message,
            current_emotion=current_output.label,
            current_emotion_confidence=current_output.confidence,
            forecast_emotion=forecast_label,
            forecast_emotion_projected=forecast_projected,
            forecast_confidence=forecast_output["confidence"],
            deviation_level=deviation["deviation_level"],
            deviation_score=deviation["deviation_score"],
            strategy=strategy["strategy"],
            safety=safety,
            dialogue_history=dialogue_history,
        )
        response = {"response": reply_state["reply"], "strategy": strategy["strategy"]}
    else:
        response = generate_response(
            user_message=user_message,
            current_emotion=current_output.label,
            forecasted_emotion=forecast_projected,
            strategy=strategy["strategy"],
        )
        reply_state = {
            "reply": response["response"],
            "source": "template",
            "route": "template",
            "route_reason": "Reply generation is switched off; templates are used.",
            "node_trace": [],
        }

    # 7. explanations -------------------------------------------------------
    explanations: Dict[str, object] = {
        "strategy_rules": strategy["rule_trace"],
        "label_mapping_note": describe_mapping(forecast_label),
        "label_mapping_lossy": is_lossy(forecast_projected, forecast_label),
    }
    if explain:
        explanations["classifier"] = xai.explain_classifier(classifier, user_message)
        context_text = forecast_output.get("context_text") or ""
        if context_text:
            explanations["forecaster"] = xai.explain_forecaster(
                forecaster, context_text, int(forecast_output.get("aux_emotion_id", 0))
            )
            explanations["forecaster_counterfactual"] = xai.counterfactual_current_emotion(
                forecaster, context_text
            )
            explanations["forecaster_vocabulary"] = xai.vocabulary_coverage(
                forecaster, context_text
            )

    trace = {
        "user_message": user_message,
        "current_emotion": current_output.label,
        "current_emotion_confidence": current_output.confidence,
        "current_emotion_probabilities": current_output.probabilities,
        "current_emotion_source": current_output.source,
        "previous_emotion": deviation["previous_emotion"],
        "deviation_score": deviation["deviation_score"],
        "deviation_level": deviation["deviation_level"],
        "forecasted_next_emotion": forecast_label,
        "forecasted_next_emotion_projected": forecast_projected,
        "forecast_confidence": forecast_output["confidence"],
        "forecast_probabilities": forecast_output.get("probabilities"),
        "forecast_probabilities_projected": project_probabilities(
            forecast_output.get("probabilities") or {}
        ),
        "forecast_source": forecast_output.get("source", "rule"),
        "forecast_context_text": forecast_output.get("context_text", ""),
        "forecast_unk_rate": forecast_output.get("unk_rate", 0.0),
        "forecast_aux_emotion": forecast_output.get("aux_emotion_label"),
        "selected_strategy": strategy["strategy"],
        "strategy_reason": strategy["reason"],
        "supportive_response": response["response"],
        "safety": safety,
        "explanations": explanations,
        # --- reply-generation stage (LangGraph) ---------------------------
        "reply_source": reply_state.get("source", "template"),
        "reply_route": reply_state.get("route", "template"),
        "reply_route_reason": reply_state.get("route_reason", ""),
        "reply_esconv_strategy": reply_state.get("esconv_strategy"),
        "reply_plan": reply_state.get("response_plan"),
        "reply_strategy_mapping_reason": reply_state.get("strategy_mapping_reason"),
        "reply_prompt": reply_state.get("prompt_text", ""),
        "reply_raw": reply_state.get("raw_reply", ""),
        "reply_guard": reply_state.get("guard", {}),
        "reply_generation": reply_state.get("generation", {}),
        "reply_attempts": reply_state.get("attempts", 0),
        "reply_node_trace": reply_state.get("node_trace", []),
    }

    conversation_state["pipeline_traces"].append(trace)
    conversation_state["emotion_history"].append(
        {
            "turn": len(conversation_state["emotion_history"]) + 1,
            "emotion": current_output.label,
            "confidence": current_output.confidence,
            "deviation_level": deviation["deviation_level"],
            "deviation_score": deviation["deviation_score"],
            "strategy": strategy["strategy"],
            "forecasted_emotion": forecast_label,
            "forecast_confidence": forecast_output["confidence"],
        }
    )

    return trace
