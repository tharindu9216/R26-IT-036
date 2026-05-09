from typing import Dict, Optional

from config import CURRENT_EMOTION_MODEL_PATH, EMOTION_LABELS, NEXT_EMOTION_MODEL_PATH
from .deviation_tracker import compute_deviation
from .emotion_classifier import EmotionClassifier
from .emotion_forecaster import EmotionForecaster
from .response_generator import generate_response
from .safety import check_safety
from .strategy_selector import select_strategy


def run_c4_pipeline(
    user_message: str,
    conversation_state: Dict[str, object],
    classifier: Optional[EmotionClassifier] = None,
    forecaster: Optional[EmotionForecaster] = None,
    force_fallback_forecaster: bool = False,
) -> Dict[str, object]:
    conversation_state.setdefault("emotion_history", [])
    conversation_state.setdefault("pipeline_traces", [])

    classifier = classifier or EmotionClassifier(CURRENT_EMOTION_MODEL_PATH)
    forecaster = forecaster or EmotionForecaster(
        NEXT_EMOTION_MODEL_PATH, labels=list(EMOTION_LABELS), force_fallback=force_fallback_forecaster
    )

    safety = check_safety(user_message)

    current_output = classifier.predict(user_message)
    previous_emotion = (
        conversation_state["emotion_history"][-1]["emotion"]
        if conversation_state["emotion_history"]
        else None
    )

    deviation = compute_deviation(previous_emotion, current_output.label)

    history_emotions = [item["emotion"] for item in conversation_state["emotion_history"]]
    forecast_output = forecaster.predict(
        current_emotion=current_output.label,
        history=history_emotions,
        deviation_level=deviation["deviation_level"],
        previous_emotion=previous_emotion,
    )

    strategy = select_strategy(
        current_emotion=current_output.label,
        forecasted_emotion=forecast_output["label"],
        deviation_level=deviation["deviation_level"],
        safety_risk_detected=bool(safety["risk_detected"]),
    )

    response = generate_response(
        user_message=user_message,
        current_emotion=current_output.label,
        forecasted_emotion=forecast_output["label"],
        strategy=strategy["strategy"],
    )

    trace = {
        "user_message": user_message,
        "current_emotion": current_output.label,
        "current_emotion_confidence": current_output.confidence,
        "current_emotion_probabilities": current_output.probabilities,
        "previous_emotion": deviation["previous_emotion"],
        "deviation_score": deviation["deviation_score"],
        "deviation_level": deviation["deviation_level"],
        "forecasted_next_emotion": forecast_output["label"],
        "forecast_confidence": forecast_output["confidence"],
        "forecast_probabilities": forecast_output.get("probabilities"),
        "selected_strategy": strategy["strategy"],
        "strategy_reason": strategy["reason"],
        "supportive_response": response["response"],
        "safety": safety,
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
            "forecasted_emotion": forecast_output["label"],
        }
    )

    return trace
