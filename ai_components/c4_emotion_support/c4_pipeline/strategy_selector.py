from typing import Dict


def select_strategy(
    current_emotion: str,
    forecasted_emotion: str,
    deviation_level: str,
    safety_risk_detected: bool,
) -> Dict[str, str]:
    if safety_risk_detected:
        return {
            "strategy": "Safe Fallback",
            "reason": "Safety risk keywords were detected, so a safe fallback response is selected.",
        }

    if current_emotion in ["sadness", "disgust"] and forecasted_emotion in ["sadness", "fear"]:
        return {
            "strategy": "Comfort",
            "reason": "Low mood with a likely negative next emotion suggests a comforting response.",
        }

    if current_emotion == "fear" or deviation_level == "High":
        return {
            "strategy": "Reassure",
            "reason": "Fear or high emotional deviation was detected, so reassurance is selected.",
        }

    if current_emotion in ["neutral"] and forecasted_emotion in ["joy", "neutral"]:
        return {
            "strategy": "Maintain Tone",
            "reason": "The tone appears steady or positive, so maintaining the tone is suitable.",
        }

    if current_emotion in ["sadness", "fear"] and forecasted_emotion in ["neutral", "joy"]:
        return {
            "strategy": "Encourage",
            "reason": "A shift toward neutral or positive emotion suggests an encouraging response.",
        }

    return {"strategy": "Listen", "reason": "Listening is a safe default strategy."}
