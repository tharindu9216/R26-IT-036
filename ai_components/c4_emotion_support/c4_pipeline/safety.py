from typing import Dict

RISK_KEYWORDS = {
    "suicide",
    "kill myself",
    "self harm",
    "self-harm",
    "hurt myself",
    "end my life",
    "take my life",
    "die",
}


def check_safety(user_message: str) -> Dict[str, str]:
    text = user_message.lower()
    for keyword in RISK_KEYWORDS:
        if keyword in text:
            return {"risk_detected": True, "risk_type": "crisis_or_self_harm"}
    return {"risk_detected": False, "risk_type": "none"}
