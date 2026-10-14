from typing import Dict

TEMPLATES = {
    "Listen": "I hear you. It sounds like this has been affecting you, and it is okay to take a moment to explain what feels hardest right now.",
    "Comfort": "I am sorry you are feeling this way. That sounds difficult, but you do not have to process everything at once. Let us take it step by step.",
    "Reassure": "It makes sense to feel unsettled. Try to focus on the next small thing you can control right now, and we can work through it calmly.",
    "Encourage": "You are already doing something positive by talking about it. Let us focus on one small step that could make the situation feel more manageable.",
    "Maintain Tone": "Thanks for sharing that. I will stay with your current pace and continue supporting you based on how you are feeling.",
    "Safe Fallback": "I am really sorry you are feeling this way. I am not a replacement for professional support, but reaching out to someone you trust or a qualified support service could help right now.",
}


def generate_response(
    user_message: str,
    current_emotion: str,
    forecasted_emotion: str,
    strategy: str,
) -> Dict[str, str]:
    template = TEMPLATES.get(strategy, TEMPLATES["Listen"])
    return {"response": template, "strategy": strategy}
