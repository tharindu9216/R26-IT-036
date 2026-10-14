"""Bridge C4's rule-selected strategy into the vocabulary the adapter was trained on.

The ESConv LoRA adapter is conditioned on a `Response strategy: <name>` line, and
during fine-tuning that name always came from ESConv's own 8-strategy annotation
scheme (`preprocess.py` copies the `strategy` column straight into the prompt).
C4's rule engine emits a different, smaller vocabulary. Feeding a C4 name such as
"Maintain Tone" into that slot puts the adapter out of distribution: it has never
seen the token in that position and the conditioning silently degrades into noise.

So the two vocabularies are mapped explicitly here, with the reasoning recorded
next to each pair rather than left implicit in a dict literal.

The second job of this module is routing. The ESConv EDA found **no positive
emotion label anywhere in the corpus** -- the 11 conversation-level labels are
anxiety, depression, sadness, anger, fear, shame, disgust, nervousness, guilt,
jealousy, pain. The adapter is therefore only trained on distress. Sending a
cheerful turn through it produces consolation nobody asked for, so positive and
neutral states are routed to the unmodified base model instead.
"""

from typing import Dict, List, Optional, Sequence, Tuple

from config import NEGATIVE, NEUTRAL, POSITIVE

# ESConv's annotation scheme, in corpus-frequency order. Only these strings may
# ever reach the adapter's `Response strategy:` slot.
ESCONV_STRATEGIES = (
    "Question",                      # 20.7% of supporter turns
    "Others",                        # 18.2%
    "Providing Suggestions",         # 16.1%
    "Affirmation and Reassurance",   # 15.4%
    "Self-disclosure",               #  9.3%
    "Reflection of feelings",        #  7.8%
    "Information",                   #  6.6%
    "Restatement or Paraphrasing",   #  5.9%
)

# (C4 strategy, ESConv strategy, response plan handed to the model, rationale).
# The response plan is prose rather than a label because the base model -- which
# serves the positive/neutral route and has never seen ESConv -- reads it too.
_STRATEGY_TABLE: Tuple[Tuple[str, str, str, str], ...] = (
    (
        "Listen",
        "Question",
        "Show you followed what they said, then ask one open question that "
        "invites them to say more about the part that matters most.",
        "Listening in ESConv is enacted by asking rather than asserting; "
        "Question is the annotation that carries that move.",
    ),
    (
        "Comfort",
        "Reflection of feelings",
        "Name the feeling you heard without overstating it, and let them know "
        "the reaction makes sense. Do not rush to fix anything.",
        "Comfort is acknowledgement, which is exactly what Reflection of "
        "feelings annotates: mirroring the emotion back.",
    ),
    (
        "Reassure",
        "Affirmation and Reassurance",
        "Steady the moment: affirm something they are already doing, and offer "
        "one calm sentence that reduces the sense of threat.",
        "Direct one-to-one match; this pairing is the least ambiguous in the table.",
    ),
    (
        "Encourage",
        "Providing Suggestions",
        "Recognise the effort they have made, then offer one small, concrete, "
        "genuinely optional next step.",
        "Encouragement in C4 fires when the forecast improves, so the useful "
        "move is forward-looking; Providing Suggestions is that annotation. "
        "Affirmation alone would repeat the Reassure branch.",
    ),
    (
        "Maintain Tone",
        "Restatement or Paraphrasing",
        "Stay at their pace. Reflect back what they said in your own words and "
        "leave room for them to continue.",
        "Maintaining tone means adding no new emotional colour, which is what "
        "restatement does by construction.",
    ),
)

_TO_ESCONV: Dict[str, str] = {c4: esconv for c4, esconv, _, _ in _STRATEGY_TABLE}
_TO_PLAN: Dict[str, str] = {c4: plan for c4, _, plan, _ in _STRATEGY_TABLE}
_TO_REASON: Dict[str, str] = {c4: reason for c4, _, _, reason in _STRATEGY_TABLE}

# Used when the rule engine returns something unmapped. "Others" is ESConv's own
# catch-all, so the adapter has seen it ~3300 times and it degrades gracefully.
_FALLBACK_ESCONV = "Others"
_FALLBACK_PLAN = "Respond supportively and stay with what the person actually said."

# Routes. "crisis" never reaches a language model at all.
ROUTE_CRISIS = "crisis"
ROUTE_ADAPTER = "adapter"
ROUTE_BASE = "base"


def to_esconv_strategy(c4_strategy: str) -> str:
    """C4 strategy name -> the ESConv label the adapter was conditioned on."""
    return _TO_ESCONV.get(c4_strategy, _FALLBACK_ESCONV)


def response_plan(c4_strategy: str) -> str:
    """Plain-language instruction describing the move the reply should make."""
    return _TO_PLAN.get(c4_strategy, _FALLBACK_PLAN)


def mapping_reason(c4_strategy: str) -> str:
    """Why this C4 strategy maps where it does, for the UI's explanation panel."""
    return _TO_REASON.get(
        c4_strategy,
        f"'{c4_strategy}' has no explicit mapping, so ESConv's catch-all "
        f"'{_FALLBACK_ESCONV}' annotation is used.",
    )


def mapping_table() -> List[Dict[str, str]]:
    """The full table, for display."""
    return [
        {
            "c4_strategy": c4,
            "esconv_strategy": esconv,
            "response_plan": plan,
            "why": reason,
        }
        for c4, esconv, plan, reason in _STRATEGY_TABLE
    ]


def valence_of(emotion: Optional[str]) -> str:
    """positive / negative / neutral / unknown, across both label spaces."""
    if not emotion:
        return "unknown"
    label = str(emotion).lower()
    if label in POSITIVE:
        return "positive"
    if label in NEGATIVE:
        return "negative"
    if label in NEUTRAL:
        return "neutral"
    return "unknown"


def select_route(
    safety_risk_detected: bool,
    current_emotion: str,
    forecast_emotion: str,
) -> Dict[str, object]:
    """Choose crisis / adapter / base, and report the rule that decided it.

    The forecast is consulted as well as the current emotion: a turn that reads
    neutral now but is predicted to deteriorate is still a distress turn, and the
    adapter handles those better than the base model does.
    """
    current_valence = valence_of(current_emotion)
    forecast_valence = valence_of(forecast_emotion)

    if safety_risk_detected:
        return {
            "route": ROUTE_CRISIS,
            "reason": (
                "Crisis language was detected, so no generated text is used at "
                "all and the fixed safe-fallback message is returned."
            ),
            "current_valence": current_valence,
            "forecast_valence": forecast_valence,
        }

    if current_valence == "negative" or forecast_valence == "negative":
        driver = "current emotion" if current_valence == "negative" else "forecast"
        return {
            "route": ROUTE_ADAPTER,
            "reason": (
                f"The {driver} is negative, which is the distribution the ESConv "
                "adapter was fine-tuned on, so the adapter is enabled."
            ),
            "current_valence": current_valence,
            "forecast_valence": forecast_valence,
        }

    return {
        "route": ROUTE_BASE,
        "reason": (
            f"Current emotion is {current_valence} and the forecast is "
            f"{forecast_valence}. ESConv contains no positive conversations, so "
            "the adapter is disabled and the base model answers instead."
        ),
        "current_valence": current_valence,
        "forecast_valence": forecast_valence,
    }


def format_dialogue(history: Sequence[Tuple[str, str]], max_turns: int) -> str:
    """Render prior turns in the exact `User:` / `Assistant:` shape used in training.

    `preprocess.py` wrote seeker turns as `User:` and supporter turns as
    `Assistant:`; anything else here would be a train/inference mismatch.
    """
    selected = list(history)[-max_turns:] if max_turns > 0 else list(history)
    lines = []
    for speaker, text in selected:
        if not str(text).strip():
            continue
        display = "User" if speaker in ("user", "seeker") else "Assistant"
        lines.append(f"{display}: {str(text).strip()}")
    return "\n".join(lines)
