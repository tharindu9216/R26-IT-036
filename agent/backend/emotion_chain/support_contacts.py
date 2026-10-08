"""Human support contacts, and the rule that decides when to surface them.

A phone number is the one part of a reply that must never be generated. Qwen
invents plausible-looking hotline numbers when a prompt nudges it towards one,
so the number lives here as data, the escalation rule below decides whether the
turn warrants it, and the sentence is appended to the finished reply rather than
asked for. The system prompt tells the model not to produce numbers of its own.

Two things trigger it. The safety node's crisis branch always does -- that reply
is a fixed template and the contact belongs in it. Otherwise the user must be in
a negative emotion AND at least ``MIN_ESCALATION_SIGNALS`` of these must agree:

* the next-intensity forecaster expects the *high* intensity state of that
  emotion next (``high_sadness`` / ``high_anger`` / ``high_fear``);
* the deviation tracker scored the move from the previous turn "High", i.e. the
  user crossed from positive to negative in one turn;
* the C1 wearable reports stress;
* the fused C3 Stress+CBT text signal is flagged;
* the C2 voice appraisal reports stress.

Requiring two is the whole design, and the forecast clause is why. The
forecaster genuinely predicts intensity rather than restating a prior -- that
is the point of replacing the 13-state model, whose own metadata recorded that
text did not separate its reachable next states -- but it predicts it weakly:
ROC-AUC 0.58 on the held-out split, recalling 9% of the high-intensity turns,
against AI pseudo-labels rather than human gold ones. That is worth one
agreeing signal and no more. "I feel overwhelmed today" classifies as sadness
and can forecast ``high_sadness``; an ordinary hard day would otherwise be
handed a phone number, which is both wrong and the fastest way to teach someone
to ignore the one that matters.

The same logic applies to the others taken alone: a C3 stress flag is common in
ordinary venting, and one positive-to-negative turn is a normal shape for a
conversation. Two independent signals is the point at which the turn stops
looking like an ordinary bad day.

This is a routing aid built from classifier output. It is not a validated risk
assessment, and the UI text should not present it as one.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .core import NEGATIVE_EMOTION_FAMILIES, emotion_family


@dataclass(frozen=True)
class SupportContact:
    """One person or service the user can be handed at an extreme turn."""

    name: str
    phone: str
    note: str = ""
    # What the TTS voice should say instead of ``name``. Kokoro runs an
    # English voice, so a Sinhala name has to be handed to it transliterated
    # or it is dropped from the spoken reply. Empty means ``name`` is already
    # speakable.
    spoken_name: str = ""

    @property
    def spoken(self) -> str:
        return self.spoken_name or self.name

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "phone": self.phone, "note": self.note}


# Sinhala name first, transliteration after it, so the line still reads on a
# terminal or a device without a Sinhala font.
SUMITHURO_CONTACT = SupportContact(
    name="සුමිතුරෝ (Sumithuro)",
    phone="077766889",
    note="Emotional support contact",
    spoken_name="Sumithuro",
)

DEFAULT_SUPPORT_CONTACTS: tuple[SupportContact, ...] = (SUMITHURO_CONTACT,)

# The high-intensity negative states, i.e. a "high" forecast on a turn whose
# current emotion is anger, fear or sadness.
EXTREME_NEGATIVE_STATES = frozenset({"high_anger", "high_fear", "high_sadness"})

# How many of the signals below must agree before contacts are offered. Two,
# because every one of them fires on ordinary bad days on its own.
MIN_ESCALATION_SIGNALS = 2


def escalation_signals(
    current_emotion: str,
    next_emotion: str,
    *,
    deviation_level: str = "None",
    sensor_stress: bool = False,
    text_signal_flagged: bool = False,
    voice_stress: bool = False,
    multimodal_stress: bool = False,
) -> tuple[str, ...]:
    """Name every signal agreeing that this turn is extreme.

    Returns the reasons rather than a count so a decision can be explained --
    "why did this turn get a phone number" is a question worth being able to
    answer. An empty tuple for a non-negative current emotion is deliberate:
    the other signals only mean escalation relative to distress.
    """

    if emotion_family(current_emotion) not in NEGATIVE_EMOTION_FAMILIES:
        return ()

    reasons = []
    if next_emotion.strip().lower() in EXTREME_NEGATIVE_STATES:
        reasons.append("high_intensity_forecast")
    if deviation_level.strip().capitalize() == "High":
        reasons.append("high_deviation")
    if sensor_stress:
        reasons.append("sensor_stress")
    if text_signal_flagged:
        reasons.append("text_signal")
    if voice_stress:
        reasons.append("voice_stress")
    if multimodal_stress:
        reasons.append("multimodal_stress")
    return tuple(reasons)


def is_extremely_negative(
    current_emotion: str,
    next_emotion: str,
    *,
    deviation_level: str = "None",
    safety_status: str = "normal",
    sensor_stress: bool = False,
    text_signal_flagged: bool = False,
    voice_stress: bool = False,
    multimodal_stress: bool = False,
) -> bool:
    """Decide whether this turn is extreme enough to offer a human contact."""

    if safety_status.strip().lower() == "crisis":
        return True

    signals = escalation_signals(
        current_emotion,
        next_emotion,
        deviation_level=deviation_level,
        sensor_stress=sensor_stress,
        text_signal_flagged=text_signal_flagged,
        voice_stress=voice_stress,
        multimodal_stress=multimodal_stress,
    )
    return len(signals) >= MIN_ESCALATION_SIGNALS


def contact_message(
    contacts: Sequence[SupportContact] = DEFAULT_SUPPORT_CONTACTS,
) -> str:
    """One sentence naming every configured contact, or "" when there is none."""

    listed = " or ".join(f"{contact.name} on {contact.phone}" for contact in contacts)
    if not listed:
        return ""
    return (
        "You do not have to carry this alone. If you would like to talk to a "
        f"person right now, you can contact {listed}."
    )


def append_contact_message(
    reply: str,
    contacts: Sequence[SupportContact] = DEFAULT_SUPPORT_CONTACTS,
) -> str:
    """Append the contact sentence unless the reply already carries the number.

    Idempotent on purpose: the crisis template already names the contacts, and
    a generated reply that happens to repeat the number should not get it
    twice.
    """

    message = contact_message(contacts)
    if not message:
        return reply
    if any(contact.phone in reply for contact in contacts):
        return reply
    return f"{reply.rstrip()}\n\n{message}"


def speakable(
    reply: str,
    contacts: Sequence[SupportContact] = DEFAULT_SUPPORT_CONTACTS,
) -> str:
    """Rewrite contact details for the TTS voice before synthesis.

    Voice mode reads the reply out loud, contact sentence included. Kokoro is
    an English voice and silently drops a Sinhala name, which would leave a
    spoken reply offering a phone number belonging to nobody. Phone numbers
    are also expanded digit by digit so Kokoro does not pronounce them as one
    large number. The displayed text stays untouched; only the audio uses
    these rewrites.
    """

    digit_names = {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
    }
    for contact in contacts:
        if contact.spoken != contact.name:
            reply = reply.replace(contact.name, contact.spoken)
        if contact.phone and contact.phone in reply:
            spoken_phone = ", ".join(
                digit_names.get(character, character) for character in contact.phone
            )
            reply = reply.replace(contact.phone, spoken_phone)
    return reply


def as_dicts(contacts: Iterable[SupportContact]) -> list[dict[str, str]]:
    """JSON-serializable form for the API and the LangGraph state."""

    return [contact.as_dict() for contact in contacts]
