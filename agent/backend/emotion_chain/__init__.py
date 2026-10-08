"""Current emotion, next-intensity forecast and supportive reply generation."""

from .core import (
    CURRENT_EMOTIONS,
    INTENSITY_LABELS,
    INTENSITY_STATES,
    STATE_LABELS,
    STATE_TRANSITIONS,
    allowed_next_states,
    constrain_probability_map,
    emotion_family,
    intensity_probability_map,
    intensity_state,
    normalize_intensity,
    select_reply_route,
    tokenize_state_text,
)
from .deviation_tracker import EmotionDeviation, compute_deviation, valence_group
from .fusion import FusedTextSignal, fuse_stress_and_cbt
from .support_contacts import (
    DEFAULT_SUPPORT_CONTACTS,
    SupportContact,
    append_contact_message,
    contact_message,
    is_extremely_negative,
)

__all__ = [
    "CURRENT_EMOTIONS",
    "DEFAULT_SUPPORT_CONTACTS",
    "INTENSITY_LABELS",
    "INTENSITY_STATES",
    "STATE_LABELS",
    "STATE_TRANSITIONS",
    "EmotionDeviation",
    "FusedTextSignal",
    "SupportContact",
    "allowed_next_states",
    "append_contact_message",
    "compute_deviation",
    "constrain_probability_map",
    "contact_message",
    "emotion_family",
    "fuse_stress_and_cbt",
    "intensity_probability_map",
    "intensity_state",
    "is_extremely_negative",
    "normalize_intensity",
    "select_reply_route",
    "tokenize_state_text",
    "valence_group",
]
