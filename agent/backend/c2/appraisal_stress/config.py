from dataclasses import dataclass

RAW_FEATURES = ["arousal", "dominance", "valence"]

# Reference-derived uncertainty threshold:
# samples below the 10th percentile of reference confidence are flagged uncertain.
REFERENCE_CONFIDENCE_QUANTILE = 0.10

@dataclass
class MembershipReference:
    """
    Quantile-based fuzzy membership thresholds in normalized z-score space.
    """

    arousal_q33: float
    arousal_q67: float

    dominance_q33: float
    dominance_q67: float

    valence_q33: float
    valence_q67: float

    arousal_scale: float
    dominance_scale: float
    valence_scale: float
