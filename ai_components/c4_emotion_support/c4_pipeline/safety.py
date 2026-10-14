"""Keyword safety screen.

Prototype-level, deliberately: this is a demonstrable guard rail, not a
clinical risk assessment.

Two fixes over the original substring list:

* **Morphology.** Literal `in` tests missed the most common phrasings --
  "ending my life" does not contain "end my life", and "killed myself" does not
  contain "kill myself". The patterns are regexes with inflection allowed.
* **False positives.** Bare "die" fired on "I'm dying to see it" and "the phone
  died". Every pattern now needs a first-person or explicit-intent context.

Recall still beats precision here: a false alarm costs one over-cautious reply,
a miss costs a crisis message answered as if it were ordinary.
"""

import re
from typing import Dict, List, Tuple

# (risk_type, pattern). \w* suffixes cover -s / -ed / -ing without listing them.
RISK_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("crisis_or_self_harm", re.compile(r"\bsuicid\w*", re.I)),
    ("crisis_or_self_harm", re.compile(r"\bkill\w*\s+(?:my\s?self|me)\b", re.I)),
    ("crisis_or_self_harm", re.compile(r"\bend\w*\s+(?:my|it)\s+(?:own\s+)?(?:life|all)\b", re.I)),
    ("crisis_or_self_harm", re.compile(r"\bend\w*\s+it\s+all\b", re.I)),
    ("crisis_or_self_harm", re.compile(r"\btak\w*\s+my\s+(?:own\s+)?life\b", re.I)),
    ("crisis_or_self_harm", re.compile(r"\bself[\s-]?harm\w*", re.I)),
    ("crisis_or_self_harm", re.compile(r"\b(?:hurt|harm|cut)\w*\s+my\s?self\b", re.I)),
    ("crisis_or_self_harm", re.compile(r"\b(?:want|wanna|going)\w*\s+to\s+die\b", re.I)),
    ("crisis_or_self_harm", re.compile(r"\b(?:wish|wished)\s+I\s+(?:was|were)\s+dead\b", re.I)),
    ("crisis_or_self_harm", re.compile(r"\bbetter\s+off\s+(?:dead|without\s+me)\b", re.I)),
    ("crisis_or_self_harm", re.compile(r"\bno\s+(?:reason|point)\s+(?:to|in)\s+liv\w*", re.I)),
    ("crisis_or_self_harm", re.compile(r"\bdon'?t\s+want\s+to\s+(?:live|be\s+here|exist)\b", re.I)),
    ("crisis_or_self_harm", re.compile(r"\boverdos\w*\b", re.I)),
    ("harm_to_others", re.compile(r"\b(?:kill|hurt|harm)\w*\s+(?:him|her|them|someone|everyone)\b", re.I)),
]


def check_safety(user_message: str) -> Dict[str, object]:
    """Screen a user turn for crisis language.

    Returns the original `risk_detected` / `risk_type` contract plus the
    phrases that matched, so the UI can show *why* the safe fallback fired
    rather than asserting it.
    """
    matched: List[str] = []
    risk_type = "none"
    for candidate_type, pattern in RISK_PATTERNS:
        found = pattern.search(user_message)
        if found:
            matched.append(found.group(0))
            if risk_type == "none":
                risk_type = candidate_type

    return {
        "risk_detected": bool(matched),
        "risk_type": risk_type,
        "matched_phrases": matched,
    }
