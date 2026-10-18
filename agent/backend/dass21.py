"""Adult DASS-21, past-week self-report; separate from live stress inference.

English items and scoring follow the supplied DASS-21 form (Lovibond &
Lovibond, 1995). Scores are symptom-severity measures, not probabilities or
diagnoses. No reverse scoring applies to this instrument.
"""

from typing import Mapping

VERSION = "dass21-adult-en-v1"
REASSESS_AFTER_SECONDS = 7 * 86_400
ITEMS = (
    "I found it hard to wind down",
    "I was aware of dryness of my mouth",
    "I couldn't seem to experience any positive feeling at all",
    "I experienced breathing difficulty (e.g. excessively rapid breathing, breathlessness in the absence of physical exertion)",
    "I found it difficult to work up the initiative to do things",
    "I tended to over-react to situations",
    "I experienced trembling (e.g. in the hands)",
    "I felt that I was using a lot of nervous energy",
    "I was worried about situations in which I might panic and make a fool of myself",
    "I felt that I had nothing to look forward to",
    "I found myself getting agitated",
    "I found it difficult to relax",
    "I felt down-hearted and blue",
    "I was intolerant of anything that kept me from getting on with what I was doing",
    "I felt I was close to panic",
    "I was unable to become enthusiastic about anything",
    "I felt I wasn't worth much as a person",
    "I felt that I was rather touchy",
    "I was aware of the action of my heart in the absence of physical exertion (e.g. sense of heart rate increase, heart missing a beat)",
    "I felt scared without any good reason",
    "I felt that life was meaningless",
)
OPTIONS = (
    "Did not apply to me at all",
    "Applied to me to some degree, or some of the time",
    "Applied to me to a considerable degree or a good part of time",
    "Applied to me very much or most of the time",
)
SCALE_ITEMS = {
    "depression": (3, 5, 10, 13, 16, 17, 21),
    "anxiety": (2, 4, 7, 9, 15, 19, 20),
    "stress": (1, 6, 8, 11, 12, 14, 18),
}
UPPER_BOUNDS = {
    "depression": (9, 13, 20, 27),
    "anxiety": (7, 9, 14, 19),
    "stress": (14, 18, 25, 33),
}
SEVERITIES = ("Normal", "Mild", "Moderate", "Severe", "Extremely Severe")


def form_definition() -> dict:
    return {
        "version": VERSION,
        "title": "DASS-21",
        "timeframe": "Over the past week",
        "instructions": "Please indicate how much each statement applied to you over the past week. There are no right or wrong answers. Do not spend too much time on any statement.",
        "questions": [{"id": str(i), "text": text} for i, text in enumerate(ITEMS, 1)],
        "options": [{"value": i, "label": label} for i, label in enumerate(OPTIONS)],
    }


def severity_for(scale: str, score: int) -> str:
    if scale not in UPPER_BOUNDS or type(score) is not int or not 0 <= score <= 42:
        raise ValueError("DASS-21 scale scores must be integers from 0 to 42")
    for upper, label in zip(UPPER_BOUNDS[scale], SEVERITIES):
        if score <= upper:
            return label
    return SEVERITIES[-1]


def score_answers(answers: Mapping[str, int]) -> dict:
    if set(answers) != {str(i) for i in range(1, 22)}:
        raise ValueError("Answer all 21 DASS-21 questions using their original question numbers")
    if any(type(value) is not int or not 0 <= value <= 3 for value in answers.values()):
        raise ValueError("Each DASS-21 answer must be an integer from 0 to 3")
    scores = {}
    for scale, items in SCALE_ITEMS.items():
        raw = sum(answers[str(i)] for i in items)
        scores[scale] = {"raw_score": raw, "score": raw * 2,
                         "severity": severity_for(scale, raw * 2)}
    return scores
