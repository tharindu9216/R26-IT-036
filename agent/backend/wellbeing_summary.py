"""Per-patient wellbeing summary over days or weeks, for the doctor portal.

Chat sessions expire after a day (``chat_history.ttl_seconds``), which is far
too short for "how was this patient's week". So every completed turn also
leaves a small numeric record -- ``turn_metrics`` -- in the chat store: the
stress readings each model produced, the emotion label, and the safety flags.
No message text is kept there.

``summarize`` turns those records into what the summary view draws. Each stress
source is averaged on its own, never blended into one invented score: the
check-in/wearable fusion, the written-text model (C3), the voice model (C2) and
the camera measure different things on different populations.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta
from typing import Any, Iterable, Mapping

from emotion_chain.core import NEGATIVE_EMOTION_FAMILIES, emotion_family

# (key, label) of every stress reading recorded per turn. ``combined`` is the
# app's own 60% check-in + 40% wearable fusion (multimodal_stress.py).
STRESS_SOURCES = (
    ("combined", "Check-in + wearable"),
    ("questionnaire", "Check-in answers"),
    ("sensor", "Wearable"),
    ("text", "Written words (C3)"),
    ("voice", "Voice tone (C2)"),
    ("face", "Facial expression"),
)


def _probability(state: Mapping[str, Any], key: str, available_key: str | None) -> float | None:
    if available_key is not None and not state.get(available_key):
        return None
    value = state.get(key)
    return None if value is None else round(float(value), 4)


def turn_metrics(result: Mapping[str, Any], mode: str) -> dict[str, Any]:
    """The numbers worth keeping from one finished graph run.

    A source that did not run this turn is stored as None, never as 0 -- a
    missing wearable is not a calm one.
    """

    return {
        "mode": mode,
        "stress": {
            "combined": _probability(
                result, "multimodal_stress_probability", "multimodal_stress_available"
            ),
            "questionnaire": _probability(
                result, "questionnaire_stress_probability", "questionnaire_stress_available"
            ),
            "sensor": _probability(
                result, "sensor_stress_probability", "sensor_stress_available"
            ),
            "text": _probability(result, "text_signal_probability", None)
            if mode == "text"
            else None,
            "voice": _probability(result, "voice_stress_score", None)
            if mode == "voice"
            else None,
            "face": _probability(result, "face_stress_probability", "face_stress_available"),
        },
        "current_emotion": result.get("current_emotion"),
        "next_intensity": result.get("next_intensity"),
        "deviation_level": result.get("deviation_level"),
        "route": result.get("reply_route"),
        "crisis": result.get("safety_status") == "crisis",
        "contacts_offered": bool(result.get("support_contacts")),
        "out_of_scope": result.get("scope_status") == "out_of_scope",
    }


def _mean(values: Iterable[float | None]) -> float | None:
    kept = [value for value in values if value is not None]
    return round(sum(kept) / len(kept), 4) if kept else None


def _is_negative(label: str | None) -> bool:
    return bool(label) and emotion_family(str(label)) in NEGATIVE_EMOTION_FAMILIES


def _source_averages(rows: list[dict]) -> dict[str, float | None]:
    return {
        key: _mean(row["metrics"]["stress"].get(key) for row in rows)
        for key, _ in STRESS_SOURCES
    }


def summarize(
    rows: Iterable[Mapping[str, Any]],
    *,
    days: int,
    today: date | None = None,
) -> dict[str, Any]:
    """Aggregate ``turn_metrics`` rows into the summary view's payload.

    ``rows`` are ``{"created_at": epoch seconds, "metrics": {...}}`` covering
    the period and the one before it (for the change figures). Days are
    calendar days in the server's local time zone, ending today.
    """

    days = max(1, int(days))
    today = today or date.today()
    start = today - timedelta(days=days - 1)
    previous_start = start - timedelta(days=days)

    current: list[dict] = []
    previous: list[dict] = []
    for row in rows:
        day = datetime.fromtimestamp(float(row["created_at"])).date()
        record = {"day": day, "created_at": float(row["created_at"]), "metrics": row["metrics"]}
        if start <= day <= today:
            current.append(record)
        elif previous_start <= day < start:
            previous.append(record)
    current.sort(key=lambda record: record["created_at"])

    by_day: dict[date, list[dict]] = {}
    for record in current:
        by_day.setdefault(record["day"], []).append(record)

    daily = []
    for offset in range(days):
        day = start + timedelta(days=offset)
        records = by_day.get(day, [])
        emotions = Counter(
            r["metrics"].get("current_emotion")
            for r in records
            if r["metrics"].get("current_emotion")
        )
        daily.append(
            {
                "date": day.isoformat(),
                "check_ins": len(records),
                "stress": _source_averages(records),
                "dominant_emotion": emotions.most_common(1)[0][0] if emotions else None,
                "crisis": sum(1 for r in records if r["metrics"].get("crisis")),
                "contacts_offered": sum(
                    1 for r in records if r["metrics"].get("contacts_offered")
                ),
            }
        )

    current_averages = _source_averages(current)
    previous_averages = _source_averages(previous)
    sources = []
    for key, label in STRESS_SOURCES:
        average = current_averages[key]
        before = previous_averages[key]
        sources.append(
            {
                "key": key,
                "label": label,
                "average": average,
                "previous_average": before,
                "change": None
                if average is None or before is None
                else round(average - before, 4),
                "readings": sum(
                    1 for r in current if r["metrics"]["stress"].get(key) is not None
                ),
            }
        )

    emotion_counts = Counter(
        r["metrics"].get("current_emotion")
        for r in current
        if r["metrics"].get("current_emotion")
    )
    labelled = sum(emotion_counts.values())
    emotions = [
        {"label": label, "count": count, "share": round(count / labelled, 4)}
        for label, count in emotion_counts.most_common()
    ]
    negative = sum(1 for r in current if _is_negative(r["metrics"].get("current_emotion")))
    previous_labelled = [r for r in previous if r["metrics"].get("current_emotion")]
    previous_negative = sum(
        1 for r in previous_labelled if _is_negative(r["metrics"].get("current_emotion"))
    )

    events = [
        {
            "time": datetime.fromtimestamp(r["created_at"]).isoformat(timespec="minutes"),
            "mode": r["metrics"].get("mode"),
            "kind": "crisis" if r["metrics"].get("crisis") else "contacts_offered",
        }
        for r in current
        if r["metrics"].get("crisis") or r["metrics"].get("contacts_offered")
    ]

    return {
        "days": days,
        "start": start.isoformat(),
        "end": today.isoformat(),
        "totals": {
            "check_ins": len(current),
            "previous_check_ins": len(previous),
            "text": sum(1 for r in current if r["metrics"].get("mode") == "text"),
            "voice": sum(1 for r in current if r["metrics"].get("mode") == "voice"),
            "active_days": len(by_day),
            "crisis": sum(1 for r in current if r["metrics"].get("crisis")),
            "contacts_offered": sum(
                1 for r in current if r["metrics"].get("contacts_offered")
            ),
            "high_deviation": sum(
                1 for r in current if r["metrics"].get("deviation_level") == "High"
            ),
            "high_intensity_forecasts": sum(
                1 for r in current if r["metrics"].get("next_intensity") == "high"
            ),
            "out_of_scope": sum(1 for r in current if r["metrics"].get("out_of_scope")),
        },
        "negative_share": round(negative / labelled, 4) if labelled else None,
        "previous_negative_share": round(previous_negative / len(previous_labelled), 4)
        if previous_labelled
        else None,
        "sources": sources,
        "daily": daily,
        "emotions": emotions,
        "events": events,
    }
