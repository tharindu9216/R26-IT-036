from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from .fusion.decision import decide_fusion

from .config import DIARY_DB_PATH


@contextmanager
def _connect():
    DIARY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DIARY_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS diary_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL,
                stress_label TEXT NOT NULL,
                stress_is_stressed INTEGER NOT NULL,
                stress_confidence REAL NOT NULL,
                stress_probabilities TEXT NOT NULL,
                distortion_label TEXT NOT NULL,
                distortion_has_distortion INTEGER NOT NULL,
                distortion_confidence REAL NOT NULL,
                distortion_probabilities TEXT NOT NULL,
                topic_theme TEXT,
                topic_similarity REAL,
                topic_terms TEXT
            )
            """
        )


def _row_to_entry(row: sqlite3.Row) -> dict:
    is_stressed = bool(row["stress_is_stressed"])
    has_distortion = bool(row["distortion_has_distortion"])
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "title": row["title"],
        "content": row["content"],
        "stress": {
            "label": row["stress_label"],
            "is_stressed": is_stressed,
            "confidence": row["stress_confidence"],
            "probabilities": json.loads(row["stress_probabilities"]),
        },
        "distortion": {
            "label": row["distortion_label"],
            "has_distortion": has_distortion,
            "confidence": row["distortion_confidence"],
            "probabilities": json.loads(row["distortion_probabilities"]),
        },
        "topic": (
            {
                "theme": row["topic_theme"],
                "similarity": row["topic_similarity"],
                "terms": json.loads(row["topic_terms"]),
            }
            if row["topic_theme"] is not None
            else None
        ),
        # Derived rather than stored so historical diary rows immediately gain
        # the combined state without a schema migration.
        "fusion": decide_fusion(is_stressed, has_distortion).as_dict(),
    }


def insert_entry(
    content: str,
    title: str,
    stress_label: str,
    stress_is_stressed: bool,
    stress_confidence: float,
    stress_probabilities: dict,
    distortion_label: str,
    distortion_has_distortion: bool,
    distortion_confidence: float,
    distortion_probabilities: dict,
    topic_theme: str | None = None,
    topic_similarity: float | None = None,
    topic_terms: list[str] | None = None,
) -> dict:
    created_at = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO diary_entries (
                created_at, title, content,
                stress_label, stress_is_stressed, stress_confidence, stress_probabilities,
                distortion_label, distortion_has_distortion, distortion_confidence, distortion_probabilities,
                topic_theme, topic_similarity, topic_terms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                title,
                content,
                stress_label,
                int(stress_is_stressed),
                stress_confidence,
                json.dumps(stress_probabilities),
                distortion_label,
                int(distortion_has_distortion),
                distortion_confidence,
                json.dumps(distortion_probabilities),
                topic_theme,
                topic_similarity,
                json.dumps(topic_terms) if topic_terms is not None else None,
            ),
        )
        entry_id = cursor.lastrowid
        row = conn.execute(
            "SELECT * FROM diary_entries WHERE id = ?", (entry_id,)
        ).fetchone()
        return _row_to_entry(row)


def list_entries(limit: int = 50, offset: int = 0) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM diary_entries ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [_row_to_entry(row) for row in rows]


def get_entry(entry_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM diary_entries WHERE id = ?", (entry_id,)
        ).fetchone()
        return _row_to_entry(row) if row else None


def delete_entry(entry_id: int) -> bool:
    with _connect() as conn:
        cursor = conn.execute("DELETE FROM diary_entries WHERE id = ?", (entry_id,))
        return cursor.rowcount > 0


def summary(trend_days: int = 14) -> dict:
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM diary_entries").fetchone()["n"]
        stressed = conn.execute(
            "SELECT COUNT(*) AS n FROM diary_entries WHERE stress_is_stressed = 1"
        ).fetchone()["n"]
        distorted = conn.execute(
            "SELECT COUNT(*) AS n FROM diary_entries WHERE distortion_has_distortion = 1"
        ).fetchone()["n"]

        since = (datetime.now(timezone.utc) - timedelta(days=trend_days - 1)).date()
        rows = conn.execute(
            """
            SELECT substr(created_at, 1, 10) AS day,
                   COUNT(*) AS total,
                   SUM(stress_is_stressed) AS stressed,
                   SUM(distortion_has_distortion) AS distorted
            FROM diary_entries
            WHERE substr(created_at, 1, 10) >= ?
            GROUP BY day
            ORDER BY day ASC
            """,
            (since.isoformat(),),
        ).fetchall()
        by_day = {row["day"]: row for row in rows}

        trend = []
        for offset in range(trend_days):
            day = (since + timedelta(days=offset)).isoformat()
            row = by_day.get(day)
            trend.append(
                {
                    "date": day,
                    "total": row["total"] if row else 0,
                    "stressed": row["stressed"] if row else 0,
                    "distorted": row["distorted"] if row else 0,
                }
            )

    return {
        "total_entries": total,
        "stressed_count": stressed,
        "stressed_pct": (stressed / total * 100) if total else 0.0,
        "distortion_count": distorted,
        "distortion_pct": (distorted / total * 100) if total else 0.0,
        "trend": trend,
    }
