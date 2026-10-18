"""Per-session chat history for the API, kept in a small SQLite file.

The supportive graph is stateless between invocations: every turn is scored on
its own, and whatever earlier conversation should influence the reply has to be
handed back in by the caller (see ``reply_generator._messages``). This module is
where the API keeps that conversation.

SQLite from the standard library rather than a service: it adds no dependency
and no process to start, the whole store is one inspectable file next to
``config.yaml``, and it survives the ``uvicorn --reload`` restarts that a
process-global list does not. Sessions are keyed by an opaque id the client
generates, so two browser tabs are two conversations rather than one shared one.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id       TEXT PRIMARY KEY,
    previous_emotion TEXT,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    text       TEXT NOT NULL,
    -- JSON record of what each pipeline component returned for this turn, set
    -- on the assistant message only. See emotion_chain/trace.py.
    trace      TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_by_session ON messages (session_id, id);
-- What a turn's XAI explanation needs (sensor windows, check-in answers, voice
-- appraisal, ...), serialized by api.py. Kept here rather than only in memory
-- so "Explain predictions" still works after the backend restarts.
CREATE TABLE IF NOT EXISTS turn_contexts (
    turn_id    TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    context    BLOB NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS turn_contexts_by_session ON turn_contexts (session_id, created_at);
-- Numbers only (stress readings, emotion label, safety flags; no message text)
-- per completed turn, for the doctor's multi-day summary. Deliberately outside
-- the session TTL, New chat and the message cap: it is the patient's wellbeing
-- history, not the conversation. Aged out by ``purge_metrics`` instead.
CREATE TABLE IF NOT EXISTS turn_metrics (
    turn_id    TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    metrics    TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS turn_metrics_by_session ON turn_metrics (session_id, created_at);
-- Assessments are clinical-review history, independent of chat TTL/New chat.
CREATE TABLE IF NOT EXISTS dass21_assessments (
    assessment_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    version TEXT NOT NULL,
    answers TEXT NOT NULL,
    scores TEXT NOT NULL,
    completed_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS dass21_by_patient ON dass21_assessments (session_id, completed_at);
"""

DEFAULT_SESSION_ID = "default"


def normalize_session_id(value: str | None) -> str:
    """Accept a missing or blank id so an older client still works."""

    normalized = (value or "").strip()
    return normalized or DEFAULT_SESSION_ID


class ChatStore:
    """Append-only turn log per session, trimmed by count and by age.

    One connection shared under a lock rather than a connection per request:
    the API is threaded but the traffic is a handful of writes per turn, and a
    single connection keeps the WAL file and the trimming logic in one place.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        max_messages: int = 200,
        ttl_seconds: float = 86_400.0,
    ) -> None:
        self.db_path = str(db_path)
        self.max_messages = max(2, int(max_messages))
        self.ttl_seconds = float(ttl_seconds)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        with self._lock:
            # WAL lets the history reads run while a turn is being written.
            # ":memory:" databases reject it, which is fine -- they are only
            # used by the tests, where there is nothing to concurrently read.
            try:
                self._connection.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                pass
            self._connection.executescript(SCHEMA)
            self._migrate()
            self._connection.commit()

    def _migrate(self) -> None:
        """Add columns introduced after a database file was first created."""

        columns = {
            row[1]
            for row in self._connection.execute("PRAGMA table_info(messages)")
        }
        if "trace" not in columns:
            self._connection.execute("ALTER TABLE messages ADD COLUMN trace TEXT")

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def save_dass21(self, session_id: str, version: str, answers: dict, scores: dict) -> dict:
        assessment_id, completed_at = uuid.uuid4().hex, time.time()
        with self._lock:
            self._connection.execute(
                "INSERT INTO dass21_assessments VALUES (?, ?, ?, ?, ?, ?)",
                (assessment_id, normalize_session_id(session_id), version,
                 json.dumps(answers), json.dumps(scores), completed_at),
            )
            self._connection.commit()
        return {"assessment_id": assessment_id, "version": version, "answers": answers,
                "scores": scores, "completed_at": completed_at}

    def dass21_history(self, session_id: str, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT assessment_id, version, answers, scores, completed_at "
                "FROM dass21_assessments WHERE session_id = ? "
                "ORDER BY completed_at DESC, rowid DESC LIMIT ?",
                (normalize_session_id(session_id), max(1, min(limit, 100))),
            ).fetchall()
        return [{"assessment_id": row[0], "version": row[1], "answers": json.loads(row[2]),
                 "scores": json.loads(row[3]), "completed_at": row[4]} for row in rows]

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def append(
        self,
        session_id: str,
        role: str,
        text: str,
        trace: dict[str, Any] | None = None,
    ) -> None:
        """Record one message and age out whatever no longer belongs.

        ``trace`` is the pipeline record for an assistant message. It is stored
        with the turn so the history drawer can still show how a reply was
        produced after a reload, when the response that carried it is long gone.
        """

        if role not in {"user", "assistant"}:
            raise ValueError(f"role must be 'user' or 'assistant', got {role!r}")
        if not text.strip():
            return
        session_id = normalize_session_id(session_id)
        now = time.time()
        with self._lock:
            self._touch(session_id, now)
            self._connection.execute(
                "INSERT INTO messages (session_id, role, text, trace, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    session_id,
                    role,
                    text,
                    json.dumps(trace, separators=(",", ":")) if trace else None,
                    now,
                ),
            )
            self._trim(session_id)
            self._purge_expired(now)
            self._connection.commit()

    def save_turn_context(self, turn_id: str, session_id: str, context: bytes) -> None:
        session_id = normalize_session_id(session_id)
        now = time.time()
        with self._lock:
            self._touch(session_id, now)
            self._connection.execute(
                "INSERT OR REPLACE INTO turn_contexts (turn_id, session_id, context, created_at) "
                "VALUES (?, ?, ?, ?)",
                (turn_id, session_id, sqlite3.Binary(context), now),
            )
            self._trim(session_id)
            self._connection.commit()

    def record_metrics(
        self,
        turn_id: str,
        session_id: str,
        metrics: dict[str, Any],
        created_at: float | None = None,
    ) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO turn_metrics (turn_id, session_id, metrics, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    turn_id,
                    normalize_session_id(session_id),
                    json.dumps(metrics, separators=(",", ":")),
                    time.time() if created_at is None else float(created_at),
                ),
            )
            self._connection.commit()

    def purge_metrics(self, older_than_seconds: float) -> int:
        if older_than_seconds <= 0:
            return 0
        with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM turn_metrics WHERE created_at < ?",
                (time.time() - older_than_seconds,),
            )
            self._connection.commit()
        return cursor.rowcount

    def set_previous_emotion(self, session_id: str, label: str | None) -> None:
        """Carry this turn's emotion forward as the deviation baseline.

        A crisis turn short-circuits before the classifier runs and so has no
        label; leave the previous baseline in place rather than clearing it.
        """

        if not label:
            return
        session_id = normalize_session_id(session_id)
        now = time.time()
        with self._lock:
            self._touch(session_id, now)
            self._connection.execute(
                "UPDATE sessions SET previous_emotion = ? WHERE session_id = ?",
                (str(label), session_id),
            )
            self._connection.commit()

    def reset(self, session_id: str) -> None:
        """Start a fresh conversation: drop the turns and the baseline."""

        session_id = normalize_session_id(session_id)
        with self._lock:
            for table in ("messages", "turn_contexts", "sessions"):
                self._connection.execute(
                    f"DELETE FROM {table} WHERE session_id = ?", (session_id,)
                )
            self._connection.commit()

    def purge_expired(self) -> int:
        """Drop sessions untouched for longer than the TTL; return the count."""

        with self._lock:
            removed = self._purge_expired(time.time())
            self._connection.commit()
        return removed

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def history(self, session_id: str, limit: int | None = None) -> list[tuple[str, str]]:
        """The newest ``limit`` messages, oldest first.

        The ``(speaker, text)`` tuples are the shape ``QwenReplyGenerator``
        already replays into the prompt, so this drops straight into the graph
        state without any conversion.
        """

        session_id = normalize_session_id(session_id)
        window = self.max_messages if limit is None else max(0, int(limit))
        if window == 0:
            return []
        with self._lock:
            rows = self._connection.execute(
                "SELECT role, text FROM messages WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, window),
            ).fetchall()
        return [(role, text) for role, text in reversed(rows)]

    def transcript(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        """The same messages as ``history``, plus each turn's stored trace.

        Separate from ``history`` on purpose: that one feeds the prompt and has
        to keep returning bare ``(speaker, text)`` tuples, while this one feeds
        the UI and is free to grow.
        """

        session_id = normalize_session_id(session_id)
        window = self.max_messages if limit is None else max(0, int(limit))
        if window == 0:
            return []
        with self._lock:
            rows = self._connection.execute(
                "SELECT role, text, trace, created_at FROM messages WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, window),
            ).fetchall()

        messages = []
        for role, text, trace, created_at in reversed(rows):
            message: dict[str, Any] = {"role": role, "text": text, "created_at": created_at}
            if trace:
                try:
                    message["trace"] = json.loads(trace)
                except json.JSONDecodeError:
                    # A trace written by an older/newer shape should never cost
                    # the user the message itself.
                    pass
            messages.append(message)
        return messages

    def turn_context(self, turn_id: str) -> tuple[str, bytes] | None:
        """``(session_id, serialized context)`` for one turn, or None."""

        with self._lock:
            row = self._connection.execute(
                "SELECT session_id, context FROM turn_contexts WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
        return (row[0], bytes(row[1])) if row else None

    def turn_context_ids(self, session_id: str) -> set[str]:
        session_id = normalize_session_id(session_id)
        with self._lock:
            rows = self._connection.execute(
                "SELECT turn_id FROM turn_contexts WHERE session_id = ?", (session_id,)
            ).fetchall()
        return {row[0] for row in rows}

    def metrics(self, session_id: str, since: float) -> list[dict[str, Any]]:
        """Every turn_metrics row of a session from ``since`` on, oldest first."""

        with self._lock:
            rows = self._connection.execute(
                "SELECT metrics, created_at FROM turn_metrics "
                "WHERE session_id = ? AND created_at >= ? ORDER BY created_at",
                (normalize_session_id(session_id), float(since)),
            ).fetchall()
        result = []
        for metrics, created_at in rows:
            try:
                result.append({"metrics": json.loads(metrics), "created_at": created_at})
            except json.JSONDecodeError:
                continue
        return result

    def previous_emotion(self, session_id: str) -> str | None:
        session_id = normalize_session_id(session_id)
        with self._lock:
            row = self._connection.execute(
                "SELECT previous_emotion FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row[0] if row else None

    # ------------------------------------------------------------------
    # Internals -- all called with ``self._lock`` already held.
    # ------------------------------------------------------------------

    def _touch(self, session_id: str, now: float) -> None:
        self._connection.execute(
            "INSERT INTO sessions (session_id, previous_emotion, created_at, updated_at) "
            "VALUES (?, NULL, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET updated_at = excluded.updated_at",
            (session_id, now, now),
        )

    def _trim(self, session_id: str) -> None:
        self._connection.execute(
            "DELETE FROM messages WHERE session_id = ? AND id NOT IN ("
            "  SELECT id FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?"
            ")",
            (session_id, session_id, self.max_messages),
        )
        # One context per exchange, so half as many as messages.
        self._connection.execute(
            "DELETE FROM turn_contexts WHERE session_id = ? AND turn_id NOT IN ("
            "  SELECT turn_id FROM turn_contexts WHERE session_id = ? "
            "  ORDER BY created_at DESC LIMIT ?"
            ")",
            (session_id, session_id, max(1, self.max_messages // 2)),
        )

    def _purge_expired(self, now: float) -> int:
        if self.ttl_seconds <= 0:
            return 0
        cutoff = now - self.ttl_seconds
        stale = [
            row[0]
            for row in self._connection.execute(
                "SELECT session_id FROM sessions WHERE updated_at < ?", (cutoff,)
            ).fetchall()
        ]
        if not stale:
            return 0
        placeholders = ",".join("?" * len(stale))
        for table in ("messages", "turn_contexts", "sessions"):
            self._connection.execute(
                f"DELETE FROM {table} WHERE session_id IN ({placeholders})", stale
            )
        return len(stale)
