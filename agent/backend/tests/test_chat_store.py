from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from chat_store import DEFAULT_SESSION_ID, ChatStore, normalize_session_id


class ChatStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = ChatStore(":memory:")
        self.addCleanup(self.store.close)

    def _exchange(self, session_id: str, user_text: str, reply: str) -> None:
        self.store.append(session_id, "user", user_text)
        self.store.append(session_id, "assistant", reply)

    def test_history_is_returned_oldest_first(self) -> None:
        self._exchange("s1", "first", "reply one")
        self._exchange("s1", "second", "reply two")
        self.assertEqual(
            self.store.history("s1"),
            [
                ("user", "first"),
                ("assistant", "reply one"),
                ("user", "second"),
                ("assistant", "reply two"),
            ],
        )

    def test_limit_keeps_the_newest_messages(self) -> None:
        self._exchange("s1", "first", "reply one")
        self._exchange("s1", "second", "reply two")
        self.assertEqual(
            self.store.history("s1", limit=2),
            [("user", "second"), ("assistant", "reply two")],
        )
        self.assertEqual(self.store.history("s1", limit=0), [])

    def test_sessions_do_not_see_each_other(self) -> None:
        self._exchange("s1", "mine", "yours")
        self._exchange("s2", "theirs", "ours")
        self.assertEqual(
            self.store.history("s1"), [("user", "mine"), ("assistant", "yours")]
        )
        self.assertEqual(
            self.store.history("s2"), [("user", "theirs"), ("assistant", "ours")]
        )

    def test_reset_clears_only_the_named_session(self) -> None:
        self._exchange("s1", "mine", "yours")
        self._exchange("s2", "theirs", "ours")
        self.store.set_previous_emotion("s1", "sadness")

        self.store.reset("s1")

        self.assertEqual(self.store.history("s1"), [])
        self.assertIsNone(self.store.previous_emotion("s1"))
        self.assertEqual(len(self.store.history("s2")), 2)

    def test_previous_emotion_round_trips_and_survives_a_crisis_turn(self) -> None:
        self.assertIsNone(self.store.previous_emotion("s1"))
        self.store.set_previous_emotion("s1", "sadness")
        self.assertEqual(self.store.previous_emotion("s1"), "sadness")
        # A crisis turn short-circuits before the classifier runs and reports no
        # label; the baseline has to stay put rather than being wiped.
        self.store.set_previous_emotion("s1", None)
        self.assertEqual(self.store.previous_emotion("s1"), "sadness")

    def test_max_messages_trims_the_oldest(self) -> None:
        store = ChatStore(":memory:", max_messages=4)
        self.addCleanup(store.close)
        for index in range(5):
            store.append("s1", "user", f"message {index}")
        self.assertEqual(
            store.history("s1"),
            [("user", f"message {index}") for index in range(1, 5)],
        )

    def test_blank_and_missing_session_ids_share_the_default(self) -> None:
        self.assertEqual(normalize_session_id(None), DEFAULT_SESSION_ID)
        self.assertEqual(normalize_session_id("   "), DEFAULT_SESSION_ID)
        self.assertEqual(normalize_session_id(" abc "), "abc")

        self.store.append(None, "user", "hello")
        self.assertEqual(self.store.history(DEFAULT_SESSION_ID), [("user", "hello")])

    def test_empty_messages_are_not_recorded(self) -> None:
        self.store.append("s1", "user", "   ")
        self.assertEqual(self.store.history("s1"), [])

    def test_unknown_role_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.store.append("s1", "system", "nope")

    def test_purge_expired_drops_stale_sessions(self) -> None:
        store = ChatStore(":memory:", ttl_seconds=0.05)
        self.addCleanup(store.close)
        store.append("old", "user", "long ago")
        time.sleep(0.1)
        store.append("fresh", "user", "just now")

        # The write above already purges, so "old" is gone by now.
        self.assertEqual(store.history("old"), [])
        self.assertEqual(store.history("fresh"), [("user", "just now")])
        self.assertEqual(store.purge_expired(), 0)

    def test_history_survives_reopening_the_database_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chat.db"
            first = ChatStore(path)
            first.append("s1", "user", "before the restart")
            first.set_previous_emotion("s1", "joy")
            first.close()

            second = ChatStore(path)
            try:
                self.assertEqual(
                    second.history("s1"), [("user", "before the restart")]
                )
                self.assertEqual(second.previous_emotion("s1"), "joy")
            finally:
                # Windows will not remove the temporary directory while the
                # database file is still open, so close before leaving the
                # `with` block rather than in addCleanup.
                second.close()


if __name__ == "__main__":
    unittest.main()
