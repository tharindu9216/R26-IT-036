"""The doctor's multi-day summary: what each turn keeps, and how it adds up."""

from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta

from chat_store import ChatStore
from wellbeing_summary import summarize, turn_metrics

TODAY = date(2026, 10, 11)


def _at(day: date, hour: int = 12) -> float:
    return datetime(day.year, day.month, day.day, hour).timestamp()


def _row(days_ago: int, **stress) -> dict:
    metrics = turn_metrics({}, "text")
    metrics["stress"].update(stress)
    return {"created_at": _at(TODAY - timedelta(days=days_ago)), "metrics": metrics}


class TurnMetricsTests(unittest.TestCase):
    def test_sources_that_did_not_run_are_none_not_calm(self) -> None:
        metrics = turn_metrics(
            {
                "multimodal_stress_available": False,
                "multimodal_stress_probability": 0.0,
                "sensor_stress_available": False,
                "sensor_stress_probability": None,
                "text_signal_probability": 0.71,
                "voice_stress_score": 0.9,
                "current_emotion": "sadness",
                "safety_status": "normal",
                "support_contacts": [{"name": "x"}],
            },
            "text",
        )
        self.assertIsNone(metrics["stress"]["combined"])
        self.assertIsNone(metrics["stress"]["sensor"])
        self.assertEqual(metrics["stress"]["text"], 0.71)
        # Voice belongs to voice turns only.
        self.assertIsNone(metrics["stress"]["voice"])
        self.assertTrue(metrics["contacts_offered"])
        self.assertFalse(metrics["crisis"])

    def test_no_message_text_is_kept(self) -> None:
        metrics = turn_metrics({"user_text": "secret", "reply": "secret"}, "voice")
        self.assertNotIn("secret", repr(metrics))


class SummarizeTests(unittest.TestCase):
    def test_every_day_of_the_period_is_present(self) -> None:
        summary = summarize([_row(0, combined=0.6)], days=7, today=TODAY)
        self.assertEqual(len(summary["daily"]), 7)
        self.assertEqual(summary["daily"][0]["date"], "2026-10-05")
        self.assertEqual(summary["daily"][-1]["date"], "2026-10-11")
        self.assertEqual(summary["daily"][0]["check_ins"], 0)
        self.assertIsNone(summary["daily"][0]["stress"]["combined"])
        self.assertEqual(summary["totals"]["active_days"], 1)

    def test_sources_are_averaged_separately_and_compared(self) -> None:
        rows = [
            _row(0, combined=0.8, text=0.4),
            _row(1, combined=0.6),
            _row(8, combined=0.3),  # previous week
        ]
        summary = summarize(rows, days=7, today=TODAY)
        sources = {item["key"]: item for item in summary["sources"]}
        self.assertAlmostEqual(sources["combined"]["average"], 0.7)
        self.assertAlmostEqual(sources["combined"]["previous_average"], 0.3)
        self.assertAlmostEqual(sources["combined"]["change"], 0.4)
        self.assertEqual(sources["text"]["readings"], 1)
        self.assertIsNone(sources["text"]["change"])
        self.assertIsNone(sources["voice"]["average"])
        self.assertEqual(summary["totals"]["check_ins"], 2)
        self.assertEqual(summary["totals"]["previous_check_ins"], 1)

    def test_emotions_and_safety_events(self) -> None:
        rows = [_row(0), _row(0), _row(2)]
        rows[0]["metrics"].update(current_emotion="sadness", crisis=True)
        rows[1]["metrics"].update(current_emotion="sadness", contacts_offered=True)
        rows[2]["metrics"].update(current_emotion="joy")
        summary = summarize(rows, days=7, today=TODAY)
        self.assertEqual(summary["emotions"][0], {"label": "sadness", "count": 2, "share": 0.6667})
        self.assertAlmostEqual(summary["negative_share"], 0.6667)
        self.assertEqual(summary["totals"]["crisis"], 1)
        self.assertEqual(summary["totals"]["contacts_offered"], 1)
        self.assertEqual([event["kind"] for event in summary["events"]], ["crisis", "contacts_offered"])
        self.assertEqual(summary["daily"][-1]["dominant_emotion"], "sadness")


class MetricsStoreTests(unittest.TestCase):
    def test_metrics_survive_new_chat_and_age_out(self) -> None:
        store = ChatStore(":memory:")
        self.addCleanup(store.close)
        store.record_metrics("t1", "patient", {"mode": "text"})
        store.record_metrics("t0", "patient", {"mode": "voice"}, created_at=0.0)
        store.reset("patient")
        self.assertEqual([row["metrics"]["mode"] for row in store.metrics("patient", 0)], ["voice", "text"])
        self.assertEqual(store.purge_metrics(86_400), 1)
        self.assertEqual([row["metrics"]["mode"] for row in store.metrics("patient", 0)], ["text"])


if __name__ == "__main__":
    unittest.main()
