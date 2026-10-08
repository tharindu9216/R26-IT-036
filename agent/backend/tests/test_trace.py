from __future__ import annotations

import unittest

from emotion_chain.core import reply_route_reasons, select_reply_route
from emotion_chain.trace import build_trace


def _steps(trace: dict) -> dict[str, dict]:
    return {step["key"]: step for step in trace["steps"]}


class ReplyRouteReasonsTests(unittest.TestCase):
    """The reasons must never disagree with the route they explain."""

    def test_no_reason_means_the_base_route(self) -> None:
        self.assertEqual(reply_route_reasons("joy", "high_joy"), ())
        self.assertEqual(select_reply_route("joy", "high_joy"), "base")

    def test_each_trigger_is_named_on_its_own(self) -> None:
        cases = {
            "negative_current_emotion": {"current_emotion": "sadness", "next_emotion": "neutral"},
            "negative_forecast": {"current_emotion": "joy", "next_emotion": "sadness"},
            "sensor_stress": {"current_emotion": "joy", "next_emotion": "high_joy", "sensor_stress": True},
            "text_signal": {"current_emotion": "joy", "next_emotion": "high_joy", "text_signal_flagged": True},
            "voice_stress": {"current_emotion": "joy", "next_emotion": "high_joy", "voice_stress": True},
        }
        for expected, kwargs in cases.items():
            current = kwargs.pop("current_emotion")
            following = kwargs.pop("next_emotion")
            with self.subTest(expected):
                self.assertEqual(
                    reply_route_reasons(current, following, **kwargs), (expected,)
                )
                self.assertEqual(
                    select_reply_route(current, following, **kwargs),
                    "supportive_adapter",
                )

    def test_an_unconfident_negative_forecast_is_not_a_reason(self) -> None:
        kwargs = {"next_emotion_confidence": 0.2, "next_negative_min_confidence": 0.5}
        self.assertEqual(reply_route_reasons("joy", "sadness", **kwargs), ())
        self.assertEqual(select_reply_route("joy", "sadness", **kwargs), "base")

    def test_reasons_accumulate(self) -> None:
        self.assertEqual(
            reply_route_reasons("sadness", "high_sadness", sensor_stress=True),
            ("negative_current_emotion", "negative_forecast", "sensor_stress"),
        )

    def test_validation_still_rejects_out_of_range_confidence(self) -> None:
        with self.assertRaises(ValueError):
            reply_route_reasons("joy", "joy", next_emotion_confidence=1.4)
        with self.assertRaises(ValueError):
            select_reply_route("joy", "joy", next_negative_min_confidence=-0.1)


class BuildTraceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = {
            "user_text": "I cannot sleep",
            "safety_status": "normal",
            "sensor_stress": True,
            "sensor_stress_probability": 0.71,
            "sensor_stress_available": True,
            "text_stress": True,
            "text_stress_probability": 0.82,
            "has_cbt_distortion": False,
            "cbt_probability": 0.12,
            "text_signal_flagged": True,
            "text_signal_probability": 0.63,
            "text_signal_method": "weighted_average",
            "current_emotion": "sadness",
            "current_emotion_confidence": 0.78,
            "next_emotion": "high_sadness",
            "next_emotion_confidence": 0.66,
            "previous_emotion": "joy",
            "deviation_level": "High",
            "deviation_score": 0.9,
            "reply_route": "supportive_adapter",
            "reply_source": "qwen",
            "reply": "That sounds exhausting.",
            "support_contacts": [{"name": "Sumithuro", "phone": "1333"}],
        }

    def test_the_flow_starts_at_the_input_and_ends_at_the_reply(self) -> None:
        steps = build_trace(self.state)["steps"]
        self.assertEqual(steps[0]["key"], "input")
        self.assertEqual(steps[0]["summary"], "I cannot sleep")
        self.assertEqual(steps[-1]["key"], "output")
        self.assertEqual(steps[-1]["summary"], "That sounds exhausting.")

    def test_strategy_step_names_every_condition_that_fired(self) -> None:
        step = _steps(build_trace(self.state, next_negative_min_confidence=0.5))["strategy"]
        self.assertEqual(step["summary"], "supportive_adapter")
        self.assertEqual(
            [reason["code"] for reason in step["reasons"]],
            ["negative_current_emotion", "negative_forecast", "sensor_stress", "text_signal"],
        )

    def test_strategy_step_explains_the_base_route_too(self) -> None:
        calm = dict(
            self.state,
            current_emotion="joy",
            next_emotion="high_joy",
            sensor_stress=False,
            text_signal_flagged=False,
            reply_route="base",
        )
        step = _steps(build_trace(calm))["strategy"]
        self.assertEqual(step["summary"], "base")
        self.assertEqual([reason["code"] for reason in step["reasons"]], ["no_trigger"])

    def test_the_trace_route_matches_what_the_graph_decided(self) -> None:
        trace = build_trace(self.state, next_negative_min_confidence=0.5)
        self.assertEqual(trace["route"], "supportive_adapter")
        self.assertEqual(_steps(trace)["generate"]["outputs"][0]["value"], "yes")

    def test_a_missing_sensor_is_skipped_rather_than_reported_calm(self) -> None:
        step = _steps(build_trace(dict(self.state, sensor_stress_available=False)))["sensor"]
        self.assertEqual(step["status"], "skipped")
        self.assertNotEqual(step["summary"], "calm")

    def test_shared_stress_score_matches_the_questionnaire_sensor_fusion(self) -> None:
        combined = dict(
            self.state,
            sensor_stress=False,
            sensor_stress_probability=0.05,
            multimodal_stress_available=True,
            multimodal_stress=True,
            multimodal_stress_probability=0.62,
            multimodal_stress_weights={"questionnaire": 0.6, "sensor": 0.4},
            multimodal_stress_inputs={"questionnaire": 1.0, "sensor": 0.05},
        )
        steps = _steps(build_trace(combined))
        sensor = steps["sensor"]
        self.assertEqual(sensor["component"], "Combined stress")
        self.assertEqual(sensor["summary"], "Stressed · 62.0%")
        self.assertNotIn("multimodal", steps)
        self.assertEqual(sensor["outputs"][0]["value"], "62.0%")
        self.assertEqual(
            [output["label"] for output in sensor["outputs"]],
            ["combined stress score", "check-in", "wearable"],
        )
        self.assertEqual(
            _steps(build_trace(combined, mode="voice"))["sensor"],
            sensor,
        )

    def test_voice_mode_records_c2_and_marks_c3_absent(self) -> None:
        voice = dict(
            self.state,
            voice_stress=True,
            voice_stress_score=0.74,
            voice_appraisal_state="threat",
        )
        steps = _steps(build_trace(voice, mode="voice"))
        self.assertEqual(steps["voice"]["summary"], "stressed")
        self.assertEqual(steps["c3"]["status"], "skipped")
        self.assertNotIn("fusion", steps)

    def test_crisis_short_circuit_marks_the_classifiers_unreached(self) -> None:
        crisis = {
            "user_text": "i want to end my life",
            "safety_status": "crisis",
            "sensor_stress_available": False,
            "reply_route": "crisis",
            "reply_source": "safety_template",
            "reply": "Your immediate safety matters most.",
            "support_contacts": [{"name": "Sumithuro", "phone": "1333"}],
        }
        steps = _steps(build_trace(crisis))
        self.assertEqual(steps["safety"]["summary"], "crisis")
        self.assertEqual(steps["emotion"]["status"], "skipped")
        self.assertEqual(steps["deviation"]["status"], "skipped")
        self.assertEqual(steps["strategy"]["summary"], "crisis")
        self.assertEqual(steps["escalation"]["summary"], "contacts offered")

    def test_a_template_fallback_is_called_out(self) -> None:
        step = _steps(build_trace(dict(self.state, reply_source="template_fallback")))["generate"]
        self.assertIn("could not be reached", step["summary"])
        self.assertIn("/api/deployment", step["detail"])

    def test_every_step_is_json_safe(self) -> None:
        import json

        json.dumps(build_trace(self.state))


if __name__ == "__main__":
    unittest.main()
