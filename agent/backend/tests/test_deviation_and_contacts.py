"""Deviation tracking and the extreme-negative contact escalation."""

from __future__ import annotations

import unittest

from emotion_chain.agent_graph import AgentComponents, SupportiveAgentWorkflow
from emotion_chain.deviation_tracker import compute_deviation, valence_group
from emotion_chain.support_contacts import (
    DEFAULT_SUPPORT_CONTACTS,
    SUMITHURO_CONTACT,
    SupportContact,
    append_contact_message,
    contact_message,
    escalation_signals,
    is_extremely_negative,
    speakable,
)


class DeviationTests(unittest.TestCase):
    def test_first_turn_has_nothing_to_deviate_from(self) -> None:
        deviation = compute_deviation(None, "sadness")
        self.assertEqual(deviation.level, "None")
        self.assertEqual(deviation.score, 0.0)
        self.assertIsNone(deviation.previous_emotion)

    def test_repeated_emotion_is_not_a_deviation(self) -> None:
        self.assertEqual(compute_deviation("anger", "anger").level, "None")

    def test_moving_through_neutral_is_low(self) -> None:
        self.assertEqual(compute_deviation("neutral", "sadness").level, "Low")
        self.assertEqual(compute_deviation("joy", "neutral").level, "Low")

    def test_negative_to_negative_is_moderate(self) -> None:
        deviation = compute_deviation("sadness", "anger")
        self.assertEqual(deviation.level, "Moderate")
        self.assertAlmostEqual(deviation.score, 0.60)

    def test_valence_reversal_is_high(self) -> None:
        self.assertEqual(compute_deviation("joy", "sadness").level, "High")
        self.assertEqual(compute_deviation("fear", "joy").level, "High")
        self.assertTrue(compute_deviation("joy", "anger").is_high)

    def test_intensity_states_group_with_their_family(self) -> None:
        self.assertEqual(valence_group("high_sadness"), "negative")
        self.assertEqual(valence_group("low_joy"), "positive")
        self.assertEqual(compute_deviation("joy", "high_sadness").level, "High")


class EscalationRuleTests(unittest.TestCase):
    def test_crisis_always_escalates(self) -> None:
        self.assertTrue(
            is_extremely_negative("joy", "high_joy", safety_status="crisis")
        )

    def test_positive_turn_never_escalates(self) -> None:
        self.assertFalse(is_extremely_negative("joy", "high_joy"))
        self.assertFalse(
            is_extremely_negative(
                "neutral", "sadness", sensor_stress=True, voice_stress=True
            )
        )

    def test_an_ordinary_hard_day_does_not_escalate(self) -> None:
        """"I feel overwhelmed today": sadness now, high_sadness forecast, and
        nothing else agreeing. One signal is not a phone number."""

        self.assertFalse(is_extremely_negative("sadness", "high_sadness"))
        self.assertEqual(
            escalation_signals("sadness", "high_sadness"),
            ("high_intensity_forecast",),
        )

    def test_no_single_signal_escalates(self) -> None:
        for kwargs in (
            {"deviation_level": "High"},
            {"sensor_stress": True},
            {"text_signal_flagged": True},
            {"voice_stress": True},
        ):
            with self.subTest(**kwargs):
                self.assertFalse(
                    is_extremely_negative("sadness", "low_sadness", **kwargs)
                )

    def test_two_agreeing_signals_escalate(self) -> None:
        self.assertTrue(
            is_extremely_negative(
                "sadness", "high_sadness", text_signal_flagged=True
            )
        )
        self.assertTrue(
            is_extremely_negative(
                "sadness", "neutral", deviation_level="High", sensor_stress=True
            )
        )
        self.assertTrue(
            is_extremely_negative(
                "fear", "low_fear", sensor_stress=True, voice_stress=True
            )
        )

    def test_signals_are_named_for_explanation(self) -> None:
        self.assertEqual(
            escalation_signals(
                "sadness",
                "high_sadness",
                deviation_level="High",
                voice_stress=True,
            ),
            ("high_intensity_forecast", "high_deviation", "voice_stress"),
        )
        self.assertEqual(escalation_signals("joy", "high_joy", sensor_stress=True), ())


class ContactMessageTests(unittest.TestCase):
    def test_default_contact_is_named_with_its_number(self) -> None:
        message = contact_message(DEFAULT_SUPPORT_CONTACTS)
        self.assertIn(SUMITHURO_CONTACT.name, message)
        self.assertIn("077766889", message)

    def test_appending_is_idempotent(self) -> None:
        once = append_contact_message("Take care.", DEFAULT_SUPPORT_CONTACTS)
        twice = append_contact_message(once, DEFAULT_SUPPORT_CONTACTS)
        self.assertEqual(once, twice)
        self.assertEqual(once.count("077766889"), 1)

    def test_no_contacts_configured_leaves_the_reply_alone(self) -> None:
        self.assertEqual(append_contact_message("Take care.", ()), "Take care.")
        self.assertEqual(contact_message(()), "")

    def test_extra_contacts_are_all_listed(self) -> None:
        contacts = (SUMITHURO_CONTACT, SupportContact("Night line", "0112345678"))
        message = contact_message(contacts)
        self.assertIn("077766889", message)
        self.assertIn("0112345678", message)


class SpokenReplyTests(unittest.TestCase):
    def test_tts_gets_the_transliterated_name_and_digit_spoken_number(self) -> None:
        reply = append_contact_message("Take care.", DEFAULT_SUPPORT_CONTACTS)
        spoken = speakable(reply, DEFAULT_SUPPORT_CONTACTS)
        self.assertIn("Sumithuro", spoken)
        self.assertNotIn("සුමිතුරෝ", spoken)
        self.assertIn(
            "zero, seven, seven, seven, six, six, eight, eight, nine",
            spoken,
        )
        self.assertNotIn("077766889", spoken)
        # The displayed reply is untouched; only the audio is rewritten.
        self.assertIn("සුමිතුරෝ", reply)
        self.assertIn("077766889", reply)

    def test_a_speakable_name_stays_and_its_number_is_expanded(self) -> None:
        contacts = (SupportContact("Night line", "0112345678"),)
        reply = append_contact_message("Take care.", contacts)
        spoken = speakable(reply, contacts)
        self.assertIn("Night line", spoken)
        self.assertIn(
            "zero, one, one, two, three, four, five, six, seven, eight",
            spoken,
        )


class GraphNodeTests(unittest.TestCase):
    """The nodes are plain callables, so they run without LangGraph installed."""

    def setUp(self) -> None:
        self.workflow = SupportiveAgentWorkflow(
            AgentComponents(emotion_chain=None, reply_generator=None)
        )

    def test_deviation_node_scores_against_the_carried_emotion(self) -> None:
        state = {"previous_emotion": "joy", "current_emotion": "sadness"}
        self.assertEqual(
            self.workflow.deviation_node(state),
            {
                "previous_emotion": "joy",
                "deviation_score": 0.90,
                "deviation_level": "High",
            },
        )

    def test_deviation_node_handles_a_first_turn(self) -> None:
        result = self.workflow.deviation_node({"current_emotion": "neutral"})
        self.assertEqual(result["deviation_level"], "None")
        self.assertIsNone(result["previous_emotion"])

    def test_crisis_node_carries_the_contacts(self) -> None:
        result = self.workflow.crisis_node({"user_text": "..."})
        self.assertEqual(result["reply_route"], "crisis")
        self.assertIn("077766889", result["reply"])
        self.assertEqual(
            result["support_contacts"],
            [SUMITHURO_CONTACT.as_dict()],
        )


if __name__ == "__main__":
    unittest.main()
