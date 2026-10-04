import unittest

from emotion_chain.multimodal_stress import (
    fuse_multimodal_stress,
    score_questionnaire,
)
from emotion_chain.agent_graph import AgentComponents, SupportiveAgentWorkflow


class QuestionnaireTests(unittest.TestCase):
    def test_stress_worded_and_calm_worded_items_are_scored_in_one_direction(self):
        stressed = score_questionnaire(
            {
                "manageable": 0,
                "tense": 3,
                "overwhelmed": 3,
                "relaxed": 0,
                "worried": 3,
                "in_control": 0,
            }
        )
        self.assertEqual(stressed.probability, 1.0)
        self.assertTrue(stressed.is_stressed)

    def test_all_six_answers_are_required(self):
        with self.assertRaises(ValueError):
            score_questionnaire({"tense": 2})


class MultimodalFusionTests(unittest.TestCase):
    def test_questionnaire_has_the_requested_high_weight(self):
        fused = fuse_multimodal_stress(questionnaire=1.0, sensor=0.0, voice=0.0)
        self.assertAlmostEqual(fused.probability, 0.6)
        self.assertEqual(
            fused.normalized_weights,
            {"questionnaire": 0.6, "sensor": 0.4},
        )

    def test_missing_modalities_are_removed_and_weights_are_normalized(self):
        fused = fuse_multimodal_stress(questionnaire=0.8, sensor=0.2)
        self.assertAlmostEqual(fused.normalized_weights["questionnaire"], 0.6)
        self.assertAlmostEqual(fused.normalized_weights["sensor"], 0.4)
        self.assertAlmostEqual(fused.probability, 0.56)

    def test_voice_does_not_change_the_shared_sensor_questionnaire_score(self):
        calm_voice = fuse_multimodal_stress(
            questionnaire=0.8, sensor=0.2, voice=0.0
        )
        stressed_voice = fuse_multimodal_stress(
            questionnaire=0.8, sensor=0.2, voice=1.0
        )
        self.assertAlmostEqual(calm_voice.probability, 0.56)
        self.assertEqual(calm_voice, stressed_voice)
        self.assertNotIn("voice", calm_voice.normalized_weights)
        self.assertNotIn("voice", calm_voice.probabilities)

    def test_no_inputs_is_an_unavailable_result(self):
        fused = fuse_multimodal_stress()
        self.assertFalse(fused.available)
        self.assertIsNone(fused.probability)

    def test_combined_decision_replaces_raw_sensor_and_voice_for_routing(self):
        workflow = SupportiveAgentWorkflow(
            AgentComponents(emotion_chain=None, reply_generator=None)
        )
        calm = workflow.route_node(
            {
                "current_emotion": "neutral",
                "next_emotion": "neutral",
                "sensor_stress": True,
                "voice_stress": True,
                "multimodal_stress_available": True,
                "multimodal_stress": False,
            }
        )
        stressed = workflow.route_node(
            {
                "current_emotion": "neutral",
                "next_emotion": "neutral",
                "sensor_stress": False,
                "voice_stress": False,
                "multimodal_stress_available": True,
                "multimodal_stress": True,
            }
        )
        self.assertEqual(calm["reply_route"], "base")
        self.assertEqual(stressed["reply_route"], "supportive_adapter")


if __name__ == "__main__":
    unittest.main()
