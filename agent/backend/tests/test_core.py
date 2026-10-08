import unittest

from emotion_chain.core import (
    CURRENT_EMOTIONS,
    INTENSITY_LABELS,
    STATE_LABELS,
    allowed_next_states,
    constrain_probability_map,
    emotion_family,
    intensity_probability_map,
    intensity_state,
    normalize_intensity,
    select_reply_route,
    tokenize_state_text,
    validate_forecaster_metadata,
)


def _metadata(**overrides):
    base = {
        "task": "next_intensity",
        "labels": ["low", "high"],
        "current_emotions": list(CURRENT_EMOTIONS),
        "blend": {
            "baseline_weight": 0.5,
            "fusion_weight": 0.5,
            "threshold": 0.595,
        },
    }
    base.update(overrides)
    return base


class TransitionTests(unittest.TestCase):
    def test_anger_can_only_move_to_reachable_states(self) -> None:
        self.assertEqual(
            allowed_next_states("anger"),
            ("neutral", "low_anger", "high_anger"),
        )

    def test_constraint_masks_and_renormalizes(self) -> None:
        raw = {
            "neutral": 0.1,
            "low_sadness": 0.2,
            "high_sadness": 0.3,
            "joy": 0.4,
        }
        constrained = constrain_probability_map(raw, "sadness")
        self.assertAlmostEqual(sum(constrained.values()), 1.0)
        self.assertEqual(constrained["joy"], 0.0)
        self.assertGreater(constrained["high_sadness"], constrained["neutral"])


class IntensityMappingTests(unittest.TestCase):
    """The forecast must always name a state the transition table allows."""

    def test_every_pair_names_a_reachable_state(self) -> None:
        for emotion in CURRENT_EMOTIONS:
            for level in INTENSITY_LABELS:
                with self.subTest(emotion=emotion, intensity=level):
                    self.assertIn(
                        intensity_state(emotion, level),
                        allowed_next_states(emotion),
                    )

    def test_the_family_is_carried_over_from_the_current_emotion(self) -> None:
        self.assertEqual(intensity_state("sadness", "high"), "high_sadness")
        self.assertEqual(intensity_state("anger", "low"), "low_anger")
        self.assertEqual(emotion_family(intensity_state("fear", "high")), "fear")

    def test_neutral_has_no_low_high_state_to_name(self) -> None:
        self.assertEqual(intensity_state("neutral", "low"), "neutral")
        self.assertEqual(intensity_state("neutral", "high"), "neutral")

    def test_probability_map_covers_every_label_and_sums_to_one(self) -> None:
        for emotion in CURRENT_EMOTIONS:
            with self.subTest(emotion=emotion):
                mapped = intensity_probability_map(emotion, 0.7)
                self.assertEqual(set(mapped), set(STATE_LABELS))
                self.assertAlmostEqual(sum(mapped.values()), 1.0)
                unreachable = set(STATE_LABELS) - set(allowed_next_states(emotion))
                self.assertTrue(all(mapped[label] == 0.0 for label in unreachable))

    def test_probability_map_splits_between_the_two_intensities(self) -> None:
        mapped = intensity_probability_map("sadness", 0.7)
        self.assertAlmostEqual(mapped["high_sadness"], 0.7)
        self.assertAlmostEqual(mapped["low_sadness"], 0.3)
        self.assertEqual(mapped["neutral"], 0.0)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_intensity("medium")
        with self.assertRaises(ValueError):
            intensity_state("excited", "high")
        with self.assertRaises(ValueError):
            intensity_probability_map("sadness", 1.4)


class ForecasterMetadataTests(unittest.TestCase):
    def test_the_shipped_shape_is_accepted(self) -> None:
        validate_forecaster_metadata(_metadata())

    def test_a_reordered_emotion_vector_is_rejected(self) -> None:
        # Both branches read the probabilities positionally, so a different
        # order would score every turn without raising anything by itself.
        reordered = list(reversed(CURRENT_EMOTIONS))
        with self.assertRaises(ValueError):
            validate_forecaster_metadata(_metadata(current_emotions=reordered))

    def test_a_different_task_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_forecaster_metadata(_metadata(task="next_emotion_state"))

    def test_weights_must_sum_to_one_and_the_threshold_must_be_a_probability(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            validate_forecaster_metadata(
                _metadata(blend={"baseline_weight": 0.5, "fusion_weight": 0.9, "threshold": 0.5})
            )
        with self.assertRaises(ValueError):
            validate_forecaster_metadata(
                _metadata(blend={"baseline_weight": 0.5, "fusion_weight": 0.5, "threshold": 1.4})
            )


class RoutingTests(unittest.TestCase):
    def test_neutral_and_joy_use_base(self) -> None:
        self.assertEqual(select_reply_route("neutral", "joy"), "base")
        self.assertEqual(select_reply_route("joy", "high_joy"), "base")

    def test_negative_family_uses_adapter(self) -> None:
        self.assertEqual(
            select_reply_route("sadness", "low_sadness"),
            "supportive_adapter",
        )
        self.assertEqual(emotion_family("high_fear"), "fear")

    def test_stress_or_cbt_overrides_happy_route(self) -> None:
        self.assertEqual(
            select_reply_route("joy", "high_joy", sensor_stress=True),
            "supportive_adapter",
        )

    def test_c2_voice_stress_overrides_happy_route(self) -> None:
        self.assertEqual(
            select_reply_route("joy", "high_joy", voice_stress=True),
            "supportive_adapter",
        )

    def test_uncertain_negative_forecast_does_not_override_positive_current(self) -> None:
        self.assertEqual(
            select_reply_route(
                "joy",
                "low_sadness",
                next_emotion_confidence=0.49,
                next_negative_min_confidence=0.5,
            ),
            "base",
        )
        self.assertEqual(
            select_reply_route(
                "joy",
                "low_sadness",
                next_emotion_confidence=0.5,
                next_negative_min_confidence=0.5,
            ),
            "supportive_adapter",
        )

    def test_current_negative_and_stress_still_override_forecast_gate(self) -> None:
        self.assertEqual(
            select_reply_route(
                "sadness",
                "neutral",
                next_emotion_confidence=0.1,
                next_negative_min_confidence=0.5,
            ),
            "supportive_adapter",
        )
        self.assertEqual(
            select_reply_route(
                "joy",
                "neutral",
                sensor_stress=True,
                next_emotion_confidence=0.1,
                next_negative_min_confidence=0.5,
            ),
            "supportive_adapter",
        )
        self.assertEqual(
            select_reply_route("neutral", "neutral", text_signal_flagged=True),
            "supportive_adapter",
        )


class TokenizationTests(unittest.TestCase):
    def test_preserves_contractions_and_punctuation(self) -> None:
        self.assertEqual(
            tokenize_state_text("I'm okay, really!"),
            ["i'm", "okay", ",", "really", "!"],
        )


if __name__ == "__main__":
    unittest.main()
