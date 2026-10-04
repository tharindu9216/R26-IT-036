"""The exported next-intensity hybrid, loaded and scored as the app loads it.

These run against the real artifacts in ``model/forcast``.
The threshold and blend weights were selected on the validation split during
training, so the point of the parity test is narrow but important: the backend
recomputes the current-emotion probabilities and the frozen embeddings at
request time rather than reading them from the training cache, and a
silently-wrong feature order or a skipped standardization step would still
produce plausible probabilities.
"""

from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

import torch

from emotion_chain.core import CURRENT_EMOTIONS, INTENSITY_LABELS, allowed_next_states
from emotion_chain.pipeline import (
    CurrentEmotionClassifier,
    EmotionPrediction,
    NextIntensityForecaster,
)

REPOSITORY_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = REPOSITORY_DIR / "model"
FORECAST_DIR = MODEL_DIR / "forcast"
VALID_CSV = FORECAST_DIR / "final_datasets" / "valid.csv"


def _stored_prediction(row: dict[str, str]) -> EmotionPrediction:
    """The current-emotion output the trainer recorded for one dataset row."""

    return EmotionPrediction(
        row["current_emotion_pred"],
        float(row["current_emotion_confidence"]),
        {name: float(row[f"current_prob_{name}"]) for name in CURRENT_EMOTIONS},
    )


class ExportedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cpu")
        cls.classifier = CurrentEmotionClassifier(
            MODEL_DIR / "current_emotion_classifier", cls.device
        )
        cls.forecaster = NextIntensityForecaster(
            FORECAST_DIR, cls.classifier, cls.device
        )
        cls.metadata = json.loads(
            (FORECAST_DIR / "metadata.json").read_text(encoding="utf-8")
        )

    def test_the_locked_selection_is_loaded_not_recomputed(self) -> None:
        blend = self.metadata["blend"]
        self.assertEqual(self.forecaster.threshold, blend["threshold"])
        self.assertEqual(self.forecaster.baseline_weight, blend["baseline_weight"])
        self.assertEqual(self.forecaster.fusion_weight, blend["fusion_weight"])
        self.assertEqual(len(self.forecaster.fusion_models), 3)

    def test_a_prediction_is_a_blend_of_both_branches(self) -> None:
        current = self.classifier.predict("I cannot stop crying about this.")
        forecast = self.forecaster.predict(
            "I cannot stop crying about this.", current
        )
        self.assertIn(forecast.label, INTENSITY_LABELS)
        self.assertAlmostEqual(
            forecast.probability_high,
            0.5 * forecast.baseline_probability + 0.5 * forecast.fusion_probability,
            places=9,
        )
        self.assertEqual(
            forecast.label,
            "high" if forecast.probability_high >= forecast.threshold else "low",
        )
        # Confidence is the probability of the class reported, not of "high".
        # It is deliberately allowed below 0.5: the threshold is 0.595, so a
        # "low" call made just under it is a weak one and should read as such.
        expected = (
            forecast.probability_high
            if forecast.label == "high"
            else 1.0 - forecast.probability_high
        )
        self.assertAlmostEqual(forecast.confidence, expected, places=9)

    def test_the_named_state_stays_inside_the_transition_table(self) -> None:
        from emotion_chain.core import intensity_state

        for emotion in CURRENT_EMOTIONS:
            for level in INTENSITY_LABELS:
                with self.subTest(emotion=emotion, intensity=level):
                    self.assertIn(
                        intensity_state(emotion, level), allowed_next_states(emotion)
                    )

    def test_the_state_probability_agrees_with_the_reported_confidence(self) -> None:
        """The chain must not report one number and chart another."""

        from emotion_chain.core import intensity_probability_map, intensity_state

        for emotion in ("sadness", "anger", "fear", "joy"):
            for probability in (0.2, 0.6, 0.9):
                level = "high" if probability >= 0.595 else "low"
                with self.subTest(emotion=emotion, probability=probability):
                    label = intensity_state(emotion, level)
                    mapped = intensity_probability_map(emotion, probability)
                    confidence = probability if level == "high" else 1.0 - probability
                    self.assertAlmostEqual(mapped[label], confidence)

    def test_predictions_are_deterministic(self) -> None:
        text = "Everything is falling apart and nobody is listening."
        current = self.classifier.predict(text)
        first = self.forecaster.predict(text, current)
        second = self.forecaster.predict(text, current)
        self.assertEqual(first.probability_high, second.probability_high)

    def test_empty_text_is_rejected(self) -> None:
        current = self.classifier.predict("something")
        with self.assertRaises(ValueError):
            self.forecaster.predict("   ", current)

    @unittest.skipUnless(VALID_CSV.is_file(), "training splits are not present")
    def test_it_reproduces_the_recorded_validation_accuracy(self) -> None:
        """Replay the validation split through the live inference path.

        The dataset's own stored current-emotion columns are used, exactly as
        the trainer did, so a mismatch here is the forecaster's and not the
        classifier's.
        """

        with VALID_CSV.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        correct = 0
        for row in rows:
            forecast = self.forecaster.predict(
                row["current_user_text"], _stored_prediction(row)
            )
            correct += forecast.label == row["next_intensity_label"]
        accuracy = correct / len(rows)
        self.assertAlmostEqual(
            accuracy, self.metadata["validation_metrics"]["accuracy"], places=3
        )


class ExplanationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        device = torch.device("cpu")
        cls.classifier = CurrentEmotionClassifier(
            MODEL_DIR / "current_emotion_classifier", device
        )
        cls.forecaster = NextIntensityForecaster(FORECAST_DIR, cls.classifier, device)

    def test_attributions_cover_the_message_tokens(self) -> None:
        text = "I feel worse every single day."
        current = self.classifier.predict(text)
        items = self.forecaster.explain(text, current, n_steps=4)
        self.assertGreater(len(items), 0)
        self.assertTrue(all(item.score == item.score for item in items))  # not NaN


if __name__ == "__main__":
    unittest.main()
