from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from emotion_chain.fusion import InfluenceWeights
from emotion_chain.reply_generator import QwenReplyGenerator, ReplySignals
from emotion_chain.trace import build_trace
from facial import aggregate_frame_probabilities, unavailable

MODELS_DIR = Path(__file__).resolve().parents[2] / "model"
CHECKPOINT = (
    MODELS_DIR / "WorkStress3D_Trained_Model" / "workstress3d_facial_stress_cnn_v2.pth"
)


class AggregationTests(unittest.TestCase):
    def test_too_few_faces_is_no_reading(self) -> None:
        result = aggregate_frame_probabilities(
            [0.9, 0.8], frames_received=10, min_face_frames=3, threshold=0.5
        )
        self.assertFalse(result.available)
        self.assertIsNone(result.probability)
        self.assertFalse(result.is_stressed)
        self.assertIn("only 2 of the 3", result.reason)

    def test_no_face_at_all_says_so(self) -> None:
        result = aggregate_frame_probabilities(
            [], frames_received=6, min_face_frames=3, threshold=0.5
        )
        self.assertEqual(result.reason, "no face was visible to the camera")

    def test_mean_over_face_frames_is_the_turn_result(self) -> None:
        result = aggregate_frame_probabilities(
            [0.2, 0.4, 0.9], frames_received=5, min_face_frames=3, threshold=0.5
        )
        self.assertTrue(result.available)
        self.assertAlmostEqual(result.probability, 0.5)
        self.assertTrue(result.is_stressed)
        self.assertEqual(result.faces_detected, 3)


@unittest.skipUnless(CHECKPOINT.is_file(), "facial checkpoint not present")
class AnalyzerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import cv2  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("opencv-python-headless not installed")
        from facial import FacialStressAnalyzer

        cls.analyzer = FacialStressAnalyzer(CHECKPOINT, min_face_frames=2)
        cls.analyzer.load()

    @staticmethod
    def _jpeg(image: np.ndarray) -> bytes:
        import cv2

        ok, encoded = cv2.imencode(".jpg", image)
        assert ok
        return encoded.tobytes()

    def test_checkpoint_loads_strictly(self) -> None:
        self.assertTrue(self.analyzer.is_loaded)

    def test_faceless_frames_are_never_scored(self) -> None:
        """A flat gray frame reads as ~95% stress to the CNN, so it must be dropped."""

        gray = np.full((240, 320), 128, dtype=np.uint8)
        frames = [self._jpeg(gray)] * 5
        result = self.analyzer.predict_frames(frames)
        self.assertFalse(result.available)
        self.assertEqual(result.faces_detected, 0)
        self.assertFalse(self.analyzer.detect_face(frames[0]))

    def test_undecodable_frames_are_skipped(self) -> None:
        result = self.analyzer.predict_frames([b"not an image", b""])
        self.assertFalse(result.available)
        self.assertEqual(result.frames_received, 2)

    def test_no_frames_means_camera_off(self) -> None:
        result = self.analyzer.predict_frames([])
        self.assertEqual(result.reason, "the camera was off for this turn")


class ReplyInfluenceTests(unittest.TestCase):
    def _generator(self, **weights) -> QwenReplyGenerator:
        return QwenReplyGenerator(
            MODELS_DIR / "qwen3-4b-instruct",
            MODELS_DIR / "esconv_reply_adapter",
            influence_weights=InfluenceWeights(**weights),
        )

    def test_face_is_weighted_into_the_prompt(self) -> None:
        signals = ReplySignals(
            "sadness",
            "low_sadness",
            face_stress=True,
            face_stress_probability=0.8,
            face_frames_detected=12,
        )
        lines = self._generator(face=0.15)._signal_influence(signals)
        self.assertEqual(len(lines), 1)
        self.assertIn("facial expression", lines[0])
        self.assertIn("stress detected, influence 0.12 of 1.00", lines[0])
        self.assertIn("face seen in 12 frames", lines[0])

    def test_face_is_listed_alongside_the_combined_check(self) -> None:
        signals = ReplySignals(
            "sadness",
            "low_sadness",
            multimodal_stress_available=True,
            multimodal_stress=True,
            multimodal_stress_probability=0.7,
            face_stress=False,
            face_stress_probability=0.3,
        )
        lines = self._generator()._signal_influence(signals)
        self.assertTrue(any("combined stress check" in line for line in lines))
        self.assertTrue(any("facial expression" in line for line in lines))

    def test_no_camera_reading_is_omitted(self) -> None:
        signals = ReplySignals("sadness", "low_sadness")
        self.assertEqual(self._generator()._signal_influence(signals), [])

    def test_face_never_selects_the_supportive_route(self) -> None:
        signals = ReplySignals(
            "joy", "high_joy", face_stress=True, face_stress_probability=0.99
        )
        generator = self._generator(face=1.0)

        def no_model() -> None:
            raise RuntimeError("Qwen is not loaded in unit tests")

        # The routed template answers, so the route is still reported.
        generator.load = no_model
        result = generator.generate("I am fine", signals)
        self.assertEqual(result.route, "base")
        self.assertEqual(result.support_contacts, ())

    def test_out_of_range_face_weight_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            InfluenceWeights(face=1.5)


class TraceTests(unittest.TestCase):
    @staticmethod
    def _face_step(state: dict, mode: str = "voice") -> dict | None:
        trace = build_trace({"user_text": "hi", **state}, mode=mode)
        return next((s for s in trace["steps"] if s["key"] == "face"), None)

    def test_text_mode_reports_the_camera_too(self) -> None:
        step = self._face_step({}, mode="text")
        self.assertEqual(step["status"], "skipped")
        self.assertIn("camera was off", step["detail"])

    def test_missing_reading_is_skipped_with_reason(self) -> None:
        reading = unavailable("no face was visible to the camera", frames_received=8)
        step = self._face_step(
            {
                "face_stress_available": False,
                "face_frames_received": reading.frames_received,
                "face_frames_detected": 0,
                "face_stress_reason": reading.reason,
            }
        )
        self.assertEqual(step["status"], "skipped")
        self.assertEqual(
            step["detail"], "No face was visible to the camera (0 of 8 frames held a face)."
        )

    def test_reading_reports_probability_and_frames(self) -> None:
        step = self._face_step(
            {
                "face_stress_available": True,
                "face_stress": True,
                "face_stress_probability": 0.71,
                "face_frames_received": 10,
                "face_frames_detected": 9,
            }
        )
        self.assertEqual(step["summary"], "stressed")
        values = {item["label"]: item["value"] for item in step["outputs"]}
        self.assertEqual(values["probability"], "71.0%")
        self.assertEqual(values["frames with a face"], "9 of 10")


if __name__ == "__main__":
    unittest.main()
