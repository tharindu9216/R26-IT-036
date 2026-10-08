"""Fast unit checks for the corrected C3 runtime contracts."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.c3_text_stressor_distortion.app.calibration import (
    TemperatureCalibrator,
)
from backend.c3_text_stressor_distortion.app.config import (
    CBT_CALIBRATION_PATH,
    STRESS_CALIBRATION_PATH,
    STRESS_ENSEMBLE_MEMBERS,
)
from backend.c3_text_stressor_distortion.app.fusion.xai_service import (
    FusionExplanationService,
)
from backend.c3_text_stressor_distortion.app.preprocessing import (
    preprocess_cbt_text,
    preprocess_stress_text,
)


class RuntimeContractTests(unittest.TestCase):
    def test_stress_preprocessing_matches_notebook_rules(self):
        text = "I can't cope soooo well!!! See r/anxiety https://example.org"
        self.assertEqual(
            preprocess_stress_text(text, "BERT"),
            "i cannot cope soo well! see",
        )
        self.assertEqual(
            preprocess_stress_text(text, "DeBERTa-v3"),
            "cannot cope soo well! see",
        )

    def test_cbt_preprocessing_keeps_reddit_like_literal(self):
        text = "I'm sure u/me hates me!!!"
        self.assertEqual(
            preprocess_cbt_text(text, "BERT"),
            "i am sure u me hates me!",
        )
        self.assertEqual(
            preprocess_cbt_text(text, "DeBERTa-v3"),
            "am sure me hates me!",
        )

    def test_stress_runtime_uses_reported_final_checkpoints(self):
        for specification in STRESS_ENSEMBLE_MEMBERS.values():
            path = Path(specification["checkpoint"])
            self.assertTrue(path.name.endswith("_final.pt"), path)
            self.assertTrue(path.exists(), path)

    def test_calibration_artifacts_preserve_locked_decisions(self):
        for path in (STRESS_CALIBRATION_PATH, CBT_CALIBRATION_PATH):
            calibrator = TemperatureCalibrator.from_json(path)
            grid = np.linspace(0.001, 0.999, 999)
            calibrated = calibrator.calibrate_distribution(
                np.column_stack([1.0 - grid, grid])
            )[:, 1]
            raw_decisions = grid >= calibrator.input_threshold
            calibrated_decisions = (
                calibrated >= calibrator.calibrated_threshold
            )
            np.testing.assert_array_equal(raw_decisions, calibrated_decisions)

            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertLess(
                payload["evaluation"]["after"]["ece"],
                payload["evaluation"]["before"]["ece"],
            )
            self.assertLess(
                payload["evaluation"]["after"]["brier_score"],
                payload["evaluation"]["before"]["brier_score"],
            )

    def test_xai_aggregation_uses_member_weights(self):
        rows_a = [
            {"word": "worried", "integrated_gradients": 1.0, "shap": 1.0, "lime": 1.0},
            {"word": "calm", "integrated_gradients": 0.0, "shap": 0.0, "lime": 0.0},
        ]
        rows_b = [
            {"word": "worried", "integrated_gradients": 0.0, "shap": 0.0, "lime": 0.0},
            {"word": "calm", "integrated_gradients": 1.0, "shap": 1.0, "lime": 1.0},
        ]
        combined = FusionExplanationService._aggregate_member_results([
            (SimpleNamespace(name="A", weight=0.25), {"feature_rows": rows_a}),
            (SimpleNamespace(name="B", weight=0.75), {"feature_rows": rows_b}),
        ])
        by_word = {row["word"]: row for row in combined["ranked_features"]}
        self.assertAlmostEqual(by_word["worried"]["consensus"], 0.25)
        self.assertAlmostEqual(by_word["calm"]["consensus"], 0.75)
        self.assertEqual(combined["member_weights"], {"A": 0.25, "B": 0.75})


if __name__ == "__main__":
    unittest.main()
