"""Sample-text tests for the real deployed C3 Stress header."""

from __future__ import annotations

import gc
import os
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.c3_text_stressor_distortion.app import stress_model


def _cached_model_path(model_id: str) -> str:
    """Use a complete local HF snapshot when one is already cached."""
    cache_root = Path(
        os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")
    ) / "hub"
    model_root = cache_root / f"models--{model_id.replace('/', '--')}"
    reference = model_root / "refs" / "main"
    if reference.exists():
        snapshot = model_root / "snapshots" / reference.read_text(encoding="utf-8").strip()
        if snapshot.exists():
            return str(snapshot)
    return model_id


# Balanced examples: five stressed and five not-stressed texts.
STRESS_CASES = (
    ("STR001", "I am overwhelmed by work and cannot keep up with the deadlines anymore.", 1),
    ("STR002", "Bills keep arriving and I am terrified that I will not make rent this month.", 1),
    ("STR003", "Everything is piling up at once and I feel close to breaking down.", 1),
    ("STR004", "I cannot stop worrying about the medical results and my stomach is in knots.", 1),
    ("STR005", "I have been worrying nonstop for days and cannot settle my thoughts.", 1),
    ("STR006", "I completed my work early and feel relaxed this evening.", 0),
    ("STR007", "I slept well and am looking forward to meeting my friends today.", 0),
    ("STR008", "There are two tasks left and I have enough time to complete both.", 0),
    ("STR009", "I spent the afternoon gardening and noticed how calm I felt.", 0),
    ("STR010", "The repair took longer than expected but the issue is fixed now.", 0),
)


class StressHeaderSampleTests(unittest.TestCase):
    """Compare real Stress-header predictions with expected sample labels."""

    @classmethod
    def setUpClass(cls):
        cls.original_runtime_ids = {
            name: spec["runtime_hf_id"]
            for name, spec in stress_model.STRESS_ENSEMBLE_MEMBERS.items()
        }
        for spec in stress_model.STRESS_ENSEMBLE_MEMBERS.values():
            spec["runtime_hf_id"] = _cached_model_path(spec["runtime_hf_id"])
        cls.predictor = stress_model.StressPredictor()

    @classmethod
    def tearDownClass(cls):
        del cls.predictor
        for name, runtime_id in cls.original_runtime_ids.items():
            stress_model.STRESS_ENSEMBLE_MEMBERS[name]["runtime_hf_id"] = runtime_id
        gc.collect()

    def check_sample(self, case_id: str, text: str, expected: int):
        result = self.predictor.predict(text)
        actual = int(result.is_stressed)

        # generate_report.py collects these values from the test instance.
        self.case_id = case_id
        self.sample_text = text
        self.expected_label = expected
        self.predicted_label = actual
        self.positive_probability = float(result.probabilities["stressed"])
        self.confidence = float(result.confidence)

        self.assertEqual(
            actual,
            expected,
            msg=(
                f"{case_id}: expected stress={bool(expected)}, "
                f"got {result.label!r} with confidence {result.confidence:.4f}"
            ),
        )


def _add_sample_tests() -> None:
    for case_id, text, expected in STRESS_CASES:
        def test_method(self, values=(case_id, text, expected)):
            self.check_sample(*values)

        test_method.__name__ = f"test_{case_id.lower()}"
        test_method.__doc__ = text
        setattr(StressHeaderSampleTests, test_method.__name__, test_method)


_add_sample_tests()


if __name__ == "__main__":
    unittest.main(verbosity=2)
