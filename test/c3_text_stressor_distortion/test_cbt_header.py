"""Sample-text tests for the real deployed C3 CBT header."""

from __future__ import annotations

import gc
import os
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.c3_text_stressor_distortion.app import cbt_model


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


# Balanced examples: five cognitive distortions and five balanced appraisals.
CBT_CASES = (
    ("CBT001", "If I make one mistake in this presentation then the whole thing will be a complete disaster.", 1),
    ("CBT002", "I missed one deadline so I am obviously useless at every job.", 1),
    ("CBT003", "My friend has not replied yet; she must be angry with me.", 1),
    ("CBT004", "Nothing ever works out for me and it never will.", 1),
    ("CBT005", "Either I get the highest grade or I am a failure.", 1),
    ("CBT006", "I made an error in the report and can correct it before tomorrow's review.", 0),
    ("CBT007", "My friend has not replied; she may be busy so I will check in later.", 0),
    ("CBT008", "One low grade is disappointing but it does not define my overall ability.", 0),
    ("CBT009", "I need help with this task and asking a colleague is a reasonable option.", 0),
    ("CBT010", "Tomorrow could be challenging or it could go smoothly; I cannot know yet.", 0),
)


class CBTHeaderSampleTests(unittest.TestCase):
    """Compare real CBT-header predictions with expected sample labels."""

    @classmethod
    def setUpClass(cls):
        cls.original_runtime_overrides = dict(cbt_model.CBT_RUNTIME_HF_ID_OVERRIDES)
        cbt_model.CBT_RUNTIME_HF_ID_OVERRIDES.update(
            {
                "BERT": _cached_model_path("bert-base-uncased"),
                "MentalBERT": _cached_model_path("bert-base-uncased"),
                "DeBERTa-v3": _cached_model_path("microsoft/deberta-v3-base"),
            }
        )
        cls.predictor = cbt_model.CBTPredictor()

    @classmethod
    def tearDownClass(cls):
        del cls.predictor
        cbt_model.CBT_RUNTIME_HF_ID_OVERRIDES.clear()
        cbt_model.CBT_RUNTIME_HF_ID_OVERRIDES.update(cls.original_runtime_overrides)
        gc.collect()

    def check_sample(self, case_id: str, text: str, expected: int):
        result = self.predictor.predict(text)
        actual = int(result.has_distortion)

        # generate_report.py collects these values from the test instance.
        self.case_id = case_id
        self.sample_text = text
        self.expected_label = expected
        self.predicted_label = actual
        self.positive_probability = float(result.probabilities["distortion"])
        self.confidence = float(result.confidence)

        self.assertEqual(
            actual,
            expected,
            msg=(
                f"{case_id}: expected distortion={bool(expected)}, "
                f"got {result.label!r} with confidence {result.confidence:.4f}"
            ),
        )


def _add_sample_tests() -> None:
    for case_id, text, expected in CBT_CASES:
        def test_method(self, values=(case_id, text, expected)):
            self.check_sample(*values)

        test_method.__name__ = f"test_{case_id.lower()}"
        test_method.__doc__ = text
        setattr(CBTHeaderSampleTests, test_method.__name__, test_method)


_add_sample_tests()


if __name__ == "__main__":
    unittest.main(verbosity=2)
