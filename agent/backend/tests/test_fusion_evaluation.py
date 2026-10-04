from __future__ import annotations

import unittest

from evaluate_fusion import Example, binary_metrics, evaluate


class FusionEvaluationTests(unittest.TestCase):
    def test_binary_metrics_for_perfect_predictions(self) -> None:
        metrics = binary_metrics([False, True, True], [False, True, True])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["f1_macro"], 1.0)

    def test_ablation_methods_can_produce_different_results(self) -> None:
        examples = [
            Example(0.8, 0.1, True),
            Example(0.1, 0.8, True),
            Example(0.1, 0.1, False),
        ]
        common = {
            "fusion_threshold": 0.5,
            "stress_threshold": 0.5,
            "cbt_threshold": 0.37,
        }
        or_metrics = evaluate(examples, method="or", **common)
        stress_metrics = evaluate(examples, method="stress_only", **common)
        self.assertGreater(or_metrics["recall"], stress_metrics["recall"])


if __name__ == "__main__":
    unittest.main()
