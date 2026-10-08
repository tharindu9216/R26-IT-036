from __future__ import annotations

import unittest

from emotion_chain.fusion import fuse_stress_and_cbt


class FusionTests(unittest.TestCase):
    def test_equal_weight_average(self) -> None:
        fused = fuse_stress_and_cbt(0.8, 0.2)
        self.assertAlmostEqual(fused.probability, 0.5)
        self.assertTrue(fused.is_flagged)

    def test_below_threshold_is_not_flagged(self) -> None:
        fused = fuse_stress_and_cbt(0.2, 0.2, threshold=0.5)
        self.assertAlmostEqual(fused.probability, 0.2)
        self.assertFalse(fused.is_flagged)

    def test_custom_weights_favor_the_heavier_header(self) -> None:
        fused = fuse_stress_and_cbt(
            0.9, 0.1, stress_weight=0.1, cbt_weight=0.9, threshold=0.5
        )
        self.assertAlmostEqual(fused.probability, 0.9 * 0.1 + 0.1 * 0.9)
        self.assertFalse(fused.is_flagged)

    def test_weights_are_normalized(self) -> None:
        # Un-normalized weights (e.g. 2 and 2) should behave like 0.5/0.5.
        fused = fuse_stress_and_cbt(0.8, 0.2, stress_weight=2.0, cbt_weight=2.0)
        self.assertAlmostEqual(fused.probability, 0.5)

    def test_raw_component_probabilities_are_preserved(self) -> None:
        fused = fuse_stress_and_cbt(0.7, 0.3)
        self.assertEqual(fused.stress_probability, 0.7)
        self.assertEqual(fused.cbt_probability, 0.3)

    def test_rejects_non_positive_total_weight(self) -> None:
        with self.assertRaises(ValueError):
            fuse_stress_and_cbt(0.5, 0.5, stress_weight=0.0, cbt_weight=0.0)

    def test_rejects_out_of_range_threshold(self) -> None:
        with self.assertRaises(ValueError):
            fuse_stress_and_cbt(0.5, 0.5, threshold=1.5)

    def test_or_ablation_preserves_either_header_decision(self) -> None:
        fused = fuse_stress_and_cbt(
            0.6,
            0.1,
            method="or",
            stress_flagged=True,
            cbt_flagged=False,
        )
        self.assertTrue(fused.is_flagged)
        self.assertEqual(fused.method, "or")

    def test_single_header_ablations(self) -> None:
        stress_only = fuse_stress_and_cbt(
            0.6, 0.1, method="stress_only", stress_flagged=True
        )
        cbt_only = fuse_stress_and_cbt(
            0.6, 0.1, method="cbt_only", cbt_flagged=False
        )
        self.assertTrue(stress_only.is_flagged)
        self.assertFalse(cbt_only.is_flagged)

    def test_rejects_unknown_ablation(self) -> None:
        with self.assertRaises(ValueError):
            fuse_stress_and_cbt(0.5, 0.5, method="unknown")


if __name__ == "__main__":
    unittest.main()
