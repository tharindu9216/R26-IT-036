from __future__ import annotations

import unittest
from pathlib import Path

import torch

from emotion_chain.models import build_state_forecaster


REPOSITORY_DIR = Path(__file__).resolve().parents[2]
FORECAST_DIR = REPOSITORY_DIR / "model/next_emotion_forecaster"


class ExportedCheckpointTests(unittest.TestCase):
    def test_all_architectures_strict_load_and_forward(self) -> None:
        paths = sorted(FORECAST_DIR.glob("*_state_forecast.pt"))
        self.assertEqual(len(paths), 4)
        for path in paths:
            with self.subTest(checkpoint=path.name):
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                model = build_state_forecaster(checkpoint)
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                model.eval()
                tokens = torch.zeros((2, int(checkpoint["max_len"])), dtype=torch.long)
                tokens[:, :4] = torch.tensor([3, 27, 74, 2])
                emotion_ids = torch.tensor([0, 4], dtype=torch.long)
                with torch.inference_mode():
                    logits = model(tokens, emotion_ids)
                self.assertEqual(tuple(logits.shape), (2, 13))
                self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
