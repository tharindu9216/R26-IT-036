from __future__ import annotations

import io
import unittest
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

from c2 import decode_audio
from emotion_chain.agent_graph import AgentComponents, build_voice_supportive_graph
from emotion_chain.reply_generator import ReplyGenerationResult


REPOSITORY_DIR = Path(__file__).resolve().parents[2]


class AppraisalArtifactTests(unittest.TestCase):
    def test_migrated_artifact_loads_under_c2_package(self) -> None:
        artifact = REPOSITORY_DIR / "model/c2/appraisal/advanced_appraisal_engine.joblib"
        engine = joblib.load(artifact)
        self.assertEqual(type(engine).__module__, "c2.appraisal_stress.engine")
        sample = pd.DataFrame(
            [
                {
                    "file_name": "test.wav",
                    "actual_class": "unknown",
                    "arousal": 0.5,
                    "dominance": 0.5,
                    "valence": 0.5,
                }
            ]
        )
        result = engine.transform(sample).iloc[0]
        self.assertIn(result["dominant_appraisal_state"], {
            "threat",
            "challenge",
            "withdrawal_distress",
            "negative_activation",
            "positive_activation",
            "calm_recovery",
        })

    def test_browser_audio_decode_contract(self) -> None:
        import soundfile as sf

        waveform = np.sin(np.linspace(0.0, 8.0 * np.pi, 16_000)).astype(np.float32)
        payload = io.BytesIO()
        sf.write(payload, waveform, 16_000, format="WAV")
        decoded = decode_audio(payload.getvalue())
        self.assertEqual(decoded.ndim, 1)
        self.assertEqual(decoded.shape[0], waveform.shape[0])


class VoiceGraphTests(unittest.TestCase):
    def test_voice_graph_skips_c3_and_routes_with_c2_signal(self) -> None:
        class FailingC3:
            @staticmethod
            def predict(text):
                raise AssertionError("Voice graph must not call C3")

        class FakeEmotionChain:
            @staticmethod
            def predict(text):
                prediction = lambda label, confidence: SimpleNamespace(
                    label=label,
                    confidence=confidence,
                    probabilities={label: confidence},
                )
                return SimpleNamespace(
                    current=prediction("joy", 0.9),
                    next=prediction("high_joy", 0.8),
                    forecast=SimpleNamespace(
                        label="high",
                        probability_high=0.8,
                        threshold=0.595,
                        baseline_probability=0.75,
                        fusion_probability=0.85,
                        confidence=0.8,
                    ),
                )

        class FakeReplyGenerator:
            @staticmethod
            def generate(text, signals, history=()):
                route = "supportive_adapter" if signals.voice_stress else "base"
                return ReplyGenerationResult("voice reply", route, "fake")

        components = AgentComponents(
            emotion_chain=FakeEmotionChain(),
            reply_generator=FakeReplyGenerator(),
            stress_predictor=FailingC3(),
            cbt_predictor=FailingC3(),
        )
        graph = build_voice_supportive_graph(components)
        result = graph.invoke(
            {
                "user_text": "I am speaking",
                "voice_stress": True,
                "voice_stress_score": 0.7,
                "voice_appraisal_state": "threat",
                "voice_appraisal_uncertain": False,
            }
        )
        self.assertEqual(result["reply_route"], "supportive_adapter")
        self.assertNotIn("text_stress", result)
        self.assertNotIn("has_cbt_distortion", result)


if __name__ == "__main__":
    unittest.main()
