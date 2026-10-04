from __future__ import annotations

import unittest
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from emotion_chain.fusion import InfluenceWeights
from emotion_chain.reply_generator import QwenReplyGenerator, ReplySignals


REPOSITORY_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = REPOSITORY_DIR / "model"


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    @staticmethod
    def apply_chat_template(*args, **kwargs):
        return torch.tensor([[1, 2]], dtype=torch.long)

    @staticmethod
    def decode(*args, **kwargs):
        return "generated reply"


class FakeModel:
    def __init__(self):
        self.embedding = nn.Embedding(3, 2)
        self.selected = []
        self.disable_count = 0

    def get_input_embeddings(self):
        return self.embedding

    def set_adapter(self, name):
        self.selected.append(name)

    @contextmanager
    def disable_adapter(self):
        self.disable_count += 1
        yield

    @staticmethod
    def generate(input_ids, **kwargs):
        suffix = torch.tensor([[2]], dtype=torch.long)
        return torch.cat((input_ids, suffix), dim=1)


class ReplyGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = QwenReplyGenerator(
            MODELS_DIR / "qwen3-4b-instruct",
            MODELS_DIR / "esconv_reply_adapter",
        )

    def test_crisis_status_does_not_load_qwen(self) -> None:
        result = self.generator.generate(
            "I am in danger",
            ReplySignals("sadness", "high_sadness", safety_status="crisis"),
        )
        self.assertEqual(result.route, "crisis")
        self.assertEqual(result.source, "safety_template")
        self.assertFalse(self.generator.is_loaded)

    def test_history_is_replayed_as_chat_turns(self) -> None:
        messages = self.generator._messages(
            "and today?",
            ReplySignals("sadness", "sadness"),
            "supportive_adapter",
            [("user", "yesterday was hard"), ("assistant", "that sounds heavy")],
        )
        # system prompt, then the earlier exchange, then the new message.
        self.assertEqual(
            [(item["role"], item["content"]) for item in messages[1:]],
            [
                ("user", "yesterday was hard"),
                ("assistant", "that sounds heavy"),
                ("user", "and today?"),
            ],
        )

    def test_history_turns_counts_exchanges_not_messages(self) -> None:
        self.generator.history_turns = 2
        history = [
            (role, f"{role} {index}")
            for index in range(4)
            for role in ("user", "assistant")
        ]
        messages = self.generator._messages(
            "latest",
            ReplySignals("sadness", "sadness"),
            "supportive_adapter",
            history,
        )
        # Two exchanges is four replayed messages, plus the new one.
        self.assertEqual(len(messages), 6)
        self.assertEqual(messages[1]["content"], "user 2")
        self.assertEqual(messages[-1]["content"], "latest")

    def test_zero_history_turns_replays_nothing(self) -> None:
        self.generator.history_turns = 0
        messages = self.generator._messages(
            "latest",
            ReplySignals("sadness", "sadness"),
            "supportive_adapter",
            [("user", "earlier"), ("assistant", "reply")],
        )
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[-1]["content"], "latest")

    def test_missing_runtime_packages_degrade_to_routed_template(self) -> None:
        result = self.generator.generate(
            "Today was difficult",
            ReplySignals("sadness", "low_sadness"),
        )
        self.assertEqual(result.route, "supportive_adapter")
        self.assertIn(result.source, {"template_fallback", "qwen"})

    def test_one_model_switches_between_base_and_adapter(self) -> None:
        fake_model = FakeModel()
        self.generator._model = fake_model
        self.generator._tokenizer = FakeTokenizer()

        base = self.generator.generate(
            "A good day",
            ReplySignals("joy", "high_joy"),
        )
        supportive = self.generator.generate(
            "A difficult day",
            ReplySignals("sadness", "low_sadness"),
        )
        self.assertEqual(base.route, "base")
        self.assertEqual(supportive.route, "supportive_adapter")
        self.assertEqual(fake_model.disable_count, 1)
        self.assertEqual(fake_model.selected, ["supportive"])
        self.assertIs(self.generator._model, fake_model)

    def test_extreme_turn_appends_the_configured_contact(self) -> None:
        self.generator._model = FakeModel()
        self.generator._tokenizer = FakeTokenizer()

        escalated = self.generator.generate(
            "Everything keeps getting worse",
            ReplySignals("sadness", "high_sadness", text_signal_flagged=True),
        )
        # One signal (the forecast) is an ordinary hard day, not an escalation.
        steady = self.generator.generate(
            "I feel overwhelmed today",
            ReplySignals("sadness", "high_sadness"),
        )

        self.assertTrue(escalated.escalated)
        self.assertIn("077766889", escalated.text)
        self.assertIn("generated reply", escalated.text)
        self.assertEqual(escalated.route, "supportive_adapter")
        # A negative turn on its own is not extreme; the reply stays as written.
        self.assertFalse(steady.escalated)
        self.assertNotIn("077766889", steady.text)
        self.assertEqual(steady.route, "supportive_adapter")

    def test_high_deviation_with_a_second_signal_escalates(self) -> None:
        self.generator._model = FakeModel()
        self.generator._tokenizer = FakeTokenizer()

        result = self.generator.generate(
            "It fell apart after all",
            ReplySignals(
                "sadness",
                "neutral",
                previous_emotion="joy",
                deviation_level="High",
                deviation_score=0.9,
                sensor_stress=True,
            ),
        )
        self.assertIn("077766889", result.text)

    def test_crisis_template_carries_the_contact(self) -> None:
        result = self.generator.generate(
            "I am in danger",
            ReplySignals("sadness", "high_sadness", safety_status="crisis"),
        )
        self.assertIn("077766889", result.text)
        self.assertEqual(result.support_contacts, self.generator.support_contacts)



class SignalInfluenceTests(unittest.TestCase):
    """The weights only scale how strongly a source is stated in the prompt."""

    def _generator(self, **weights) -> QwenReplyGenerator:
        return QwenReplyGenerator(
            MODELS_DIR / "qwen3-4b-instruct",
            MODELS_DIR / "esconv_reply_adapter",
            influence_weights=InfluenceWeights(**weights),
        )

    @staticmethod
    def _signals(**kwargs) -> ReplySignals:
        return ReplySignals("sadness", "low_sadness", **kwargs)

    def test_weight_scales_a_source_confidence(self) -> None:
        signals = self._signals(sensor_stress=True, sensor_stress_probability=0.62)
        full = self._generator(sensor=1.0)._signal_influence(signals)
        weak = self._generator(sensor=0.3)._signal_influence(signals)
        self.assertIn("body sensor (wearable): stress detected, influence 0.62", full[0])
        self.assertIn("body sensor (wearable): stress detected, influence 0.19", weak[0])

    def test_a_source_that_did_not_run_is_omitted(self) -> None:
        # Text mode: C2 never ran, so the model is not told the voice was calm.
        lines = self._generator()._signal_influence(
            self._signals(text_signal_flagged=True, text_signal_probability=0.7)
        )
        self.assertEqual(len(lines), 1)
        self.assertIn("text (what the user wrote", lines[0])

    def test_a_source_that_ran_and_found_nothing_still_reports(self) -> None:
        lines = self._generator()._signal_influence(
            self._signals(sensor_stress=False, sensor_stress_probability=0.12)
        )
        self.assertIn("no stress detected, influence 0.12", lines[0])

    def test_zero_weight_hides_the_source(self) -> None:
        signals = self._signals(voice_stress=True, voice_stress_score=0.9)
        self.assertEqual(self._generator(voice=0.0)._signal_influence(signals), [])

    def test_weights_do_not_change_the_route(self) -> None:
        """A down-weighted sensor still selects the supportive adapter."""

        signals = ReplySignals(
            "joy", "high_joy", sensor_stress=True, sensor_stress_probability=0.55
        )
        result = self._generator(sensor=0.01).generate("I am fine", signals)
        self.assertEqual(result.route, "supportive_adapter")

    def test_prompt_carries_the_weighted_lines(self) -> None:
        generator = self._generator(sensor=0.3, voice=0.6)
        messages = generator._messages(
            "hello",
            self._signals(
                voice_stress=True,
                voice_stress_score=0.8,
                voice_appraisal_state="threat",
            ),
            "supportive_adapter",
            (),
        )
        system = messages[0]["content"]
        self.assertIn("influence 0.48 of 1.00", system)
        self.assertIn("appraisal state=threat", system)
        self.assertNotIn("voice appraisal stress=True", system)

    def test_out_of_range_weight_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            InfluenceWeights(sensor=1.4)



if __name__ == "__main__":
    unittest.main()
