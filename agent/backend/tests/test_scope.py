"""Out-of-scope requests: the input screen, the reply filter, and the graph."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from emotion_chain.agent_graph import AgentComponents, SupportiveAgentWorkflow
from emotion_chain.reply_generator import QwenReplyGenerator, ReplySignals
from emotion_chain.scope import detect_out_of_scope, reply_contains_code
from emotion_chain.trace import build_trace


MODELS_DIR = Path(__file__).resolve().parents[2] / "model"


class DetectOutOfScopeTests(unittest.TestCase):
    def test_task_requests_are_flagged_by_category(self) -> None:
        cases = {
            "give me a python code to sort a list": "coding",
            "Write a Python script that reads a CSV": "coding",
            "can you write me a function to reverse a string": "coding",
            "make a calculator in javascript": "coding",
            "def foo():\n    return 1\nwhy doesn't this work": "coding",
            "solve 2x + 3 = 7": "maths or homework",
            "what is 25 * 4": "maths or homework",
            "do my homework": "maths or homework",
            "translate this to French": "translation",
            "how do you say thank you in japanese": "translation",
            "write an essay about climate change": "writing",
            "draft a cover letter for a software job": "writing",
        }
        for text, category in cases.items():
            with self.subTest(text=text):
                self.assertEqual(detect_out_of_scope(text), category)

    def test_feelings_that_mention_a_task_stay_in_scope(self) -> None:
        for text in (
            "I'm so stressed about my python exam tomorrow",
            "I can't finish my program at uni, I feel like a failure",
            "I want to fix the error I made with my friend",
            "how do I solve this problem with my boss",
            "I'm struggling with python and feel stupid",
            "I can't translate my feelings into words",
            "give me a method to calm down",
            "I can't make my class today, I'm too anxious",
            "I want to write a letter to my late mother",
            "I feel like I can't function anymore",
        ):
            with self.subTest(text=text):
                self.assertIsNone(detect_out_of_scope(text))

    def test_code_in_a_reply_is_recognised(self) -> None:
        self.assertTrue(reply_contains_code("Sure:\n```python\nprint('x')\n```"))
        self.assertTrue(reply_contains_code("def add(a, b):\n    return a + b"))
        self.assertFalse(reply_contains_code("That sounds hard. What's on your mind?"))
        self.assertFalse(reply_contains_code("Try to import some calm into your evening."))


class ScopedReplyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = QwenReplyGenerator(
            MODELS_DIR / "qwen3-4b-instruct",
            MODELS_DIR / "esconv_reply_adapter",
            support_contacts=(),
        )
        self.generator._model = object()
        self.generator._tokenizer = object()

    def _generate(self, text: str, completion: str, **signals):
        with patch.object(
            self.generator, "_local_completion", return_value=completion
        ) as completion_mock:
            result = self.generator.generate(
                text, ReplySignals("neutral", "neutral", **signals)
            )
        system_prompt = completion_mock.call_args.args[0][0]["content"]
        return result, system_prompt

    def test_every_prompt_carries_the_scope_rules(self) -> None:
        _, prompt = self._generate("hello", "Hi there, how are you?")
        self.assertIn("You do not write or fix code", prompt)
        self.assertNotIn("seems to involve", prompt)

    def test_a_flagged_request_adds_the_directive(self) -> None:
        result, prompt = self._generate(
            "write me a python script",
            "I can't help with code here, but how are you feeling?",
        )
        self.assertIn("seems to involve writing or fixing code", prompt)
        self.assertEqual(result.source, "qwen")
        self.assertEqual(result.out_of_scope_category, "coding")

    def test_a_reply_with_code_is_replaced(self) -> None:
        result, _ = self._generate(
            "write me a python script", "Sure!\n```python\nprint('hi')\n```"
        )
        self.assertEqual(result.source, "scope_template")
        self.assertNotIn("print(", result.text)
        self.assertIn("emotional support", result.text)

    def test_code_is_filtered_even_when_the_screen_missed_the_request(self) -> None:
        result, _ = self._generate(
            "can you help me with a thing", "def thing():\n    pass"
        )
        self.assertEqual(result.source, "scope_template")
        self.assertEqual(result.out_of_scope_category, "coding")

    def test_the_graph_screen_result_is_used_instead_of_rechecking(self) -> None:
        _, prompt = self._generate(
            "write me a python script",
            "How are you feeling?",
            scope_checked=True,
            out_of_scope_category=None,
        )
        self.assertNotIn("seems to involve", prompt)

    def test_fallback_for_an_off_topic_request_does_not_offer_help(self) -> None:
        generator = QwenReplyGenerator(
            MODELS_DIR / "qwen3-4b-instruct",
            MODELS_DIR / "esconv_reply_adapter",
            support_contacts=(),
        )
        with patch.object(generator, "load", side_effect=RuntimeError("no model")):
            result = generator.generate(
                "translate this to French", ReplySignals("neutral", "neutral")
            )
        self.assertEqual(result.source, "template_fallback")
        self.assertIn("translating text", result.text)


class ScopeNodeTests(unittest.TestCase):
    def test_scope_node_sets_state_and_trace(self) -> None:
        workflow = SupportiveAgentWorkflow(
            AgentComponents(emotion_chain=SimpleNamespace(), reply_generator=None)
        )
        state = workflow.scope_node({"user_text": "solve 2x + 3 = 7"})
        self.assertEqual(
            state,
            {"scope_status": "out_of_scope", "out_of_scope_category": "maths or homework"},
        )
        steps = {step["key"]: step for step in build_trace({"user_text": "x", **state})["steps"]}
        self.assertEqual(steps["scope"]["summary"], "out of scope")
        self.assertEqual(steps["scope"]["outputs"][0]["value"], "maths or homework")

    def test_crisis_turns_never_reach_the_scope_screen(self) -> None:
        steps = {
            step["key"]: step
            for step in build_trace({"user_text": "x", "safety_status": "crisis"})["steps"]
        }
        self.assertEqual(steps["scope"]["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
