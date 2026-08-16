"""Headless end-to-end check of the C4 pipeline.

Runs a multi-turn conversation through every forecaster checkpoint plus the
rule fallback, exercising classification, deviation tracking, forecasting,
strategy selection and all three XAI methods. Prints a compact report and exits
non-zero on failure, so it is usable as a pre-demo sanity check.

    python smoke_test.py             # textcnn + bilstm + rule fallback
    python smoke_test.py --all       # also distilbert (265 MB, slower)
    python smoke_test.py --llm       # also load Qwen3-4B and generate for real
"""

import argparse
import json
import sys
import traceback

from c4_pipeline import run_c4_pipeline
from c4_pipeline.emotion_classifier import EmotionClassifier
from c4_pipeline.emotion_forecaster import EmotionForecaster
from config import (
    CURRENT_EMOTION_MODEL_PATH,
    EMOTION_LABELS,
    FORECAST_LABELS,
    LOW_VRAM,
    NEXT_EMOTION_MODEL_PATH,
    SMALL_MODEL_DEVICE,
)

CONVERSATION = [
    "I feel sad and stressed about my exams.",
    "I am getting scared that I will fail and disappoint my parents.",
    "I feel a little better after talking about it.",
    "Honestly some days I think about ending my life.",
]

failures = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"    ok   {message}")
    else:
        print(f"    FAIL {message}")
        failures.append(message)


def run_case(classifier, architecture, force_fallback):
    label = "rule-fallback" if force_fallback else architecture
    print(f"\n=== forecaster: {label} " + "=" * (54 - len(label)))
    forecaster = EmotionForecaster(
        NEXT_EMOTION_MODEL_PATH,
        labels=list(FORECAST_LABELS),
        force_fallback=force_fallback,
        architecture=architecture,
        device=SMALL_MODEL_DEVICE,
    )
    status = forecaster.status()
    print(f"  loaded={not status['fallback']} type={status['model_type']} "
          f"vocab={status['vocab_size']} max_len={status['max_len']} "
          f"err={status['load_error']}")
    if not force_fallback:
        check(not status["fallback"], f"{architecture} checkpoint loaded")

    state = {"messages": [], "emotion_history": [], "pipeline_traces": []}
    for turn, message in enumerate(CONVERSATION, start=1):
        trace = run_c4_pipeline(
            user_message=message,
            conversation_state=state,
            classifier=classifier,
            forecaster=forecaster,
            explain=True,
        )
        state["messages"].append({"role": "user", "content": message})
        state["messages"].append(
            {"role": "assistant", "content": trace["supportive_response"]}
        )

        print(
            f"  turn {turn}: {trace['current_emotion']:8s}"
            f"({trace['current_emotion_confidence']:.2f})"
            f" | prev={str(trace['previous_emotion']):8s}"
            f" dev={trace['deviation_level']:8s}"
            f" | next={trace['forecasted_next_emotion']:8s}"
            f"({trace['forecast_confidence']:.2f})"
            f" -> {trace['selected_strategy']}"
        )

        check(trace["current_emotion"] in EMOTION_LABELS, f"turn {turn} emotion in label set")
        probabilities = trace["forecast_probabilities"] or {}
        check(
            abs(sum(probabilities.values()) - 1.0) < 1e-3,
            f"turn {turn} forecast distribution sums to 1",
        )
        check(
            set(probabilities) == set(FORECAST_LABELS),
            f"turn {turn} forecast covers all 8 labels",
        )
        projected = trace["forecast_probabilities_projected"]
        check(
            abs(sum(projected.values()) - 1.0) < 1e-3,
            f"turn {turn} projected distribution sums to 1",
        )
        check(json.dumps(trace, default=float) is not None, f"turn {turn} trace is JSON-serialisable")

        explanations = trace["explanations"]
        check(bool(explanations["strategy_rules"]), f"turn {turn} strategy rule trace present")
        classifier_xai = explanations.get("classifier", {})
        check(classifier_xai.get("available", False), f"turn {turn} classifier IG available")
        if classifier_xai.get("available"):
            relative = classifier_xai["completeness_gap_relative"]
            check(
                relative < 0.15,
                f"turn {turn} IG relative completeness gap {relative:.3f} < 0.15",
            )
            check(
                bool(classifier_xai["top_tokens"]),
                f"turn {turn} classifier attributions non-empty",
            )
            occlusion = classifier_xai.get("occlusion", {})
            check(occlusion.get("available", False), f"turn {turn} occlusion available")

        forecaster_xai = explanations.get("forecaster", {})
        if not force_fallback:
            check(forecaster_xai.get("available", False), f"turn {turn} forecaster IG available")
            counterfactual = explanations.get("forecaster_counterfactual", {})
            check(
                len(counterfactual.get("sweep", {})) == 8,
                f"turn {turn} counterfactual sweep covers 8 current emotions",
            )

        if turn == 1 and not force_fallback:
            print("    IG (classifier):",
                  [(w, round(s, 2)) for w, s in classifier_xai["top_tokens"][:5]])
            print("    IG (forecaster):",
                  [(w, round(s, 2)) for w, s in forecaster_xai["top_tokens"][:5]])
            print("    aux ablation   :", {
                k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in forecaster_xai["aux_contribution"].items()
            })

    # The crisis line must route to Safe Fallback regardless of the emotion path.
    final = state["pipeline_traces"][-1]
    check(final["safety"]["risk_detected"], "crisis message flags a safety risk")
    check(final["selected_strategy"] == "Safe Fallback", "crisis message selects Safe Fallback")
    check(len(state["emotion_history"]) == len(CONVERSATION), "emotion history has one row per turn")

    # Context must actually accumulate across turns.
    if not force_fallback:
        check(
            "[supporter]" in final["forecast_context_text"],
            "dialogue context includes previous supporter turns",
        )
        check(
            "[SEP]" in final["forecast_context_text"],
            "dialogue context uses the trained [SEP] format",
        )


SAFETY_CASES = [
    ("I think about ending my life.", True),
    ("Some days I want to die.", True),
    ("I have been cutting myself again.", True),
    ("I wish I were dead.", True),
    ("I feel suicidal.", True),
    ("I'm dying to see that film.", False),
    ("My phone battery died during the exam.", False),
    ("I am stressed about my deadline.", False),
]


def run_safety_cases() -> None:
    from c4_pipeline.safety import check_safety

    print("\n=== safety screen " + "=" * 52)
    for message, expected in SAFETY_CASES:
        result = check_safety(message)
        check(
            result["risk_detected"] == expected,
            f"{'flags' if expected else 'ignores'}: {message!r}"
            + (f" (matched {result['matched_phrases']})" if result["matched_phrases"] else ""),
        )


REPLY_GRAPH_CASES = [
    # (message, current, forecast, strategy, crisis, expected route)
    ("I feel hopeless about my exams.", "sadness", "sad", "Comfort", False, "adapter"),
    ("I got the job I wanted!", "joy", "happy", "Maintain Tone", False, "base"),
    ("Nothing much, just a normal day.", "neutral", "neutral", "Maintain Tone", False, "base"),
    # Neutral now, deteriorating: the forecast alone must pull it to the adapter.
    ("It is fine I suppose.", "neutral", "stressed", "Reassure", False, "adapter"),
    ("I think about ending my life.", "sadness", "sad", "Safe Fallback", True, "crisis"),
]


def run_reply_graph_cases(with_llm: bool = False) -> None:
    """Routing, strategy translation and the guard.

    With no generator attached, every non-crisis case is expected to fall
    through to the template branch -- the graph must stay usable when the
    language model is absent, and that is the default path this checks.

    With `--llm`, Qwen3-4B is loaded and the same cases run for real, so the
    prompt, the generation and the output guard are exercised end to end.
    """
    from c4_pipeline.reply_graph import generate_supportive_reply, set_generator
    from c4_pipeline.strategy_mapping import ESCONV_STRATEGIES

    generator = None
    if with_llm:
        from c4_pipeline.qwen_generator import QwenReplyGenerator

        print("\n=== loading Qwen3-4B " + "=" * 49)
        generator = QwenReplyGenerator()
        generator.load()
        status = generator.status()
        print(f"  loaded={status['loaded']} adapter={status['adapter_loaded']} "
              f"device={status['device']} dtype={status['dtype']}")
        if status["load_error"]:
            print(f"  load_error: {status['load_error']}")
        if status["adapter_error"]:
            print(f"  adapter: {status['adapter_error']}")
        check(status["loaded"], "Qwen3-4B loaded")
        set_generator(generator)

    label = "Qwen3-4B attached" if with_llm else "no LLM attached"
    print(f"\n=== reply graph (LangGraph, {label}) " + "=" * max(4, 40 - len(label)))
    for message, current, forecast, strategy, crisis, expected_route in REPLY_GRAPH_CASES:
        state = generate_supportive_reply(
            user_message=message,
            current_emotion=current,
            current_emotion_confidence=0.8,
            forecast_emotion=forecast,
            forecast_emotion_projected=forecast,
            forecast_confidence=0.7,
            deviation_level="Low",
            deviation_score=0.2,
            strategy=strategy,
            safety={
                "risk_detected": crisis,
                "risk_type": "crisis_or_self_harm" if crisis else "none",
                "matched_phrases": ["ending my life"] if crisis else [],
            },
            dialogue_history=[("user", "hi"), ("supporter", "Hello, how are you?")],
        )
        nodes = [entry["node"] for entry in state["node_trace"]]
        print(f"  {expected_route:8s} | {strategy:13s} -> {' -> '.join(nodes)}")
        if with_llm:
            generation = state.get("generation") or {}
            print(f"           reply: {state['reply']}")
            if generation.get("available"):
                print(
                    f"           [{generation['completion_tokens']} tokens, "
                    f"{generation['latency_seconds']:.2f}s, "
                    f"adapter={'on' if generation['adapter_used'] else 'off'}]"
                )

        check(state["route"] == expected_route, f"{message[:28]!r} routes to {expected_route}")
        check(bool(state["reply"].strip()), f"{message[:28]!r} produced a non-empty reply")

        if expected_route == "crisis":
            # The whole point of the crisis branch: no model call at all.
            check("generate_reply" not in nodes, "crisis turn never calls the model")
            check(state["source"] == "safe_fallback", "crisis turn uses the fixed text")
        else:
            check(
                state.get("esconv_strategy") in ESCONV_STRATEGIES,
                f"{strategy} maps into ESConv's vocabulary "
                f"({state.get('esconv_strategy')})",
            )
            prompt = state.get("prompt_text", "")
            check("Conversation so far:" in prompt, f"{expected_route} prompt keeps the trained header")
            # The strategy line is an adapter-only cue; the base model never saw it.
            has_strategy_line = "Response strategy:" in prompt
            check(
                has_strategy_line == (expected_route == "adapter"),
                f"{expected_route} prompt "
                f"{'includes' if expected_route == 'adapter' else 'omits'} the strategy line",
            )
            if with_llm:
                # The model answered, so the guard must have accepted it and the
                # adapter must be on exactly for the adapter route.
                check(
                    state["source"].startswith("qwen"),
                    f"{expected_route} route answered by the model, not a template "
                    f"(source={state['source']})",
                )
                check(
                    (state.get("guard") or {}).get("passed", False),
                    f"{expected_route} generated reply passed the output guard",
                )
                adapter_used = (state.get("generation") or {}).get("adapter_used", False)
                if generator is not None and generator.adapter_loaded:
                    check(
                        adapter_used == (expected_route == "adapter"),
                        f"adapter {'enabled' if expected_route == 'adapter' else 'disabled'} "
                        f"on the {expected_route} route",
                    )
            else:
                check(
                    state["source"] == "template",
                    "falls back to a template with no LLM attached",
                )


def run_guard_cases() -> None:
    """The output guard must catch leaks without flagging ordinary support talk."""
    from c4_pipeline.reply_graph import guard_reply

    print("\n=== output guard " + "=" * 53)
    cases = [
        # --- must pass: ordinary supportive speech --------------------------
        ("That sounds really hard. I am sorry you are going through it.", True),
        ("I hear how sad you feel right now.", True),          # bare emotion word
        ("I know the anxiety feels overwhelming, and that makes sense.", True),
        ("At least the weather forecast looks better tomorrow.", True),  # not the pipeline sense
        ("You are not a bad person for feeling angry about it.", True),
        # --- must fail: the pipeline narrating itself -----------------------
        ("Your current emotion is sadness, so I will comfort you.", False),
        ("The classifier predicted you will feel worse tomorrow.", False),
        ("I am using Reflection of feelings to respond.", False),
        ("Based on the internal signals, your deviation level is High.", False),
        ("My model says you are 81% likely to stay sad.", False),
        ("You have clinical depression and should get medication.", False),
        ("", False),
    ]
    for text, should_pass in cases:
        result = guard_reply({"raw_reply": text})
        passed = result["guard"]["passed"]
        check(
            passed == should_pass,
            f"{'accepts' if should_pass else 'rejects'}: {text[:46]!r}"
            + ("" if passed == should_pass else f" (failures={result['guard']['failures']})"),
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="include the distilbert checkpoint")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="load Qwen3-4B and generate for real (needs a CUDA GPU and ~8 GB of downloads)",
    )
    args = parser.parse_args()

    if LOW_VRAM:
        print(f"LOW-VRAM MODE: classifier and forecaster pinned to {SMALL_MODEL_DEVICE}.")

    run_safety_cases()
    run_guard_cases()
    try:
        run_reply_graph_cases(with_llm=args.llm)
    except Exception:
        traceback.print_exc()
        failures.append("reply graph raised an exception")

    print("\n=== classifier " + "=" * 60)
    # Honours C4_LOW_VRAM so this is a true rehearsal of the app's configuration
    # on the machine it runs on, not a different one.
    classifier = EmotionClassifier(CURRENT_EMOTION_MODEL_PATH, device=SMALL_MODEL_DEVICE)
    status = classifier.status()
    print(f"  loaded={status['available']} name={status['model_name']} "
          f"device={status['device']} labels={status['labels']}")
    check(status["available"], "classifier checkpoint loaded")
    check(list(status["labels"]) == list(EMOTION_LABELS), "classifier labels match config")

    architectures = ["textcnn", "bilstm"] + (["distilbert"] if args.all else [])
    for architecture in architectures:
        try:
            run_case(classifier, architecture, force_fallback=False)
        except Exception:
            traceback.print_exc()
            failures.append(f"{architecture} raised an exception")
    try:
        run_case(classifier, "textcnn", force_fallback=True)
    except Exception:
        traceback.print_exc()
        failures.append("rule fallback raised an exception")

    print("\n" + "=" * 74)
    if failures:
        print(f"FAILED — {len(failures)} check(s):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
