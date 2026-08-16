"""LangGraph reply-generation stage for C4.

The EDA notebook's closing recommendation was specific about the shape this
stage should take:

    1. Fine-tune the ESConv LoRA adapter for negative/supportive situations.
    2. Preserve the original train, validation, and test splits.
    3. Route positive and neutral inputs to base Qwen3-4B with suitable prompts.
    4. Pass outputs from your other models through the LangGraph runtime prompt.

Points 3 and 4 are what this module implements. The graph is a graph rather than
a chain because two of its edges are genuinely conditional: crisis turns bypass
generation entirely, and a reply that fails the output guard loops back to be
regenerated under stricter settings.

    START
      |
      v
    route_request ------- crisis ------> crisis_reply -----------------+
      |                                                               |
      | adapter / base                                                |
      v                                                               |
    plan_response                                                     |
      |                                                               |
      v                                                               |
    build_prompt <--------------- retry ---------------+              |
      |                                                |              |
      v                                                |              |
    generate_reply                                     |              |
      |                                                |              |
      v                                                |              |
    guard_reply --- pass ---------------------------------------------+--> END
      |     |                                          |              |
      |     +---- retry (attempt < max) ---------------+              |
      |                                                               |
      +---- fail (budget spent / no model) --> template_fallback -----+

Every node appends to `node_trace`, so the UI can show the path the turn
actually took instead of asserting one.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict

from langgraph.graph import END, START, StateGraph

from config import (
    LLM_MAX_CONTEXT_TURNS,
    LLM_MAX_NEW_TOKENS,
    LLM_MAX_REPLY_SENTENCES,
    LLM_MAX_RETRIES,
    LLM_MIN_REPLY_WORDS,
    LLM_RETRY_TEMPERATURE,
    LLM_TEMPERATURE,
)
from .response_generator import TEMPLATES, generate_response
from .strategy_mapping import (
    ROUTE_ADAPTER,
    ROUTE_BASE,
    ROUTE_CRISIS,
    format_dialogue,
    mapping_reason,
    response_plan,
    select_route,
    to_esconv_strategy,
)

# Mirrors `SYSTEM_PROMPT` in qwen3_esconv_finetune/preprocess.py. The adapter saw
# this exact string in every one of its 12,235 training examples; changing a word
# here moves the model off the prompt it was tuned against.
TRAINED_SYSTEM_PROMPT = (
    "You are a supportive conversational assistant. Generate the next natural "
    "assistant reply from the conversation context. Follow the response strategy "
    "when one is provided. Be empathetic, relevant, concise, and non-judgmental. "
    "Do not mention internal labels, models, predictions, or datasets. Do not "
    "diagnose the user or make unsupported medical claims. Return only the reply."
)

# The base route never went through fine-tuning, so it gets an explicit brief
# instead of relying on learned behaviour.
BASE_SYSTEM_PROMPT = (
    "You are a supportive conversational assistant talking with someone who is "
    "not in distress. Match their energy without exaggerating it, respond to what "
    "they actually said, and keep it to one to three natural sentences. Do not "
    "offer consolation for a problem they have not raised. Do not mention "
    "internal labels, models, predictions, or datasets. Do not diagnose the user "
    "or make unsupported medical claims. Return only the reply."
)

# Vocabulary that indicates the model is narrating the pipeline instead of
# answering.
#
# Two calibration decisions here. Bare emotion words are deliberately NOT
# matched -- "I hear how sad you feel" is ordinary supportive speech, and a
# guard that rejects it would reject most correct replies. Conversely the
# pipeline terms are anchored to their meta sense rather than matched bare:
# "forecast" alone would fire on "the weather forecast", and "the model" alone
# on "the model UN conference". Where a phrase is ambiguous the guard still
# errs toward rejecting, because the cost is one extra generation pass and the
# cost of a miss is the demo explaining its own internals to a distressed user.
_LEAK_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    ("names_a_strategy", re.compile(
        r"\b(?:response\s+strategy|reflection of feelings|restatement or paraphrasing"
        r"|affirmation and reassurance|providing suggestions|self-disclosure"
        r"|maintain tone|safe fallback)\b", re.I)),
    ("names_the_pipeline", re.compile(
        r"\b(?:classifier|forecaster|language model|training data|dataset"
        r"|confidence (?:score|level)|probability distribution|sensor reading"
        r"|(?:my|the|our)\s+model\b|internal (?:signal|label)s?)", re.I)),
    ("reports_an_internal_label", re.compile(
        r"\b(?:current|detected|predicted|forecast(?:ed)?)\s+emotion(?:al state)?\b"
        r"|\bemotion label\b|\bdeviation (?:score|level)\b|\bvalence\b"
        r"|\bpredicted (?:that\s+)?you(?:'ll| will)\b", re.I)),
    ("diagnoses", re.compile(
        r"\byou (?:have|are suffering from|are experiencing)\s+"
        r"(?:clinical\s+)?(?:depression|anxiety disorder|ptsd|bipolar)\b"
        r"|\byou (?:are|'re) (?:clinically|medically)\b|\bdiagnos\w*", re.I)),
)

# Cosmetic artefacts stripped before the guard runs, so a leading "Assistant:"
# costs a regex rather than a whole extra generation pass.
_ROLE_PREFIX = re.compile(r"^\s*(?:assistant|supporter|reply|response)\s*:\s*", re.I)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class ReplyState(TypedDict, total=False):
    """State carried between nodes.

    Upstream C4 signals arrive as inputs; everything from `route` down is
    produced by the graph.
    """

    # --- inputs from the C4 pipeline ---------------------------------------
    user_message: str
    dialogue_history: List[Tuple[str, str]]
    current_emotion: str
    current_emotion_confidence: float
    forecast_emotion: str
    forecast_emotion_projected: str
    forecast_confidence: float
    deviation_level: str
    deviation_score: float
    strategy: str
    safety: Dict[str, Any]

    # --- routing / planning -------------------------------------------------
    route: str
    route_reason: str
    current_valence: str
    forecast_valence: str
    esconv_strategy: str
    response_plan: str
    strategy_mapping_reason: str

    # --- prompting / generation --------------------------------------------
    prompt_messages: List[Dict[str, str]]
    prompt_text: str
    attempts: int
    raw_reply: str

    # --- outputs ------------------------------------------------------------
    reply: str
    source: str
    guard: Dict[str, Any]
    generation: Dict[str, Any]
    node_trace: List[Dict[str, Any]]


def _trace(state: ReplyState, node: str, detail: str, **extra: Any) -> List[Dict[str, Any]]:
    entry: Dict[str, Any] = {"node": node, "detail": detail}
    entry.update(extra)
    return list(state.get("node_trace") or []) + [entry]


# ---------------------------------------------------------------------- nodes
def route_request(state: ReplyState) -> Dict[str, Any]:
    """Decide crisis / adapter / base before any text is generated."""
    safety = state.get("safety") or {}
    decision = select_route(
        safety_risk_detected=bool(safety.get("risk_detected")),
        current_emotion=state.get("current_emotion", ""),
        forecast_emotion=state.get("forecast_emotion_projected")
        or state.get("forecast_emotion", ""),
    )
    return {
        "route": decision["route"],
        "route_reason": decision["reason"],
        "current_valence": decision["current_valence"],
        "forecast_valence": decision["forecast_valence"],
        "attempts": 0,
        "node_trace": _trace(
            state,
            "route_request",
            decision["reason"],
            route=decision["route"],
        ),
    }


def plan_response(state: ReplyState) -> Dict[str, Any]:
    """Translate the C4 strategy into ESConv's vocabulary plus a prose plan."""
    strategy = state.get("strategy", "Listen")
    esconv_strategy = to_esconv_strategy(strategy)
    plan = response_plan(strategy)
    reason = mapping_reason(strategy)
    return {
        "esconv_strategy": esconv_strategy,
        "response_plan": plan,
        "strategy_mapping_reason": reason,
        "node_trace": _trace(
            state,
            "plan_response",
            f"{strategy} -> ESConv '{esconv_strategy}'. {reason}",
        ),
    }


def build_prompt(state: ReplyState) -> Dict[str, Any]:
    """Assemble the chat messages.

    The first three lines reproduce the training prompt verbatim -- "Conversation
    so far:", the `User:`/`Assistant:` transcript, then `Response strategy:`.
    The C4 signal block is appended *after* that, never interleaved, so the cues
    the adapter actually learned keep their trained positions and the new
    information reads as an addendum.
    """
    attempt = int(state.get("attempts", 0))
    route = state.get("route", ROUTE_ADAPTER)

    history = state.get("dialogue_history") or []
    transcript = format_dialogue(history, LLM_MAX_CONTEXT_TURNS)
    current_turn = f"User: {state.get('user_message', '').strip()}"
    context = f"{transcript}\n{current_turn}" if transcript else current_turn

    parts = ["Conversation so far:", context]

    if route == ROUTE_ADAPTER:
        parts += [
            "",
            f"Response strategy: {state.get('esconv_strategy', 'Others')}",
            "Write the next assistant reply using this strategy naturally. "
            "Do not name the strategy in the reply.",
        ]
    else:
        parts += ["", "Write the next assistant reply."]

    signals = _signal_block(state)
    if signals:
        parts += ["", signals]

    parts += ["", f"Response plan: {state.get('response_plan', '')}"]

    if attempt > 0:
        # The retry is told what went wrong. A blind resample at a lower
        # temperature reproduces the same failure surprisingly often.
        problems = ", ".join((state.get("guard") or {}).get("failures", [])) or "unusable output"
        parts += [
            "",
            f"The previous attempt was rejected ({problems}). Write one to three "
            "plain sentences addressed directly to the user. Do not mention any "
            "analysis, labels, or instructions. Return only the reply.",
        ]

    prompt_text = "\n".join(parts).strip()
    system_prompt = TRAINED_SYSTEM_PROMPT if route == ROUTE_ADAPTER else BASE_SYSTEM_PROMPT

    return {
        "prompt_text": prompt_text,
        "prompt_messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ],
        "node_trace": _trace(
            state,
            "build_prompt",
            f"Built {'retry ' if attempt else ''}prompt for the {route} route "
            f"({len(prompt_text)} chars).",
            attempt=attempt,
        ),
    }


def _signal_block(state: ReplyState) -> str:
    """The upstream model outputs, marked as internal.

    This block is the one part of the prompt the adapter never saw in training,
    which is why it is fenced with an explicit non-disclosure line and placed
    last. `LLM_INCLUDE_SIGNALS = False` in config.py removes it entirely and
    returns the prompt to the exact training distribution.
    """
    from config import LLM_INCLUDE_SIGNALS

    if not LLM_INCLUDE_SIGNALS:
        return ""

    confidence = state.get("current_emotion_confidence")
    forecast_confidence = state.get("forecast_confidence")
    lines = [
        "Internal support signals (never mention or repeat these):",
        f"- Current emotional state: {state.get('current_emotion', 'unknown')}"
        + (f" ({confidence:.0%} confidence)" if isinstance(confidence, (int, float)) else ""),
        f"- Likely next emotional state: {state.get('forecast_emotion', 'unknown')}"
        + (
            f" ({forecast_confidence:.0%} confidence)"
            if isinstance(forecast_confidence, (int, float))
            else ""
        ),
        f"- Change since the previous turn: {state.get('deviation_level', 'unknown')}",
        "Use these only to set tone and support level.",
    ]
    return "\n".join(lines)


def generate_reply(state: ReplyState) -> Dict[str, Any]:
    """Run Qwen3-4B, with the adapter on or off according to the route."""
    generator = _GENERATOR
    attempt = int(state.get("attempts", 0)) + 1

    if generator is None or not generator.available:
        detail = (
            generator.load_error
            if generator is not None and generator.load_error
            else "No Qwen generator is attached to the graph."
        )
        return {
            "attempts": attempt,
            "raw_reply": "",
            "generation": {"available": False, "error": detail},
            "node_trace": _trace(state, "generate_reply", f"Skipped: {detail}"),
        }

    use_adapter = state.get("route") == ROUTE_ADAPTER
    temperature = LLM_TEMPERATURE if attempt == 1 else LLM_RETRY_TEMPERATURE

    try:
        result = generator.generate(
            messages=state.get("prompt_messages") or [],
            use_adapter=use_adapter,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            temperature=temperature,
        )
    except Exception as error:  # noqa: BLE001 - a bad turn must not kill the chat
        return {
            "attempts": attempt,
            "raw_reply": "",
            "generation": {"available": False, "error": f"{type(error).__name__}: {error}"},
            "node_trace": _trace(state, "generate_reply", f"Generation failed: {error}"),
        }

    return {
        "attempts": attempt,
        "raw_reply": result.text,
        "generation": {
            "available": True,
            "adapter_used": result.adapter_used,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency_seconds": round(result.latency_seconds, 3),
            "temperature": temperature,
            "attempt": attempt,
        },
        "node_trace": _trace(
            state,
            "generate_reply",
            f"Attempt {attempt} produced {result.completion_tokens} tokens in "
            f"{result.latency_seconds:.2f}s "
            f"({'adapter' if result.adapter_used else 'base model'}, T={temperature}).",
        ),
    }


def guard_reply(state: ReplyState) -> Dict[str, Any]:
    """Clean the output, then check it is a reply rather than a leak."""
    cleaned, cosmetic = _clean(state.get("raw_reply", ""))
    failures: List[str] = []

    if not cleaned:
        failures.append("empty output")
    elif len(cleaned.split()) < LLM_MIN_REPLY_WORDS:
        failures.append(f"shorter than {LLM_MIN_REPLY_WORDS} words")

    sentences = [s for s in _SENTENCE_SPLIT.split(cleaned) if s.strip()]
    if len(sentences) > LLM_MAX_REPLY_SENTENCES:
        failures.append(f"{len(sentences)} sentences, limit is {LLM_MAX_REPLY_SENTENCES}")

    leaks: List[Dict[str, str]] = []
    for name, pattern in _LEAK_PATTERNS:
        found = pattern.search(cleaned)
        if found:
            leaks.append({"check": name, "matched": found.group(0)})
            failures.append(name.replace("_", " "))

    passed = not failures
    guard = {
        "passed": passed,
        "failures": failures,
        "leaks": leaks,
        "cosmetic_fixes": cosmetic,
        "sentence_count": len(sentences),
        "word_count": len(cleaned.split()),
    }

    detail = (
        "Output accepted."
        if passed
        else "Rejected: " + "; ".join(failures)
    )
    return {
        "reply": cleaned if passed else "",
        "guard": guard,
        "node_trace": _trace(state, "guard_reply", detail, passed=passed),
    }


def crisis_reply(state: ReplyState) -> Dict[str, Any]:
    """Fixed safe-fallback text. Never model-generated, by design.

    A crisis turn is the one place where a fluent, plausible, slightly-wrong
    sentence is worse than a rigid correct one, so the language model is not
    consulted at all.
    """
    safety = state.get("safety") or {}
    return {
        "reply": TEMPLATES["Safe Fallback"],
        "source": "safe_fallback",
        "guard": {"passed": True, "failures": [], "leaks": [], "cosmetic_fixes": []},
        "node_trace": _trace(
            state,
            "crisis_reply",
            "Returned the fixed safe-fallback message; the language model was "
            f"bypassed (risk type: {safety.get('risk_type', 'unknown')}).",
        ),
    }


def template_fallback(state: ReplyState) -> Dict[str, Any]:
    """Deterministic template when generation is unavailable or unusable."""
    strategy = state.get("strategy", "Listen")
    template = generate_response(
        user_message=state.get("user_message", ""),
        current_emotion=state.get("current_emotion", ""),
        forecasted_emotion=state.get("forecast_emotion_projected", ""),
        strategy=strategy,
    )
    generation = state.get("generation") or {}
    if not generation.get("available", False):
        why = generation.get("error", "the generator is unavailable")
    else:
        why = (
            f"{state.get('attempts', 0)} generation attempt(s) failed the output "
            f"guard ({'; '.join((state.get('guard') or {}).get('failures', []))})"
        )
    return {
        "reply": template["response"],
        "source": "template",
        "node_trace": _trace(
            state, "template_fallback", f"Used the {strategy} template because {why}."
        ),
    }


def finalize(state: ReplyState) -> Dict[str, Any]:
    """Label the accepted generation with the route that produced it."""
    route = state.get("route", ROUTE_BASE)
    source = "qwen_adapter" if route == ROUTE_ADAPTER else "qwen_base"
    if not (state.get("generation") or {}).get("adapter_used", False):
        source = "qwen_base"
    return {
        "source": source,
        "node_trace": _trace(state, "finalize", f"Reply accepted from {source}."),
    }


# ----------------------------------------------------------------- conditionals
def _after_routing(state: ReplyState) -> str:
    return "crisis" if state.get("route") == ROUTE_CRISIS else "generate"


def _after_guard(state: ReplyState) -> str:
    """pass -> finalize, retry -> build_prompt, fail -> template_fallback."""
    if (state.get("guard") or {}).get("passed"):
        return "pass"
    if not (state.get("generation") or {}).get("available", False):
        return "fail"  # no model attached; retrying cannot help
    if int(state.get("attempts", 0)) <= LLM_MAX_RETRIES:
        return "retry"
    return "fail"


# ------------------------------------------------------------------ text tidying
def _clean(text: str) -> Tuple[str, List[str]]:
    """Strip artefacts that are formatting noise rather than content problems."""
    fixes: List[str] = []
    cleaned = (text or "").strip()

    stripped = _ROLE_PREFIX.sub("", cleaned)
    if stripped != cleaned:
        fixes.append("removed a role prefix")
        cleaned = stripped

    if len(cleaned) > 1 and cleaned[0] in "\"'" and cleaned[-1] == cleaned[0]:
        cleaned = cleaned[1:-1].strip()
        fixes.append("removed wrapping quotes")

    # A reply cut off by the token budget ends mid-sentence; drop the fragment
    # rather than shipping half a thought.
    if cleaned and cleaned[-1] not in ".!?…\"'":
        sentences = _SENTENCE_SPLIT.split(cleaned)
        if len(sentences) > 1:
            cleaned = " ".join(sentences[:-1]).strip()
            fixes.append("dropped a truncated final sentence")

    return cleaned.strip(), fixes


# ---------------------------------------------------------------------- assembly
_GENERATOR = None  # module-level so nodes stay picklable for LangGraph


def build_reply_graph():
    """Compile the graph. Structure is static; only the generator is injected."""
    graph = StateGraph(ReplyState)

    graph.add_node("route_request", route_request)
    graph.add_node("plan_response", plan_response)
    graph.add_node("build_prompt", build_prompt)
    graph.add_node("generate_reply", generate_reply)
    graph.add_node("guard_reply", guard_reply)
    graph.add_node("finalize", finalize)
    graph.add_node("crisis_reply", crisis_reply)
    graph.add_node("template_fallback", template_fallback)

    graph.add_edge(START, "route_request")
    graph.add_conditional_edges(
        "route_request",
        _after_routing,
        {"crisis": "crisis_reply", "generate": "plan_response"},
    )
    graph.add_edge("plan_response", "build_prompt")
    graph.add_edge("build_prompt", "generate_reply")
    graph.add_edge("generate_reply", "guard_reply")
    graph.add_conditional_edges(
        "guard_reply",
        _after_guard,
        {"pass": "finalize", "retry": "build_prompt", "fail": "template_fallback"},
    )
    graph.add_edge("finalize", END)
    graph.add_edge("crisis_reply", END)
    graph.add_edge("template_fallback", END)

    return graph.compile()


_COMPILED_GRAPH = None


def get_reply_graph():
    """The compiled graph, built once per process."""
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_reply_graph()
    return _COMPILED_GRAPH


def set_generator(generator) -> None:
    """Attach (or clear, with None) the Qwen generator the graph should call."""
    global _GENERATOR
    _GENERATOR = generator


def get_generator():
    return _GENERATOR


def generate_supportive_reply(
    user_message: str,
    current_emotion: str,
    current_emotion_confidence: float,
    forecast_emotion: str,
    forecast_emotion_projected: str,
    forecast_confidence: float,
    deviation_level: str,
    deviation_score: float,
    strategy: str,
    safety: Dict[str, Any],
    dialogue_history: Optional[Sequence[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """Run one turn through the graph and return its final state."""
    initial: ReplyState = {
        "user_message": user_message,
        "dialogue_history": list(dialogue_history or []),
        "current_emotion": current_emotion,
        "current_emotion_confidence": current_emotion_confidence,
        "forecast_emotion": forecast_emotion,
        "forecast_emotion_projected": forecast_emotion_projected,
        "forecast_confidence": forecast_confidence,
        "deviation_level": deviation_level,
        "deviation_score": deviation_score,
        "strategy": strategy,
        "safety": safety,
        "node_trace": [],
    }
    # recursion_limit caps the retry cycle: each attempt costs ~4 super-steps.
    final_state = get_reply_graph().invoke(
        initial, config={"recursion_limit": 8 + 4 * LLM_MAX_RETRIES}
    )
    return dict(final_state)
