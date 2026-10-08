"""LangGraph orchestration for text and voice supportive-agent turns."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

from .core import select_reply_route
from .deviation_tracker import compute_deviation
from .fusion import fuse_stress_and_cbt
from .reply_generator import (
    QwenReplyGenerator,
    ReplySignals,
    crisis_reply,
)
from .support_contacts import DEFAULT_SUPPORT_CONTACTS, SupportContact, as_dicts


class StressPredictorProtocol(Protocol):
    def predict(self, text: str) -> Any: ...


class CBTPredictorProtocol(Protocol):
    def predict(self, text: str) -> Any: ...


class EmotionChainProtocol(Protocol):
    def predict(self, current_text: str) -> Any: ...


class SupportiveAgentState(TypedDict, total=False):
    user_text: str
    history: list[tuple[str, str]]
    sensor_stress: bool
    sensor_stress_probability: float | None
    sensor_stress_available: bool
    safety_status: str
    text_stress: bool
    text_stress_probability: float
    text_stress_confidence: float
    has_cbt_distortion: bool
    cbt_probability: float
    cbt_confidence: float
    text_signal_flagged: bool
    text_signal_probability: float
    text_signal_method: str
    voice_stress: bool
    voice_stress_score: float
    questionnaire_stress_available: bool
    questionnaire_stress: bool
    questionnaire_stress_probability: float | None
    multimodal_stress_available: bool
    multimodal_stress: bool
    multimodal_stress_probability: float | None
    multimodal_stress_weights: dict[str, float]
    multimodal_stress_inputs: dict[str, float]
    voice_appraisal_state: str
    voice_appraisal_uncertain: bool
    current_emotion: str
    current_emotion_confidence: float
    current_emotion_probabilities: dict[str, float]
    # The forecaster's own low/high output, and the state label it names once
    # the current emotion supplies the family (see core.intensity_state).
    next_intensity: str
    next_intensity_probability: float
    next_intensity_branches: dict[str, float]
    next_emotion: str
    next_emotion_confidence: float
    next_emotion_probabilities: dict[str, float]
    # Carried in by the caller from the previous turn of the same conversation.
    previous_emotion: str | None
    deviation_score: float
    deviation_level: str
    reply_route: str
    reply: str
    reply_source: str
    support_contacts: list[dict[str, str]]


@dataclass(frozen=True)
class AgentComponents:
    emotion_chain: EmotionChainProtocol
    reply_generator: QwenReplyGenerator
    stress_predictor: StressPredictorProtocol | None = None
    cbt_predictor: CBTPredictorProtocol | None = None
    # User-selected decision-level fusion of the C3 Stress + CBT headers.
    fusion_stress_weight: float = 0.5
    fusion_cbt_weight: float = 0.5
    fusion_threshold: float = 0.5
    fusion_method: str = "weighted_average"
    fusion_stress_threshold: float = 0.5
    fusion_cbt_threshold: float = 0.37
    next_negative_min_confidence: float = 0.5
    # Offered on crisis turns and on turns the escalation rule reads as
    # extremely negative; see support_contacts.py.
    support_contacts: tuple[SupportContact, ...] = DEFAULT_SUPPORT_CONTACTS


# This is a conservative routing aid, not a validated suicide-risk classifier.
# The UI and documentation explicitly retain that limitation.
CRISIS_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"\b(?:kill|hurt) myself\b",
        r"\bend my life\b",
        r"\bwant to die\b",
        r"\bdo not want to live\b",
        r"\bsuicid(?:e|al)\b",
    )
)


def detect_safety_status(text: str) -> str:
    normalized = text.strip()
    if not normalized:
        raise ValueError("user_text cannot be empty")
    return "crisis" if any(item.search(normalized) for item in CRISIS_PATTERNS) else "normal"


class SupportiveAgentWorkflow:
    """Dependency-injected nodes that are compiled into a deterministic graph."""

    def __init__(self, components: AgentComponents) -> None:
        self.components = components

    def safety_node(self, state: SupportiveAgentState) -> dict[str, Any]:
        return {"safety_status": detect_safety_status(state["user_text"])}

    @staticmethod
    def after_safety(state: SupportiveAgentState) -> str:
        return "crisis" if state.get("safety_status") == "crisis" else "continue"

    def crisis_node(self, state: SupportiveAgentState) -> dict[str, Any]:
        contacts = self.components.support_contacts
        return {
            "reply_route": "crisis",
            "reply": crisis_reply(contacts),
            "reply_source": "safety_template",
            "support_contacts": as_dicts(contacts),
        }

    def c3_node(self, state: SupportiveAgentState) -> dict[str, Any]:
        if self.components.stress_predictor is None or self.components.cbt_predictor is None:
            raise RuntimeError("C3 predictors are required for the text workflow")
        text = state["user_text"]
        stress = self.components.stress_predictor.predict(text)
        cbt = self.components.cbt_predictor.predict(text)
        return {
            "text_stress": bool(stress.is_stressed),
            "text_stress_probability": float(
                stress.probabilities.get("stressed", 0.0)
            ),
            "text_stress_confidence": float(stress.confidence),
            "has_cbt_distortion": bool(cbt.has_distortion),
            "cbt_probability": float(cbt.probabilities.get("distortion", 0.0)),
            "cbt_confidence": float(cbt.confidence),
        }

    def fusion_node(self, state: SupportiveAgentState) -> dict[str, Any]:
        """Late (decision-level) fusion of the C3 Stress + CBT probabilities.

        Routing and reply generation see this one fused signal, not the two
        headers separately. C1's sensor stress is not part of this fusion.
        """

        fused = fuse_stress_and_cbt(
            state.get("text_stress_probability", 0.0),
            state.get("cbt_probability", 0.0),
            stress_weight=self.components.fusion_stress_weight,
            cbt_weight=self.components.fusion_cbt_weight,
            threshold=self.components.fusion_threshold,
            method=self.components.fusion_method,
            stress_flagged=bool(state.get("text_stress", False)),
            cbt_flagged=bool(state.get("has_cbt_distortion", False)),
            stress_threshold=self.components.fusion_stress_threshold,
            cbt_threshold=self.components.fusion_cbt_threshold,
        )
        return {
            "text_signal_flagged": fused.is_flagged,
            "text_signal_probability": fused.probability,
            "text_signal_method": fused.method,
        }

    def emotion_node(self, state: SupportiveAgentState) -> dict[str, Any]:
        result = self.components.emotion_chain.predict(state["user_text"])
        return {
            "current_emotion": result.current.label,
            "current_emotion_confidence": float(result.current.confidence),
            "current_emotion_probabilities": dict(result.current.probabilities),
            "next_intensity": result.forecast.label,
            "next_intensity_probability": float(result.forecast.probability_high),
            "next_intensity_branches": {
                "baseline": float(result.forecast.baseline_probability),
                "fusion": float(result.forecast.fusion_probability),
            },
            "next_emotion": result.next.label,
            "next_emotion_confidence": float(result.next.confidence),
            "next_emotion_probabilities": dict(result.next.probabilities),
        }

    @staticmethod
    def deviation_node(state: SupportiveAgentState) -> dict[str, Any]:
        """Score this turn's emotion against the previous turn's.

        ``previous_emotion`` is supplied by the caller (the API, Streamlit or
        the CLI keep it per conversation) because the graph itself is
        stateless between invocations.
        """

        deviation = compute_deviation(
            state.get("previous_emotion"), state["current_emotion"]
        )
        return {
            "previous_emotion": deviation.previous_emotion,
            "deviation_score": deviation.score,
            "deviation_level": deviation.level,
        }

    def route_node(self, state: SupportiveAgentState) -> dict[str, Any]:
        has_multimodal = bool(state.get("multimodal_stress_available", False))
        route = select_reply_route(
            state["current_emotion"],
            state["next_emotion"],
            sensor_stress=(
                bool(state.get("sensor_stress", False)) if not has_multimodal else False
            ),
            text_signal_flagged=bool(state.get("text_signal_flagged", False)),
            voice_stress=(
                bool(state.get("voice_stress", False)) if not has_multimodal else False
            ),
            multimodal_stress=(
                bool(state.get("multimodal_stress", False)) if has_multimodal else False
            ),
            next_emotion_confidence=state.get("next_emotion_confidence"),
            next_negative_min_confidence=self.components.next_negative_min_confidence,
        )
        return {"reply_route": route}

    def reply_node(self, state: SupportiveAgentState) -> dict[str, Any]:
        signals = ReplySignals(
            current_emotion=state["current_emotion"],
            next_emotion=state["next_emotion"],
            next_emotion_confidence=state.get("next_emotion_confidence"),
            next_negative_min_confidence=self.components.next_negative_min_confidence,
            previous_emotion=state.get("previous_emotion"),
            deviation_level=str(state.get("deviation_level", "None")),
            deviation_score=float(state.get("deviation_score", 0.0)),
            sensor_stress=bool(state.get("sensor_stress", False)),
            text_signal_flagged=bool(state.get("text_signal_flagged", False)),
            voice_stress=bool(state.get("voice_stress", False)),
            questionnaire_stress=bool(state.get("questionnaire_stress", False)),
            questionnaire_stress_probability=state.get(
                "questionnaire_stress_probability"
            ),
            multimodal_stress_available=bool(
                state.get("multimodal_stress_available", False)
            ),
            multimodal_stress=bool(state.get("multimodal_stress", False)),
            multimodal_stress_probability=state.get(
                "multimodal_stress_probability"
            ),
            sensor_stress_probability=state.get("sensor_stress_probability"),
            text_signal_probability=state.get("text_signal_probability"),
            voice_stress_score=state.get("voice_stress_score"),
            voice_appraisal_state=state.get("voice_appraisal_state"),
            voice_appraisal_uncertain=bool(
                state.get("voice_appraisal_uncertain", False)
            ),
            stress_probability=state.get("text_stress_probability"),
            cbt_probability=state.get("cbt_probability"),
            safety_status=state.get("safety_status", "normal"),
        )
        result = self.components.reply_generator.generate(
            state["user_text"],
            signals,
            history=state.get("history", ()),
        )
        return {
            "reply_route": result.route,
            "reply": result.text,
            "reply_source": result.source,
            "support_contacts": as_dicts(result.support_contacts),
        }

    def compile(self, *, include_c3: bool = True):
        """Build the requested LangGraph; imports stay lazy for light tests."""

        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise RuntimeError(
                "LangGraph is required; install backend/requirements.txt"
            ) from exc

        graph = StateGraph(SupportiveAgentState)
        graph.add_node("safety", self.safety_node)
        graph.add_node("crisis_reply", self.crisis_node)
        if include_c3:
            graph.add_node("c3_analysis", self.c3_node)
            graph.add_node("text_signal_fusion", self.fusion_node)
        graph.add_node("emotion_analysis", self.emotion_node)
        graph.add_node("deviation_tracking", self.deviation_node)
        graph.add_node("reply_routing", self.route_node)
        graph.add_node("reply_generation", self.reply_node)

        graph.add_edge(START, "safety")
        graph.add_conditional_edges(
            "safety",
            self.after_safety,
            {
                "crisis": "crisis_reply",
                "continue": "c3_analysis" if include_c3 else "emotion_analysis",
            },
        )
        graph.add_edge("crisis_reply", END)
        if include_c3:
            graph.add_edge("c3_analysis", "text_signal_fusion")
            graph.add_edge("text_signal_fusion", "emotion_analysis")
        graph.add_edge("emotion_analysis", "deviation_tracking")
        graph.add_edge("deviation_tracking", "reply_routing")
        graph.add_edge("reply_routing", "reply_generation")
        graph.add_edge("reply_generation", END)
        return graph.compile()


def build_supportive_graph(components: AgentComponents):
    return SupportiveAgentWorkflow(components).compile(include_c3=True)


def build_voice_supportive_graph(components: AgentComponents):
    """Compile the C1 + C2 + emotion path without adding any C3 nodes."""

    return SupportiveAgentWorkflow(components).compile(include_c3=False)
