"""Streamlit UI for current emotion, next intensity and an optional Qwen reply."""

from __future__ import annotations

import streamlit as st

import services
import settings
from emotion_chain.core import CURRENT_EMOTIONS
from emotion_chain.deviation_tracker import compute_deviation
from emotion_chain.pipeline import EmotionChain
from emotion_chain.reply_generator import QwenReplyGenerator, ReplySignals


@st.cache_resource(show_spinner="Loading the emotion chain...")
def load_emotion_chain() -> EmotionChain:
    return services.build_emotion_chain()


@st.cache_resource(show_spinner=False)
def load_reply_generator() -> QwenReplyGenerator:
    # Qwen itself remains lazy and is loaded only when reply generation is used.
    return services.build_reply_generator()


def percent(value: float) -> str:
    return f"{value:.1%}"


st.set_page_config(page_title="Supportive emotion pipeline", page_icon="🧠", layout="wide")
st.title("Current emotion → next intensity → supportive reply")
st.caption("All model artifacts are loaded from model/; no model download is required.")

current_text = st.text_area(
    "Current user message",
    placeholder="I feel overwhelmed and I do not know what to do next.",
)
previous_emotion = st.selectbox(
    "Previous turn's emotion (for deviation tracking)",
    ("(none)",) + CURRENT_EMOTIONS,
    help=(
        "This page scores one message at a time, so the previous turn's "
        "emotion is supplied by hand. The chat apps carry it automatically."
    ),
)
generate_reply = st.checkbox(
    "Generate a Qwen reply",
    value=False,
    help="This loads the local 4B model on first use and needs substantial RAM/VRAM.",
)

if st.button("Run pipeline", type="primary", use_container_width=True):
    try:
        chain = load_emotion_chain()
        with st.spinner("Predicting current emotion and next intensity..."):
            result = chain.predict(current_text)
    except Exception as exc:
        st.error(str(exc))
    else:
        current_column, next_column, route_column = st.columns(3)
        current_column.metric(
            "Current emotion",
            f"{result.current.label} ({percent(result.current.confidence)})",
        )
        next_column.metric(
            "Next intensity",
            f"{result.forecast.label} ({percent(result.forecast.confidence)})",
            help=(
                "The forecaster predicts how intense the next turn will be, "
                f"not which emotion it will be. Named as a state: "
                f"{result.next.label}."
            ),
        )
        route_column.metric("Reply route", result.reply_route)

        deviation = compute_deviation(
            None if previous_emotion == "(none)" else previous_emotion,
            result.current.label,
        )
        st.metric(
            "Deviation from previous turn",
            f"{deviation.level} ({deviation.score:.2f})",
        )

        if result.intensifies:
            st.info(
                f"Forecast: this is heading towards a more intense "
                f"{result.current.label} turn ({result.next.label})."
            )
        else:
            st.info(
                f"Forecast: the next turn stays at or below this intensity "
                f"({result.next.label})."
            )

        left, right = st.columns(2)
        with left:
            st.subheader("Current-emotion probabilities")
            st.bar_chart(
                {
                    "emotion": list(result.current.probabilities),
                    "probability": list(result.current.probabilities.values()),
                },
                x="emotion",
                y="probability",
                horizontal=True,
            )
        with right:
            st.subheader("Reachable next-state probabilities")
            reachable = {
                label: value
                for label, value in result.next.probabilities.items()
                if value > 0
            }
            st.bar_chart(
                {
                    "state": list(reachable),
                    "probability": list(reachable.values()),
                },
                x="state",
                y="probability",
                horizontal=True,
            )
            st.caption(
                f"P(high) = {percent(result.forecast.probability_high)}, "
                f"threshold {result.forecast.threshold:.3f}"
            )

        if generate_reply:
            generator = load_reply_generator()
            signals = ReplySignals(
                current_emotion=result.current.label,
                next_emotion=result.next.label,
                next_emotion_confidence=result.next.confidence,
                next_negative_min_confidence=settings.NEXT_NEGATIVE_MIN_CONFIDENCE,
                previous_emotion=deviation.previous_emotion,
                deviation_level=deviation.level,
                deviation_score=deviation.score,
            )
            with st.spinner("Generating reply with local Qwen..."):
                reply = generator.generate(current_text, signals)
            st.subheader("Generated reply")
            st.write(reply.text)
            st.caption(f"Route: {reply.route} • Source: {reply.source}")
            if reply.support_contacts:
                st.warning(
                    "Escalated as extremely negative — the reply names "
                    + ", ".join(
                        f"{contact.name} ({contact.phone})"
                        for contact in reply.support_contacts
                    )
                )

        with st.expander("Next-intensity forecast, per branch"):
            st.json(result.forecast.as_dict())
