import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Add ai_components to Python path so we can import c4_pipeline and config
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "ai_components" / "c4_emotion_support"))

from c4_pipeline import run_c4_pipeline
from c4_pipeline.emotion_classifier import EmotionClassifier
from c4_pipeline.emotion_forecaster import EmotionForecaster
from config import CURRENT_EMOTION_MODEL_PATH, EMOTION_LABELS, NEXT_EMOTION_MODEL_PATH

st.set_page_config(page_title="C4 Emotion Forecasting Demo", page_icon="💬", layout="wide")


def _init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("emotion_history", [])
    st.session_state.setdefault("pipeline_traces", [])


@st.cache_resource
def _load_classifier() -> EmotionClassifier:
    return EmotionClassifier(CURRENT_EMOTION_MODEL_PATH)


@st.cache_resource
def _load_forecaster(force_fallback: bool) -> EmotionForecaster:
    return EmotionForecaster(
        NEXT_EMOTION_MODEL_PATH,
        labels=list(EMOTION_LABELS),
        force_fallback=force_fallback,
    )


_init_state()

with st.sidebar:
    st.header("C4 Settings")
    show_trace = st.toggle("Show debug trace", value=True)
    force_fallback_forecaster = st.toggle("Force fallback forecaster", value=False)
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.emotion_history = []
        st.session_state.pipeline_traces = []
        st.experimental_rerun()

    classifier = _load_classifier()
    forecaster = _load_forecaster(force_fallback_forecaster)

    st.subheader("Model Status")
    if classifier.fallback:
        st.warning("Using fallback classifier because trained model was not found.")
    else:
        st.success("Current emotion classifier loaded.")

    if forecaster.fallback:
        st.warning("Using fallback forecaster because trained model was not found.")
    else:
        st.success("Next emotion forecaster loaded.")

st.title("C4 Emotion Forecasting & Supportive Dialogue Demo")
st.caption(
    "Academic research prototype. This is not a medical or therapy system and does not replace professional support."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Share how you are feeling...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    trace = run_c4_pipeline(
        user_message=user_input,
        conversation_state=st.session_state,
        classifier=classifier,
        forecaster=forecaster,
        force_fallback_forecaster=force_fallback_forecaster,
    )
    st.session_state.messages.append(
        {"role": "assistant", "content": trace["supportive_response"]}
    )
    st.experimental_rerun()

if st.session_state.pipeline_traces:
    last_trace = st.session_state.pipeline_traces[-1]

    st.subheader("Pipeline Outputs")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Emotion", last_trace["current_emotion"])
    col2.metric("Current Confidence", f"{last_trace['current_emotion_confidence']:.2f}")
    col3.metric("Previous Emotion", str(last_trace["previous_emotion"]))
    col4.metric("Deviation", f"{last_trace['deviation_level']} ({last_trace['deviation_score']:.2f})")

    col5, col6, col7 = st.columns(3)
    col5.metric("Forecasted Next Emotion", last_trace["forecasted_next_emotion"])
    col6.metric("Forecast Confidence", f"{last_trace['forecast_confidence']:.2f}")
    col7.metric("Selected Strategy", last_trace["selected_strategy"])

    st.subheader("Emotion History")
    history_df = pd.DataFrame(st.session_state.emotion_history)
    st.dataframe(history_df, use_container_width=True)

    if not history_df.empty:
        chart = px.line(
            history_df,
            x="turn",
            y="confidence",
            color="emotion",
            markers=True,
            title="Emotion Confidence Over Turns",
        )
        st.plotly_chart(chart, use_container_width=True)

    if show_trace:
        st.subheader("Conversation Trace")
        st.json(last_trace)
else:
    st.info("Enter a message to start the C4 demo.")
