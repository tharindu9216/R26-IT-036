"""C4 Emotion Forecasting & Supportive Dialogue — Streamlit demo.

Runs the full C4 turn against the two trained models and shows, for every
stage, both the output and the explanation behind it.

    streamlit run streamlit_app.py
"""

import hashlib
import html
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from c4_pipeline import run_c4_pipeline
from c4_pipeline.emotion_classifier import EmotionClassifier
from c4_pipeline.emotion_forecaster import EmotionForecaster
from c4_pipeline.qwen_generator import QwenReplyGenerator
from c4_pipeline.reply_graph import set_generator
from c4_pipeline.strategy_mapping import mapping_table
from c4_pipeline.voice import (
    SpeechSynthesizer,
    SpeechTranscriber,
    Transcript,
)
from config import (
    AVAILABLE_FORECASTERS,
    CLASSIFIER_METRICS,
    CURRENT_EMOTION_MODEL_PATH,
    DEFAULT_FORECASTER,
    EMOTION_LABELS,
    FORECAST_LABELS,
    FORECASTER_METRICS,
    LLM_ADAPTER_PATH,
    LLM_BASE_MODEL,
    LLM_VOICE_MAX_NEW_TOKENS,
    LOW_VRAM,
    NEXT_EMOTION_MODEL_PATH,
    SMALL_MODEL_DEVICE,
    STT_MODEL,
    VOICE_AUTOPLAY,
    VOICE_ENABLED,
)

st.set_page_config(page_title="C4 Emotion Forecasting Demo", page_icon="💬", layout="wide")

# Push-to-talk needs st.audio_input (Streamlit >= 1.41) and autoplay needs
# st.audio(autoplay=) (>= 1.37). requirements.txt asks for >= 1.42; an older
# install keeps the text demo rather than dying on a missing attribute.
HAS_AUDIO_INPUT = hasattr(st, "audio_input")

# Palette: single-hue sequential for magnitude, blue<->red diverging for signed
# attributions, gray for the neutral midpoint.
BLUE, BLUE_DARK = "#2a78d6", "#184f95"
ORANGE, ORANGE_DARK = "#eb6834", "#b8420f"
RED = "#e34948"
MUTED, GRID = "#898781", "#e1e0d9"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#0d366b"]

# Graph routes, named for the metric tile rather than shown as internal slugs.
_ROUTE_LABELS = {
    "adapter": "ESConv adapter",
    "base": "base model",
    "crisis": "safe fallback",
    "template": "template",
}
_SOURCE_LABELS = {
    "qwen_adapter": "Qwen3-4B + ESConv adapter",
    "qwen_base": "Qwen3-4B base",
    "safe_fallback": "fixed safe-fallback text",
    "template": "strategy template",
}


# ----------------------------------------------------------------- resources
def _init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("emotion_history", [])
    st.session_state.setdefault("pipeline_traces", [])
    # Index of the last assistant message that has already been spoken. Streamlit
    # re-runs the whole script on every interaction, so without this the newest
    # reply would autoplay again each time a sidebar toggle moved.
    st.session_state.setdefault("spoken_upto", -1)
    # Bumped after each processed recording. It is the st.audio_input widget key,
    # and changing the key is what clears the widget -- otherwise it keeps
    # returning the same clip on every rerun and the turn fires repeatedly.
    st.session_state.setdefault("voice_turn", 0)
    st.session_state.setdefault("last_audio_digest", "")


@st.cache_resource(show_spinner="Loading current-emotion classifier…")
def _load_classifier() -> EmotionClassifier:
    # SMALL_MODEL_DEVICE is "cpu" under C4_LOW_VRAM, which hands 0.47 GB of
    # measured VRAM back to the language model on a 4 GB card.
    return EmotionClassifier(CURRENT_EMOTION_MODEL_PATH, device=SMALL_MODEL_DEVICE)


@st.cache_resource(show_spinner="Loading next-emotion forecaster…")
def _load_forecaster(architecture: str, force_fallback: bool) -> EmotionForecaster:
    return EmotionForecaster(
        NEXT_EMOTION_MODEL_PATH,
        labels=list(FORECAST_LABELS),
        force_fallback=force_fallback,
        architecture=architecture,
        device=SMALL_MODEL_DEVICE,
    )


@st.cache_resource(show_spinner="Loading Qwen3-4B reply generator (first run downloads ~8 GB)…")
def _load_generator() -> QwenReplyGenerator:
    """Built once per process and shared by every session.

    `cache_resource` matters more here than for the other two models: a 4B model
    reloaded per rerun would exhaust 12 GB within a few messages.
    """
    generator = QwenReplyGenerator()
    generator.load()
    return generator


@st.cache_resource(show_spinner="Loading speech recognition (first run downloads ~150 MB)…")
def _load_transcriber() -> SpeechTranscriber:
    """CPU-only, so the 4 GB VRAM budget measured in config.py is untouched."""
    return SpeechTranscriber().load()


@st.cache_resource(show_spinner="Loading speech synthesis…")
def _load_synthesizer() -> SpeechSynthesizer:
    return SpeechSynthesizer().load()


# ----------------------------------------------------------------- chart helpers
def _probability_bars(
    probabilities: Dict[str, float],
    order: Sequence[str],
    highlight: str,
    title: str,
    base_color: str,
    accent_color: str,
) -> go.Figure:
    """Magnitude by category -> horizontal bars, one hue, predicted class accented."""
    values = [probabilities.get(label, 0.0) for label in order]
    colors = [accent_color if label == highlight else base_color for label in order]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=list(order),
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{value:.1%}" for value in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        height=60 + 34 * len(order),
        margin=dict(l=10, r=60, t=44, b=10),
        xaxis=dict(range=[0, 1.08], showgrid=True, gridcolor=GRID, tickformat=".0%",
                   zeroline=False),
        yaxis=dict(autorange="reversed", showgrid=False),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED),
    )
    return figure


def _attribution_bars(attributions: Sequence[Tuple[str, float]], title: str) -> go.Figure:
    """Signed contributions -> diverging blue(against) <-> red(toward)."""
    words = [word for word, _ in attributions][::-1]
    scores = [score for _, score in attributions][::-1]
    figure = go.Figure(
        go.Bar(
            x=scores,
            y=words,
            orientation="h",
            marker=dict(color=[RED if s >= 0 else BLUE for s in scores], line=dict(width=0)),
            hovertemplate="%{y}: %{x:+.2f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        height=60 + 26 * max(len(words), 1),
        margin=dict(l=10, r=20, t=44, b=10),
        xaxis=dict(range=[-1.05, 1.05], showgrid=True, gridcolor=GRID, zeroline=True,
                   zerolinecolor="#c3c2b7", title="← pushes away · pushes toward →"),
        yaxis=dict(showgrid=False),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED),
    )
    return figure


def _counterfactual_heatmap(sweep: Dict[str, Dict[str, object]]) -> go.Figure:
    """P(next emotion | assumed current emotion) -> sequential single-hue heatmap."""
    rows = list(sweep.keys())
    columns = list(FORECAST_LABELS)
    matrix = [[sweep[row]["probabilities"].get(col, 0.0) for col in columns] for row in rows]
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=columns,
            y=rows,
            colorscale=[[i / (len(BLUE_RAMP) - 1), c] for i, c in enumerate(BLUE_RAMP)],
            zmin=0,
            zmax=1,
            xgap=2,
            ygap=2,
            colorbar=dict(title="P", tickformat=".0%"),
            hovertemplate="current=%{y} → next=%{x}: %{z:.1%}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Forecast distribution under each assumed current emotion",
        height=80 + 36 * len(rows),
        margin=dict(l=10, r=10, t=44, b=10),
        xaxis=dict(title="forecast next emotion", side="bottom"),
        yaxis=dict(title="assumed current emotion", autorange="reversed"),
        font=dict(color=MUTED),
    )
    return figure


def _history_chart(history: pd.DataFrame) -> go.Figure:
    """One series, direct-labelled with the emotion at each turn."""
    figure = go.Figure(
        go.Scatter(
            x=history["turn"],
            y=history["confidence"],
            mode="lines+markers+text",
            line=dict(color=BLUE, width=2),
            marker=dict(size=9, color=BLUE),
            text=history["emotion"],
            textposition="top center",
            textfont=dict(color="#52514e"),
            hovertemplate="turn %{x}: %{text} (%{y:.1%})<extra></extra>",
        )
    )
    figure.update_layout(
        title="Current-emotion confidence by turn",
        height=320,
        margin=dict(l=10, r=20, t=44, b=10),
        xaxis=dict(title="turn", dtick=1, showgrid=False),
        yaxis=dict(title="confidence", range=[0, 1.15], tickformat=".0%",
                   showgrid=True, gridcolor=GRID),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED),
    )
    return figure


def _highlighted_text(attributions: Sequence[Tuple[str, float]]) -> str:
    """Inline word highlighting; opacity carries magnitude, hue carries sign."""
    pieces: List[str] = []
    for word, score in attributions:
        alpha = min(abs(float(score)), 1.0) * 0.65
        rgb = "227,73,72" if score >= 0 else "42,120,214"
        pieces.append(
            f'<span title="{score:+.2f}" style="background:rgba({rgb},{alpha:.2f});'
            f'padding:2px 4px;margin:1px;border-radius:4px;">{html.escape(word)}</span>'
        )
    return (
        '<div style="line-height:2.1;font-size:1.02rem;">' + " ".join(pieces) + "</div>"
        '<div style="color:#898781;font-size:0.82rem;margin-top:8px;">'
        '<span style="background:rgba(227,73,72,0.5);padding:1px 6px;border-radius:4px;">red</span>'
        " pushes toward the prediction · "
        '<span style="background:rgba(42,120,214,0.5);padding:1px 6px;border-radius:4px;">blue</span>'
        " pushes away</div>"
    )


# ----------------------------------------------------------------- sidebar
_init_state()

with st.sidebar:
    st.header("C4 Settings")

    architecture = st.selectbox(
        "Forecaster checkpoint",
        AVAILABLE_FORECASTERS,
        index=AVAILABLE_FORECASTERS.index(DEFAULT_FORECASTER),
        help="All three were trained by forcasting/train_deep.py on the same splits.",
    )
    force_fallback_forecaster = st.toggle("Force rule-based forecaster", value=False)

    use_llm = st.toggle(
        "Generate replies with Qwen3-4B",
        value=True,
        help="Off falls back to the fixed strategy templates, which needs no GPU.",
    )
    show_xai = st.toggle("Compute explanations (XAI)", value=True,
                         help="Integrated Gradients costs ~32 forward passes per model.")
    show_trace = st.toggle("Show JSON trace", value=False)

    voice_mode = st.toggle(
        "Voice mode (push to talk)",
        value=VOICE_ENABLED and HAS_AUDIO_INPUT,
        disabled=not HAS_AUDIO_INPUT,
        help="Speak a turn and hear the reply. Both models run on the CPU, so "
             "this costs no VRAM.",
    )
    if not HAS_AUDIO_INPUT:
        st.caption(
            f"Voice needs Streamlit >= 1.41 for st.audio_input; this is "
            f"{st.__version__}. Run `pip install -U -r requirements.txt`."
        )

    # XAI is what makes a spoken exchange stop feeling like a conversation:
    # IG_STEPS is 64 forward+backward passes across two models, on top of a
    # reply that already takes seconds to generate. Voice mode suppresses it
    # rather than disabling the toggle, so switching voice off restores whatever
    # the user had chosen.
    explain = show_xai and not voice_mode
    if show_xai and voice_mode:
        st.caption("Explanations are paused while voice mode is on — they cost "
                   "~64 passes per model and would stall the conversation.")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.emotion_history = []
        st.session_state.pipeline_traces = []
        st.rerun()

    classifier = _load_classifier()
    forecaster = _load_forecaster(architecture, force_fallback_forecaster)

    # The graph holds a module-level generator reference; attach it when reply
    # generation is on and clear it when off, so the graph's own fallback path
    # handles the toggle rather than the caller branching around it.
    generator = _load_generator() if use_llm else None
    set_generator(generator)

    # Loaded lazily: a text-only run should not pay for ~150 MB of Whisper.
    transcriber = _load_transcriber() if voice_mode else None
    synthesizer = _load_synthesizer() if voice_mode else None

    st.subheader("Model status")
    if classifier.fallback:
        st.warning("Fallback classifier — trained model not loaded.")
        st.caption(str(classifier.load_error))
    else:
        st.success(f"Classifier: {CLASSIFIER_METRICS['model']} on {classifier.device}")
        st.caption(
            f"test acc {CLASSIFIER_METRICS['test_accuracy']:.4f} · "
            f"macro-F1 {CLASSIFIER_METRICS['test_macro_f1']:.4f} · "
            f"{len(classifier.labels)} classes"
        )

    if forecaster.fallback:
        st.warning("Rule-based forecaster (persistence heuristic).")
        st.caption(str(forecaster.load_error))
    else:
        metrics = FORECASTER_METRICS.get(forecaster.architecture, {})
        st.success(f"Forecaster: {forecaster.architecture} on {forecaster.device}")
        st.caption(
            f"test acc {metrics.get('test_accuracy', float('nan')):.4f} · "
            f"macro-F1 {metrics.get('test_macro_f1', float('nan')):.4f} · "
            f"{len(forecaster.labels)} classes"
        )
        persistence = FORECASTER_METRICS["baseline_persistence"]["test_macro_f1"]
        if metrics.get("test_macro_f1", 0) < persistence:
            st.info(
                f"Persistence baseline still leads at macro-F1 {persistence:.4f}. "
                "Reported as-measured."
            )

    if not use_llm:
        st.info("Reply generation is off; strategy templates are in use.")
    elif generator is not None and generator.available:
        precision = "4-bit NF4" if generator.quantized else f"unquantized {generator.dtype}"
        # LLM_BASE_MODEL may be a hub id or a vendored local path, so normalise
        # separators before taking the last segment.
        model_label = LLM_BASE_MODEL.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        if generator.adapter_loaded:
            st.success(f"Replies: {model_label} + ESConv adapter")
            st.caption(
                f"{precision} on {generator.device} · adapter enabled for distress "
                "turns, disabled for positive and neutral ones."
            )
        else:
            st.warning("Replies: base Qwen3-4B only — no trained adapter found.")
            st.caption(str(generator.adapter_error))
        if generator.device.startswith("cpu"):
            st.caption("Running on the CPU — expect tens of seconds per reply.")
    elif generator is not None:
        st.error("Qwen3-4B did not load; the graph is using templates.")
        st.caption(str(generator.load_error))

    if voice_mode:
        if transcriber is not None and transcriber.available:
            st.success(f"Speech in: faster-whisper {STT_MODEL} on cpu")
            st.caption("int8 · transcripts are decoded in memory and never "
                       "written to disk.")
        else:
            st.error("Speech recognition did not load; voice input is off.")
            st.caption(str(transcriber.load_error if transcriber else "not loaded"))

        if synthesizer is not None and synthesizer.available:
            if synthesizer.degraded:
                # Reached when Kokoro's phonemiser will not load, which is the
                # usual way neural TTS fails on a fresh Windows machine.
                st.warning(f"Speech out: {synthesizer.describe()}")
                st.caption(f"Kokoro unavailable — {synthesizer.load_error}")
            else:
                st.success(f"Speech out: {synthesizer.describe()}")
        else:
            st.error("Speech synthesis did not load; replies stay text-only.")
            st.caption(str(
                (synthesizer.fallback_error or synthesizer.load_error)
                if synthesizer else "not loaded"
            ))
        st.caption(
            f"Reply budget trimmed to {LLM_VOICE_MAX_NEW_TOKENS} tokens while "
            "speaking — generation is most of a spoken turn's latency."
        )

    if LOW_VRAM:
        st.caption(
            f"Low-VRAM mode: classifier and forecaster on {SMALL_MODEL_DEVICE}, "
            "shorter context and reply budget. Set for ~4 GB cards."
        )

    st.caption(
        f"Label spaces: classifier {len(EMOTION_LABELS)} · forecaster "
        f"{len(FORECAST_LABELS)}. They are bridged by c4_pipeline/label_mapping.py."
    )

# ----------------------------------------------------------------- header
st.title("C4 Emotion Forecasting & Supportive Dialogue Demo")
st.caption(
    "Academic research prototype. Not a medical or therapy system and not a "
    "replacement for professional support."
)

# ----------------------------------------------------------------- conversation
def _undo_last_turn() -> None:
    """Drop the most recent exchange and everything derived from it.

    Voice turns are submitted without a confirmation step, so this is the repair
    path for a transcription that came out wrong: the turn is removed from the
    dialogue history before it can bias the forecaster's context or the emotion
    chart.
    """
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        st.session_state.messages.pop()
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        st.session_state.messages.pop()
    if st.session_state.pipeline_traces:
        st.session_state.pipeline_traces.pop()
    if st.session_state.emotion_history:
        st.session_state.emotion_history.pop()
    st.session_state.spoken_upto = len(st.session_state.messages) - 1


def _run_turn(text: str, transcript: Optional[Transcript] = None) -> None:
    """One C4 turn, from either input modality.

    Text and speech converge here: by this point a spoken turn is just a string,
    which is what keeps `run_c4_pipeline` modality-agnostic. `transcript` only
    carries the metadata worth recording about how that string was obtained.
    """
    voice_input = None
    caption = None
    if transcript is not None:
        voice_input = {
            "stt_backend": transcript.backend,
            "stt_model": STT_MODEL,
            "avg_logprob": round(transcript.avg_logprob, 3),
            "duration_seconds": round(transcript.duration, 2),
            "low_confidence": transcript.low_confidence,
        }
        if transcript.low_confidence:
            caption = (
                "⚠️ Low-confidence transcription — if this is not what you said, "
                "undo the turn and try again."
            )

    with st.spinner("Running the C4 pipeline…"):
        # Run before appending so the forecaster's dialogue context contains only
        # the turns that genuinely preceded this message.
        trace = run_c4_pipeline(
            user_message=text,
            conversation_state=st.session_state,
            classifier=classifier,
            forecaster=forecaster,
            force_fallback_forecaster=force_fallback_forecaster,
            explain=explain,
            use_llm=use_llm,
            max_new_tokens=LLM_VOICE_MAX_NEW_TOKENS if voice_mode else None,
            voice_input=voice_input,
        )

    reply = trace["supportive_response"]
    spoken = b""
    if voice_mode and synthesizer is not None and synthesizer.available:
        with st.spinner("Speaking…"):
            spoken = synthesizer.synthesize(reply)

    user_message = {"role": "user", "content": text}
    if caption:
        user_message["caption"] = caption
    st.session_state.messages.append(user_message)

    assistant_message = {"role": "assistant", "content": reply}
    if spoken:
        assistant_message["audio"] = spoken
    st.session_state.messages.append(assistant_message)


for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("caption"):
            st.caption(message["caption"])
        if message.get("audio"):
            # Autoplay the newest reply only, then mark it spoken. Every rerun
            # re-creates this element, so an unguarded autoplay would replay the
            # last reply on every widget interaction.
            first_play = index > st.session_state.spoken_upto
            st.audio(
                message["audio"],
                format="audio/wav",
                autoplay=first_play and VOICE_AUTOPLAY,
            )
            if first_play:
                st.session_state.spoken_upto = index

if voice_mode and HAS_AUDIO_INPUT:
    mic_column, undo_column = st.columns([4, 1], vertical_alignment="bottom")
    with mic_column:
        # The widget key carries the turn counter: bumping it after a processed
        # recording is what resets the widget, which is the only reliable way to
        # stop st.audio_input handing back the same clip on the next rerun.
        recording = st.audio_input(
            "Push to talk — record, stop, and the turn is sent",
            key=f"mic_{st.session_state.voice_turn}",
        )
    with undo_column:
        if st.button(
            "Undo turn",
            use_container_width=True,
            disabled=not st.session_state.messages,
            help="Remove the last exchange — use it when a transcription came "
                 "out wrong.",
        ):
            _undo_last_turn()
            st.rerun()

    if recording is not None:
        audio_bytes = recording.getvalue()
        # Second guard, belt to the widget key's braces: identical bytes are the
        # same recording, however the widget was re-rendered.
        digest = hashlib.sha1(audio_bytes).hexdigest()
        if digest != st.session_state.last_audio_digest:
            st.session_state.last_audio_digest = digest
            if transcriber is None or not transcriber.available:
                st.error("Speech recognition is unavailable — type instead.")
            else:
                with st.spinner("Transcribing…"):
                    transcript = transcriber.transcribe(audio_bytes)
                if transcript.error:
                    st.error(f"Could not transcribe that clip: {transcript.error}")
                elif not transcript.ok:
                    st.warning("Nothing was heard in that recording — try again.")
                else:
                    _run_turn(transcript.text, transcript)
                    st.session_state.voice_turn += 1
                    st.rerun()

user_input = st.chat_input("Share how you are feeling…")
if user_input:
    _run_turn(user_input)
    st.rerun()

if not st.session_state.pipeline_traces:
    st.info("Enter a message to start the C4 demo.")
    st.stop()

last_trace = st.session_state.pipeline_traces[-1]
explanations = last_trace.get("explanations", {})

# ----------------------------------------------------------------- outputs
st.subheader("Pipeline outputs")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current emotion", last_trace["current_emotion"],
            f"{last_trace['current_emotion_confidence']:.1%} confidence")
col2.metric("Previous emotion", str(last_trace["previous_emotion"] or "—"))
col3.metric("Deviation",
            f"{last_trace['deviation_level']}",
            f"score {last_trace['deviation_score']:.2f}", delta_color="off")
col4.metric("Forecast next emotion", last_trace["forecasted_next_emotion"],
            f"{last_trace['forecast_confidence']:.1%} confidence")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Selected strategy", last_trace["selected_strategy"])
col6.metric("Forecast source", last_trace["forecast_source"])
col7.metric("Safety", "risk flagged" if last_trace["safety"]["risk_detected"] else "clear")
col8.metric("Reply route", _ROUTE_LABELS.get(last_trace.get("reply_route", "template"),
                                             last_trace.get("reply_route", "—")))
st.caption(last_trace["strategy_reason"])

tab_dist, tab_reply, tab_xai, tab_rules, tab_history = st.tabs(
    ["Distributions", "Reply generation", "Explainability (XAI)", "Strategy rules",
     "Conversation history"]
)

# ----------------------------------------------------------------- distributions
with tab_dist:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            _probability_bars(
                last_trace["current_emotion_probabilities"], EMOTION_LABELS,
                last_trace["current_emotion"],
                "Current emotion — classifier (RoBERTa, 5 classes)", BLUE_RAMP[1], BLUE,
            ),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            _probability_bars(
                last_trace["forecast_probabilities"] or {}, FORECAST_LABELS,
                last_trace["forecasted_next_emotion"],
                f"Next emotion — forecaster ({last_trace['forecast_source']}, 8 classes)",
                "#f6c3ac", ORANGE,
            ),
            use_container_width=True,
        )

    st.markdown("**Forecast projected onto the classifier's 5 labels**")
    st.caption(explanations.get("label_mapping_note", ""))
    projected = last_trace.get("forecast_probabilities_projected") or {}
    if projected:
        st.dataframe(
            pd.DataFrame(
                [{"emotion": k, "probability": v} for k, v in projected.items()]
            ).sort_values("probability", ascending=False),
            hide_index=True,
            use_container_width=True,
            column_config={"probability": st.column_config.ProgressColumn(
                "probability", min_value=0.0, max_value=1.0, format="%.3f")},
        )
    if explanations.get("label_mapping_lossy"):
        st.warning(
            "This projection is lossy: the 8-label forecast distinguishes states "
            "the 5-label classifier cannot represent."
        )

# ----------------------------------------------------------------- reply generation
with tab_reply:
    st.markdown("### The reply is produced by a LangGraph state machine")
    st.caption(
        "Nodes below are the ones this turn actually executed, in order. A crisis "
        "turn stops after two nodes and never reaches the language model; a reply "
        "that fails the output guard loops back to build_prompt and is regenerated."
    )

    route = last_trace.get("reply_route", "template")
    source = last_trace.get("reply_source", "template")
    generation = last_trace.get("reply_generation") or {}
    guard = last_trace.get("reply_guard") or {}

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Route", _ROUTE_LABELS.get(route, route))
    r2.metric("Answered by", _SOURCE_LABELS.get(source, source))
    r3.metric("Attempts", last_trace.get("reply_attempts", 0))
    r4.metric(
        "Latency",
        f"{generation.get('latency_seconds', 0):.2f}s" if generation.get("available") else "—",
    )
    st.info(last_trace.get("reply_route_reason", ""))

    st.markdown("**Path through the graph**")
    node_trace = last_trace.get("reply_node_trace") or []
    if node_trace:
        st.dataframe(
            pd.DataFrame(
                [
                    {"#": index + 1, "node": entry["node"], "what happened": entry["detail"]}
                    for index, entry in enumerate(node_trace)
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.caption("Reply generation was switched off for this turn.")

    if route in ("adapter", "base"):
        st.divider()
        st.markdown("### Strategy translation")
        st.caption(
            "The adapter was fine-tuned with ESConv's own 8-strategy annotation in "
            "the `Response strategy:` slot, so C4's strategy name is translated "
            "before it reaches the prompt. Feeding it C4's vocabulary directly "
            "would put the model on tokens it never saw in that position."
        )
        m1, m2 = st.columns(2)
        m1.metric("C4 strategy", last_trace["selected_strategy"])
        m2.metric("ESConv strategy", last_trace.get("reply_esconv_strategy") or "—")
        st.caption(last_trace.get("reply_strategy_mapping_reason", ""))
        with st.expander("Full strategy mapping table"):
            st.dataframe(
                pd.DataFrame(mapping_table()),
                hide_index=True,
                use_container_width=True,
            )

        st.divider()
        st.markdown("### Prompt sent to the model")
        st.caption(
            "The first block reproduces the fine-tuning prompt verbatim. The "
            "internal-signals block is appended last because it is the one part "
            "the adapter never saw during training."
        )
        st.code(last_trace.get("reply_prompt", ""), language=None)

        st.markdown("### Output guard")
        if guard.get("passed"):
            st.success("The generated reply passed every check.")
        else:
            st.warning("Rejected: " + "; ".join(guard.get("failures", [])))
        g1, g2, g3 = st.columns(3)
        g1.metric("Words", guard.get("word_count", 0))
        g2.metric("Sentences", guard.get("sentence_count", 0))
        g3.metric("Leak matches", len(guard.get("leaks", [])))
        if guard.get("leaks"):
            st.dataframe(
                pd.DataFrame(guard["leaks"]).rename(
                    columns={"check": "failed check", "matched": "matched text"}
                ),
                hide_index=True,
                use_container_width=True,
            )
        if guard.get("cosmetic_fixes"):
            st.caption("Cleaned before checking: " + ", ".join(guard["cosmetic_fixes"]) + ".")

        raw_reply = last_trace.get("reply_raw", "")
        if raw_reply and raw_reply.strip() != last_trace["supportive_response"].strip():
            with st.expander("Raw model output before cleaning"):
                st.code(raw_reply, language=None)

        if generation.get("available"):
            st.caption(
                f"{generation.get('prompt_tokens', 0)} prompt tokens -> "
                f"{generation.get('completion_tokens', 0)} generated, "
                f"temperature {generation.get('temperature')}, "
                f"adapter {'on' if generation.get('adapter_used') else 'off'}."
            )
        elif generation.get("error"):
            st.error(generation["error"])

# ----------------------------------------------------------------- XAI
with tab_xai:
    if not explain:
        if show_xai and voice_mode:
            st.info(
                "Explanations are paused while voice mode is on — Integrated "
                "Gradients costs ~64 passes per model, which would stall a spoken "
                "exchange. Switch voice mode off to compute them."
            )
        else:
            st.info("Explanations are switched off. Enable “Compute explanations (XAI)” in the sidebar.")
    else:
        st.markdown("### 1 · Why this current emotion?")
        classifier_xai = explanations.get("classifier", {})
        if not classifier_xai.get("available"):
            st.info(classifier_xai.get("reason", "No classifier explanation available."))
        else:
            st.caption(
                f"Integrated Gradients ({classifier_xai['steps']} steps) on the "
                f"input embeddings, target **{classifier_xai['target_label']}** "
                f"(p = {classifier_xai['target_probability']:.1%}). Completeness gap "
                f"{classifier_xai['completeness_gap']:.4f} "
                f"({classifier_xai['completeness_gap_relative']:.1%} of the logit "
                "span — lower means the approximation converged)."
            )
            st.markdown(
                _highlighted_text(classifier_xai["token_attributions"]),
                unsafe_allow_html=True,
            )
            ig_col, occ_col = st.columns(2)
            with ig_col:
                st.plotly_chart(
                    _attribution_bars(classifier_xai["top_tokens"],
                                      "Integrated Gradients — top tokens"),
                    use_container_width=True,
                )
            with occ_col:
                occlusion = classifier_xai.get("occlusion", {})
                if occlusion.get("available"):
                    st.plotly_chart(
                        _attribution_bars(occlusion["top_tokens"],
                                          "Occlusion — probability drop when masked"),
                        use_container_width=True,
                    )
                    st.caption(
                        "A second, gradient-free method. Where the two agree, the "
                        "explanation is trustworthy; where they disagree, treat it "
                        "with caution."
                    )
                else:
                    st.info(occlusion.get("reason", "Occlusion not run."))

        st.divider()
        st.markdown("### 2 · Why this forecast?")
        forecaster_xai = explanations.get("forecaster", {})
        if not forecaster_xai.get("available"):
            st.info(forecaster_xai.get("reason", "No forecaster explanation available."))
        else:
            st.caption(
                f"Model input (dialogue context, {forecaster.context_turns} previous turns):"
            )
            st.code(last_trace["forecast_context_text"], language=None)
            st.markdown(
                _highlighted_text(forecaster_xai["token_attributions"]),
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                _attribution_bars(forecaster_xai["top_tokens"],
                                  "Integrated Gradients — top context tokens"),
                use_container_width=True,
            )

            aux = forecaster_xai.get("aux_contribution", {})
            if aux:
                st.markdown("**Contribution of the current-emotion feature**")
                a1, a2, a3 = st.columns(3)
                a1.metric("With the feature", f"{aux['probability_with_feature']:.1%}")
                a2.metric("Feature ablated", f"{aux['probability_without_feature']:.1%}",
                          f"{aux['delta']:+.1%}")
                a3.metric("Prediction flips?",
                          "yes" if aux["prediction_flips_without_feature"] else "no")
                st.caption(
                    f"Removing the current emotion (`{aux['current_emotion_feature']}`) "
                    f"changes the forecast to `{aux['prediction_without_feature']}`."
                    if aux["prediction_flips_without_feature"]
                    else "The forecast survives removing the current-emotion feature."
                )

            vocabulary = explanations.get("forecaster_vocabulary", {})
            if vocabulary.get("available") and vocabulary["n_unknown"]:
                st.warning(
                    f"{vocabulary['n_unknown']}/{vocabulary['n_tokens']} tokens "
                    f"({vocabulary['unknown_rate']:.0%}) are outside the forecaster's "
                    f"{vocabulary['vocab_size']}-word training vocabulary and were "
                    f"read as `<unk>`: {', '.join(vocabulary['unknown_tokens'])}. "
                    "Their attributions describe the unknown-token embedding, not the word."
                )

        st.divider()
        st.markdown("### 3 · Counterfactual: what if the current emotion were different?")
        counterfactual = explanations.get("forecaster_counterfactual", {})
        if not counterfactual.get("available"):
            st.info(counterfactual.get("reason", "No counterfactual available."))
        else:
            st.caption(counterfactual["note"])
            st.plotly_chart(
                _counterfactual_heatmap(
                    forecaster.counterfactual_by_current_emotion(
                        last_trace["forecast_context_text"]
                    )
                ),
                use_container_width=True,
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "assumed current": emotion,
                            "forecast next": result["label"],
                            "confidence": result["confidence"],
                        }
                        for emotion, result in counterfactual["sweep"].items()
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )

# ----------------------------------------------------------------- rules
with tab_rules:
    st.markdown("### Strategy selection is rule-based and fully inspectable")
    rules = explanations.get("strategy_rules", [])
    if rules:
        st.dataframe(
            pd.DataFrame(rules).rename(
                columns={"reached": "evaluated", "fired": "condition true"}
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Rules are checked top to bottom and the first true condition wins; "
            "rows with evaluated = False were never reached."
        )
    st.markdown("**Deviation tracking**")
    st.write(
        f"`{last_trace['previous_emotion'] or 'none'}` → "
        f"`{last_trace['current_emotion']}` gives deviation score "
        f"**{last_trace['deviation_score']:.2f}** (level "
        f"**{last_trace['deviation_level']}**)."
    )
    st.markdown("**Safety layer**")
    safety = last_trace["safety"]
    if safety["risk_detected"]:
        st.error(
            f"Risk type `{safety['risk_type']}` triggered by: "
            + ", ".join(f"“{phrase}”" for phrase in safety.get("matched_phrases", []))
        )
    else:
        st.success("No crisis language matched; the safe fallback was not triggered.")
    st.json(safety)

# ----------------------------------------------------------------- history
with tab_history:
    history_df = pd.DataFrame(st.session_state.emotion_history)
    st.dataframe(history_df, hide_index=True, use_container_width=True)
    if not history_df.empty:
        st.plotly_chart(_history_chart(history_df), use_container_width=True)

if show_trace:
    st.subheader("Conversation trace")
    st.json(last_trace)
