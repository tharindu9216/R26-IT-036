"""Mode-selection entry point for the supportive agent.

Flow (text path): pick "Text" -> C1 wearable sensor calibrates in the
background (~6 min: 5 min personal calibration + 1 min warm-up window, then a
prediction every 5 s) -> a chat box opens -> each message is scored by the C3
stress and CBT ensembles and the emotion chain, then routed
through the LangGraph workflow in ``emotion_chain.agent_graph`` to a Qwen
reply (ESConv LoRA adapter for negative/stressed states, plain base model for
neutral/joy). Voice mode keeps C1 running, uses C2 appraisal plus an offline
Faster-Whisper transcript, skips C3, runs the emotion chain and the voice
LangGraph, then converts the Qwen reply to audio with Kokoro.
"""

from __future__ import annotations

import time

import streamlit as st

import services
import settings
from emotion_chain.agent_graph import (
    AgentComponents,
    build_supportive_graph,
    build_voice_supportive_graph,
)
from emotion_chain.pipeline import EmotionChain
from emotion_chain.reply_generator import QwenReplyGenerator
from emotion_chain.support_contacts import speakable


# --------------------------------------------------------------------------
# Cached model loaders. Imports are lazy so the mode-selection screen never
# pulls in torch/transformers before the user has actually chosen a path.
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading C3 stress ensemble (BERT + DeBERTa-v3)...")
def load_stress_predictor():
    return services.build_stress_predictor()


@st.cache_resource(show_spinner="Loading C3 CBT ensemble (BERT + MentalBERT + DeBERTa-v3)...")
def load_cbt_predictor():
    return services.build_cbt_predictor()


@st.cache_resource(show_spinner="Loading the emotion chain...")
def load_emotion_chain() -> EmotionChain:
    return services.build_emotion_chain()


@st.cache_resource(show_spinner=False)
def load_reply_generator() -> QwenReplyGenerator:
    # Qwen + adapter stay lazy inside the generator; first generate() call
    # pays the load cost -- on this machine, or on the configured support node.
    return services.build_reply_generator()


@st.cache_resource(show_spinner=False)
def load_workflow():
    components = AgentComponents(
        stress_predictor=load_stress_predictor(),
        cbt_predictor=load_cbt_predictor(),
        emotion_chain=load_emotion_chain(),
        reply_generator=load_reply_generator(),
        fusion_stress_weight=settings.FUSION_STRESS_WEIGHT,
        fusion_cbt_weight=settings.FUSION_CBT_WEIGHT,
        fusion_threshold=settings.FUSION_THRESHOLD,
        fusion_method=settings.FUSION_METHOD,
        fusion_stress_threshold=settings.FUSION_STRESS_THRESHOLD,
        fusion_cbt_threshold=settings.FUSION_CBT_THRESHOLD,
        next_negative_min_confidence=settings.NEXT_NEGATIVE_MIN_CONFIDENCE,
        support_contacts=settings.SUPPORT_CONTACTS,
    )
    return build_supportive_graph(components)


@st.cache_resource(show_spinner=False)
def load_voice_workflow():
    components = AgentComponents(
        emotion_chain=load_emotion_chain(),
        reply_generator=load_reply_generator(),
        next_negative_min_confidence=settings.NEXT_NEGATIVE_MIN_CONFIDENCE,
        support_contacts=settings.SUPPORT_CONTACTS,
    )
    return build_voice_supportive_graph(components)


@st.cache_resource(show_spinner=False)
def load_voice_services():
    return services.build_voice_services()


# --------------------------------------------------------------------------
# C1 sensor gate
# --------------------------------------------------------------------------


def render_sensor_gate() -> bool:
    """Render calibration status; return True once the chat should unlock."""

    from c1 import C1ServiceConfig, SerialSensorService, ServiceState

    if "c1_service" not in st.session_state:
        st.session_state.c1_service = None
    if "sensor_skipped" not in st.session_state:
        st.session_state.sensor_skipped = False

    if st.session_state.sensor_skipped:
        st.caption("Sensor skipped — running without the wearable stress signal.")
        return True

    service = st.session_state.c1_service

    if service is None:
        st.header("Step 1 — Connect the wearable sensor")
        st.write(
            "C1 reads BVP/EDA continuously in the background and needs about "
            "**6 minutes** (5 min personal calibration + 1 min warm-up) "
            "before its first stress prediction. It keeps predicting every "
            "5 seconds after that."
        )
        port = st.text_input("Serial port (live wearable)", value=settings.C1_DEFAULT_PORT)
        if st.button("Start sensor", type="primary", use_container_width=True):
            new_service = SerialSensorService(
                C1ServiceConfig(
                    port=port,
                    smoothing_window=settings.C1_SMOOTHING_WINDOW,
                    smoothing_required=settings.C1_SMOOTHING_REQUIRED,
                )
            )
            try:
                new_service.start()
            except Exception as exc:  # pragma: no cover - hardware dependent
                st.error(f"Could not open {port}: {exc}")
            else:
                st.session_state.c1_service = new_service
                st.rerun()

        st.divider()
        st.caption("Or, for testing without wearing it:")
        uploads = st.file_uploader(
            "Upload a recorded capture",
            type=("txt", "log", "csv"),
            accept_multiple_files=True,
            help=(
                "Either a text file of firmware DATA,<millis>,<ir>,<eda> lines "
                "(a captured serial session), or an Empatica E4 / WESAD "
                "session's BVP.csv + EDA.csv together (other files in the "
                "folder, like TEMP.csv or tags.csv, are ignored). Either way "
                "it replays at real time, so calibration/warm-up still take "
                "the full ~6 minutes and predictions keep arriving every 5 "
                "seconds."
            ),
        )
        if st.button("Upload & start", disabled=not uploads, use_container_width=True):
            from c1 import (
                BVP_SOURCE_E4,
                BVP_SOURCE_MAX30102,
                ReplaySerialConnection,
                convert_e4_to_firmware_lines,
                parse_recording,
            )

            contents = {item.name: item.getvalue() for item in uploads}
            bvp_raw = next((c for n, c in contents.items() if "bvp" in n.lower()), None)
            eda_raw = next((c for n, c in contents.items() if "eda" in n.lower()), None)

            try:
                if bvp_raw is not None and eda_raw is not None:
                    lines = convert_e4_to_firmware_lines(bvp_raw, eda_raw)
                    label = "E4 BVP+EDA"
                    # E4 BVP is band-passed and zero-centred; the MAX30102
                    # finger-presence check would reject every window of it.
                    bvp_source = BVP_SOURCE_E4
                elif len(contents) == 1:
                    lines = parse_recording(next(iter(contents.values())))
                    label = next(iter(contents))
                    bvp_source = BVP_SOURCE_MAX30102
                else:
                    raise ValueError(
                        "Select either one firmware DATA,... capture file, or "
                        "a BVP.csv + EDA.csv pair"
                    )
            except ValueError as exc:
                st.error(str(exc))
            else:
                connection = ReplaySerialConnection(lines)
                new_service = SerialSensorService(
                    C1ServiceConfig(
                        port=f"replay:{label}",
                        smoothing_window=settings.C1_SMOOTHING_WINDOW,
                        smoothing_required=settings.C1_SMOOTHING_REQUIRED,
                        bvp_source=bvp_source,
                    ),
                    serial_factory=lambda **_: connection,
                )
                new_service.start()
                st.session_state.c1_service = new_service
                st.rerun()

        st.button(
            "No wearable, no recording — skip (chat without a stress signal)",
            use_container_width=True,
            on_click=lambda: st.session_state.update(sensor_skipped=True),
        )
        return False

    snapshot = service.snapshot()
    calibration_sec = float(service.predictor.manifest.get("calibration_sec", 300.0))

    if snapshot.state == ServiceState.ERROR:
        st.error(f"Sensor error: {snapshot.last_error}")
        if st.button("Discard sensor and skip instead"):
            service.stop()
            st.session_state.c1_service = None
            st.session_state.sensor_skipped = True
            st.rerun()
        return False

    if snapshot.state in (ServiceState.CONNECTING, ServiceState.CALIBRATING):
        progress = 0.0
        if calibration_sec > 0:
            progress = max(0.0, min(1.0, 1 - snapshot.calibration_remaining_seconds / calibration_sec))
        st.header("Step 1 — Calibrating (personal baseline)")
        st.progress(progress)
        st.caption(f"~{int(snapshot.calibration_remaining_seconds)}s left")
        time.sleep(1)
        st.rerun()

    if snapshot.state == ServiceState.WARMING_UP:
        st.header("Step 1 — Calibrated, collecting first window")
        st.progress(1.0)
        st.caption(f"~{int(snapshot.warmup_remaining_seconds)}s left")
        time.sleep(1)
        st.rerun()

    # ServiceState.RUNNING
    return True


def latest_sensor_signal() -> tuple[bool, float | None, bool]:
    """Return (sensor_stress, sensor_stress_probability, sensor_available)."""

    if st.session_state.get("sensor_skipped"):
        return False, None, False
    service = st.session_state.get("c1_service")
    if service is None:
        return False, None, False
    snapshot = service.snapshot()
    if snapshot.last_prediction is None:
        return False, None, False
    return (
        bool(snapshot.smoothed_is_stressed),
        float(snapshot.smoothed_probability),
        True,
    )


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------


def render_support_contacts(contacts: list[dict[str, str]]) -> None:
    """Show the contacts the reply already names, as a panel that can be read
    at a glance by someone who is not going to re-read a paragraph."""

    if not contacts:
        return
    lines = "\n".join(
        f"- **{contact['name']}** — {contact['phone']}"
        + (f"  \n  {contact['note']}" if contact.get("note") else "")
        for contact in contacts
    )
    st.warning("Talk to someone now\n\n" + lines)


def render_chat() -> None:
    if "history" not in st.session_state:
        st.session_state.history: list[tuple[str, str]] = []
    if "previous_emotion" not in st.session_state:
        # Deviation baseline: the graph is stateless between invocations, so
        # the caller carries the previous turn's emotion.
        st.session_state.previous_emotion = None

    sensor_stress, sensor_stress_probability, sensor_available = latest_sensor_signal()
    if sensor_available:
        st.sidebar.metric(
            "Live sensor stress",
            "Stressed" if sensor_stress else "Calm",
            f"{sensor_stress_probability:.0%}",
        )
        if st.sidebar.button("Restart calibration"):
            st.session_state.c1_service.restart_calibration()
            st.rerun()

    for speaker, text in st.session_state.history:
        with st.chat_message("user" if speaker == "user" else "assistant"):
            st.write(text)

    user_text = st.chat_input("How are you doing right now?")
    if not user_text:
        return

    with st.chat_message("user"):
        st.write(user_text)

    state = {
        "user_text": user_text,
        "history": list(st.session_state.history),
        "previous_emotion": st.session_state.previous_emotion,
        "sensor_stress": sensor_stress,
        "sensor_stress_probability": sensor_stress_probability,
        "sensor_stress_available": sensor_available,
    }
    with st.spinner("Thinking..."):
        workflow = load_workflow()
        result = workflow.invoke(state)

    with st.chat_message("assistant"):
        st.write(result["reply"])
        details = f"route={result.get('reply_route')} · source={result.get('reply_source')}"
        if result.get("reply_route") != "crisis":
            details = (
                f"current={result.get('current_emotion')} · "
                f"next={result.get('next_emotion')} "
                f"(intensity {result.get('next_intensity')}) · "
                f"deviation={result.get('deviation_level')} "
                f"({result.get('deviation_score', 0.0):.2f}) · " + details
            )
        st.caption(details)
        render_support_contacts(result.get("support_contacts", []))

    # A crisis turn never reaches the classifier, so it leaves the deviation
    # baseline untouched instead of resetting it.
    if result.get("current_emotion"):
        st.session_state.previous_emotion = result["current_emotion"]

    st.session_state.history.append(("user", user_text))
    st.session_state.history.append(("assistant", result["reply"]))


def render_voice_chat() -> None:
    """Render the C2/STT/emotion/Qwen/TTS branch without calling C3."""

    if "voice_history" not in st.session_state:
        st.session_state.voice_history = []
    if "previous_emotion" not in st.session_state:
        st.session_state.previous_emotion = None

    sensor_stress, sensor_probability, sensor_available = latest_sensor_signal()
    if sensor_available:
        st.sidebar.metric(
            "Live sensor stress",
            "Stressed" if sensor_stress else "Calm",
            f"{sensor_probability:.0%}",
        )

    for turn in st.session_state.voice_history:
        with st.chat_message("user"):
            st.caption("Transcript")
            st.write(turn["transcript"])
        with st.chat_message("assistant"):
            st.write(turn["reply"])
            st.audio(turn["audio"], format="audio/wav")
            st.caption(turn["details"])
            render_support_contacts(turn.get("support_contacts", []))

    recording = st.audio_input("Record how you are feeling")
    if recording is None:
        return
    if not st.button("Analyze voice & reply", type="primary", use_container_width=True):
        return

    content = recording.getvalue()
    if len(content) > settings.C2_MAX_UPLOAD_BYTES:
        st.error(
            f"Recording exceeds the {settings.C2_MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."
        )
        return

    from c2 import decode_audio

    with st.spinner("Analysing C2 voice appraisal and transcribing..."):
        try:
            waveform = decode_audio(content)
            duration = waveform.size / 16_000
            if duration > settings.C2_MAX_AUDIO_SECONDS:
                raise ValueError(
                    f"Recording is {duration:.1f}s; maximum is "
                    f"{settings.C2_MAX_AUDIO_SECONDS:.0f}s"
                )
            appraisal_analyzer, transcriber, synthesizer = load_voice_services()
            appraisal = appraisal_analyzer.predict(waveform, file_name=recording.name)
            transcription = transcriber.transcribe(waveform)
            state = {
                "user_text": transcription.text,
                "history": [
                    item
                    for turn in st.session_state.voice_history
                    for item in (
                        ("user", turn["transcript"]),
                        ("assistant", turn["reply"]),
                    )
                ],
                "sensor_stress": sensor_stress,
                "sensor_stress_probability": sensor_probability,
                "sensor_stress_available": sensor_available,
                "previous_emotion": st.session_state.previous_emotion,
                "voice_stress": appraisal.voice_stress,
                "voice_stress_score": appraisal.voice_stress_score,
                "voice_appraisal_state": appraisal.dominant_state,
                "voice_appraisal_uncertain": appraisal.uncertain,
            }
            result = load_voice_workflow().invoke(state)
            reply_audio = synthesizer.synthesize(
                speakable(result["reply"], settings.SUPPORT_CONTACTS)
            )
        except Exception as exc:
            st.error(str(exc))
            return

    appraisal_label = "uncertain" if appraisal.uncertain else appraisal.dominant_state
    details = (
        f"C2={appraisal_label} · voice_stress={appraisal.voice_stress} · "
        f"current={result.get('current_emotion')} · "
        f"next={result.get('next_emotion')} "
        f"(intensity {result.get('next_intensity')}) · "
        f"deviation={result.get('deviation_level')} "
        f"({result.get('deviation_score', 0.0):.2f}) · "
        f"route={result.get('reply_route')} · source={result.get('reply_source')}"
    )
    if result.get("current_emotion"):
        st.session_state.previous_emotion = result["current_emotion"]
    st.session_state.voice_history.append(
        {
            "transcript": transcription.text,
            "reply": result["reply"],
            "audio": reply_audio,
            "details": details,
            "support_contacts": result.get("support_contacts", []),
        }
    )
    st.rerun()


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

st.set_page_config(page_title="Supportive agent", page_icon="🧭", layout="wide")

if "mode" not in st.session_state:
    st.session_state.mode = None

if st.session_state.mode is not None:
    if st.sidebar.button("⬅ Change mode"):
        st.session_state.mode = None
        st.rerun()

if st.session_state.mode is None:
    st.title("Supportive agent")
    st.caption("Choose how you'd like to check in.")
    text_col, voice_col = st.columns(2)
    with text_col:
        st.subheader("💬 Text")
        st.write(
            "Type how you're feeling. A wearable sensor runs quietly in the "
            "background to sense stress."
        )
        if st.button("Continue with text", type="primary", use_container_width=True):
            st.session_state.mode = "text"
            st.rerun()
    with voice_col:
        st.subheader("🎙️ Voice")
        st.write("Speak instead of typing.")
        if st.button("Continue with voice", type="primary", use_container_width=True):
            st.session_state.mode = "voice"
            st.rerun()
elif st.session_state.mode == "voice":
    st.title("Voice check-in")
    if render_sensor_gate():
        render_voice_chat()
elif st.session_state.mode == "text":
    st.title("Text check-in")
    if render_sensor_gate():
        render_chat()
