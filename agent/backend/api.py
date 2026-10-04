"""FastAPI backend for the React supportive-agent frontend.

Text mode runs C1 -> C3 Stress/CBT -> current emotion + next intensity ->
LangGraph -> Qwen. Voice mode runs C1 + C2 voice appraisal -> speech-to-text
-> current emotion + next intensity -> LangGraph -> Qwen -> Kokoro. C3 is
deliberately absent from the voice graph. Loaded models and the C1 sensor are
process-global; the conversation itself is not -- each turn carries a
``session_id`` and its history lives in the SQLite store from ``chat_store.py``.

Run from this directory: ``uvicorn api:app --reload --port 8005 --reload-exclude ".venv/*"``
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from auth import accounts, current_user, patient_id, protect_api, router as auth_router

import services
import settings
from chat_store import DEFAULT_SESSION_ID, ChatStore, normalize_session_id
from emotion_chain.agent_graph import (
    AgentComponents,
    build_supportive_graph,
    build_voice_supportive_graph,
)
from emotion_chain.pipeline import EmotionChain
from emotion_chain.multimodal_stress import (
    QuestionnaireResult,
    fuse_multimodal_stress,
    score_questionnaire,
)
from emotion_chain.reply_generator import QwenReplyGenerator
from emotion_chain.support_contacts import speakable
from emotion_chain.trace import build_trace

logging.basicConfig(level=logging.INFO)
services.log_deployment_summary()

app = FastAPI(title="Supportive agent API", dependencies=[Depends(protect_api)])
app.include_router(auth_router)
app.add_middleware(
    CORSMiddleware,
    # Configured in config.yaml so the UI can also be opened from another
    # device on the LAN without editing this file.
    allow_origins=list(settings.API_CORS_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Lazy-loaded workflows. The preparation endpoints now load them before the
# chat UI opens; direct API clients still fall back to loading on first use.
# --------------------------------------------------------------------------

_workflow = None
_voice_workflow = None
_components: AgentComponents | None = None
_voice_components: AgentComponents | None = None
_emotion_chain = None
_reply_generator = None
_voice_services = None
_workflow_lock = threading.RLock()
_pipeline_lock = threading.Lock()

_TEXT_MODEL_KEYS = (
    "stress_language",
    "cognitive_patterns",
    "current_emotion",
    "next_turn_forecast",
)
_text_model_status_lock = threading.Lock()
_text_model_status = {
    key: {"status": "pending", "error": None} for key in _TEXT_MODEL_KEYS
}

_VOICE_MODEL_KEYS = (
    "speech_recognition",
    "vocal_cues",
    "current_emotion",
    "next_turn_forecast",
    "spoken_replies",
)
_voice_model_status_lock = threading.Lock()
_voice_prepare_lock = threading.Lock()
_voice_model_status = {
    key: {"status": "pending", "error": None} for key in _VOICE_MODEL_KEYS
}


def _set_text_model_status(
    key: str,
    status: Literal["pending", "loading", "ready", "error"],
    error: str | None = None,
) -> None:
    with _text_model_status_lock:
        _text_model_status[key] = {"status": status, "error": error}


def _text_model_status_payload() -> dict:
    with _text_model_status_lock:
        models = {key: dict(value) for key, value in _text_model_status.items()}
    return {
        "ready": all(model["status"] == "ready" for model in models.values()),
        "models": models,
    }


def _set_voice_model_status(
    key: str,
    status: Literal["pending", "loading", "ready", "error"],
    error: str | None = None,
) -> None:
    with _voice_model_status_lock:
        _voice_model_status[key] = {"status": status, "error": error}


def _voice_model_status_payload() -> dict:
    with _voice_model_status_lock:
        models = {key: dict(value) for key, value in _voice_model_status.items()}
    return {
        "ready": all(model["status"] == "ready" for model in models.values()),
        "models": models,
    }


def _initialize_shared_models() -> tuple[EmotionChain, QwenReplyGenerator]:
    global _emotion_chain, _reply_generator
    if _emotion_chain is None:
        _emotion_chain = services.build_emotion_chain()
    if _reply_generator is None:
        _reply_generator = services.build_reply_generator()
    return _emotion_chain, _reply_generator


def get_workflow():
    global _workflow, _components
    if _workflow is None:
        with _workflow_lock:
            if _workflow is None:
                active_stage = "stress_language"
                try:
                    _set_text_model_status(active_stage, "loading")
                    stress_predictor = services.build_stress_predictor()
                    _set_text_model_status(active_stage, "ready")

                    active_stage = "cognitive_patterns"
                    _set_text_model_status(active_stage, "loading")
                    cbt_predictor = services.build_cbt_predictor()
                    _set_text_model_status(active_stage, "ready")

                    # EmotionChain owns both the present-emotion classifier and
                    # its next-turn intensity forecaster, so these two stages
                    # initialize together and share the same encoder.
                    active_stage = "emotion_chain"
                    _set_text_model_status("current_emotion", "loading")
                    _set_text_model_status("next_turn_forecast", "loading")
                    emotion_chain, reply_generator = _initialize_shared_models()
                    _set_text_model_status("current_emotion", "ready")
                    _set_text_model_status("next_turn_forecast", "ready")
                except Exception as exc:
                    message = str(exc)
                    if active_stage == "emotion_chain":
                        _set_text_model_status("current_emotion", "error", message)
                        _set_text_model_status("next_turn_forecast", "error", message)
                    else:
                        _set_text_model_status(active_stage, "error", message)
                    raise
                components = AgentComponents(
                    stress_predictor=stress_predictor,
                    cbt_predictor=cbt_predictor,
                    emotion_chain=emotion_chain,
                    reply_generator=reply_generator,
                    fusion_stress_weight=settings.FUSION_STRESS_WEIGHT,
                    fusion_cbt_weight=settings.FUSION_CBT_WEIGHT,
                    fusion_threshold=settings.FUSION_THRESHOLD,
                    fusion_method=settings.FUSION_METHOD,
                    fusion_stress_threshold=settings.FUSION_STRESS_THRESHOLD,
                    fusion_cbt_threshold=settings.FUSION_CBT_THRESHOLD,
                    next_negative_min_confidence=settings.NEXT_NEGATIVE_MIN_CONFIDENCE,
                    support_contacts=settings.SUPPORT_CONTACTS,
                )
                _components = components
                _workflow = build_supportive_graph(components)
    return _workflow


def get_voice_workflow():
    """Return the LangGraph branch that intentionally contains no C3 nodes."""

    global _voice_workflow, _voice_components
    if _voice_workflow is None:
        with _workflow_lock:
            if _voice_workflow is None:
                emotion_chain, reply_generator = _initialize_shared_models()
                components = AgentComponents(
                    emotion_chain=emotion_chain,
                    reply_generator=reply_generator,
                    next_negative_min_confidence=settings.NEXT_NEGATIVE_MIN_CONFIDENCE,
                    support_contacts=settings.SUPPORT_CONTACTS,
                )
                _voice_components = components
                _voice_workflow = build_voice_supportive_graph(components)
    return _voice_workflow


@app.get("/api/models/text/status")
def text_model_status() -> dict:
    """Report real initialization state for the text-analysis workflow."""

    return _text_model_status_payload()


@app.post("/api/models/text/prepare")
def prepare_text_models() -> dict:
    """Load the text workflow before the user reaches the conversation UI."""

    try:
        get_workflow()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Text analysis could not be prepared: {exc}",
        ) from exc
    return _text_model_status_payload()


@app.get("/api/models/voice/status")
def voice_model_status() -> dict:
    """Report real initialization state for the voice conversation workflow."""

    return _voice_model_status_payload()


@app.post("/api/models/voice/prepare")
def prepare_voice_models() -> dict:
    """Load voice input, emotion and spoken-output services before recording."""

    with _voice_prepare_lock:
        if _voice_model_status_payload()["ready"]:
            return _voice_model_status_payload()
        active_stage = "speech_recognition"
        try:
            appraisal_analyzer, transcriber, synthesizer = get_voice_services()

            _set_voice_model_status(active_stage, "loading")
            transcriber.load()
            _set_voice_model_status(active_stage, "ready")

            active_stage = "vocal_cues"
            _set_voice_model_status(active_stage, "loading")
            appraisal_analyzer.load()
            _set_voice_model_status(active_stage, "ready")

            active_stage = "emotion_chain"
            _set_voice_model_status("current_emotion", "loading")
            _set_voice_model_status("next_turn_forecast", "loading")
            get_voice_workflow()
            _set_voice_model_status("current_emotion", "ready")
            _set_voice_model_status("next_turn_forecast", "ready")

            active_stage = "spoken_replies"
            _set_voice_model_status(active_stage, "loading")
            synthesizer.load()
            _set_voice_model_status(active_stage, "ready")
        except Exception as exc:
            message = str(exc)
            if active_stage == "emotion_chain":
                _set_voice_model_status("current_emotion", "error", message)
                _set_voice_model_status("next_turn_forecast", "error", message)
            else:
                _set_voice_model_status(active_stage, "error", message)
            raise HTTPException(
                status_code=503,
                detail=f"Voice conversation could not be prepared: {exc}",
            ) from exc
    return _voice_model_status_payload()


def get_voice_services():
    """Create lightweight wrappers; each heavy voice model remains lazy.

    Any of the three may be a support-node client instead of a local model --
    ``services`` decides from the config, and both kinds behave identically.
    """

    global _voice_services
    if _voice_services is None:
        with _workflow_lock:
            if _voice_services is None:
                _voice_services = services.build_voice_services()
    return _voice_services


def get_components() -> AgentComponents:
    """Return the same loaded component instances used by LangGraph."""

    get_workflow()
    if _components is None:  # defensive; get_workflow initializes both
        raise RuntimeError("Supportive-agent components were not initialized")
    return _components


def get_voice_components() -> AgentComponents:
    get_voice_workflow()
    if _voice_components is None:
        raise RuntimeError("Voice-agent components were not initialized")
    return _voice_components


# --------------------------------------------------------------------------
# C1 sensor state
# --------------------------------------------------------------------------

_c1_service = None
_sensor_skipped = False
_sensor_patient_id: str | None = None
_sensor_lock = threading.Lock()
# Conversation history and the previous turn's current-emotion label, per
# session. The graph is stateless between invocations, so both have to be read
# out and handed back in on every turn; see chat_store.py.
_store = ChatStore(
    settings.CHAT_DB_PATH,
    max_messages=settings.CHAT_HISTORY_MAX_MESSAGES,
    ttl_seconds=settings.CHAT_HISTORY_TTL_SECONDS,
)
_turn_contexts: dict[str, dict] = {}
_turn_context_lock = threading.Lock()
_voice_audio_outputs: dict[str, bytes] = {}
_voice_audio_lock = threading.Lock()
_questionnaire_results: dict[str, QuestionnaireResult] = {}
_questionnaire_lock = threading.Lock()


class StartSensorRequest(BaseModel):
    port: str = settings.C1_DEFAULT_PORT


class ChatRequest(BaseModel):
    message: str
    # Accepted for older clients; account identity always determines ownership.
    session_id: str | None = None


class ResetChatRequest(BaseModel):
    session_id: str | None = None


class QuestionnaireRequest(BaseModel):
    answers: dict[str, int]
    session_id: str | None = None


class XAIRequest(BaseModel):
    message: str
    turn_id: str | None = None


class ChatMessageOut(BaseModel):
    role: Literal["user", "assistant"]
    text: str
    # Present on assistant messages recorded since traces were stored; the
    # shape is emotion_chain/trace.py's, kept loose here so adding a step does
    # not need a matching model change.
    trace: dict | None = None


def _sensor_status_payload(*, include_waveform: bool = False) -> dict:
    if _sensor_skipped:
        return {"mode": "skipped", "ready": True}
    if _c1_service is None:
        return {"mode": "idle", "ready": False}

    from c1 import BVP_SOURCE_E4, ServiceState

    snapshot = _c1_service.snapshot()
    calibration_sec = float(_c1_service.predictor.manifest.get("calibration_sec", 300.0))
    payload = {
        "mode": "sensor",
        "state": snapshot.state.value,
        "ready": snapshot.state == ServiceState.RUNNING,
        "calibration_remaining_seconds": snapshot.calibration_remaining_seconds,
        "calibration_total_seconds": calibration_sec,
        "warmup_remaining_seconds": snapshot.warmup_remaining_seconds,
        "error": snapshot.last_error,
        "sample_count": snapshot.sample_count,
        "elapsed_seconds": snapshot.elapsed_seconds,
        "serial_line_count": snapshot.serial_line_count,
        "invalid_line_count": snapshot.invalid_line_count,
        "data_progress_age_seconds": snapshot.data_progress_age_seconds,
        "stream_stalled": snapshot.stream_stalled,
    }
    waveform_samples = (
        _c1_service.recent_samples(duration_seconds=8.0, max_points=420)
        if include_waveform
        else ()
    )
    if waveform_samples:
        is_e4 = _c1_service.config.bvp_source == BVP_SOURCE_E4
        payload["sensor_waveform"] = {
            "times": [sample.t_sec for sample in waveform_samples],
            "bvp": [sample.bvp for sample in waveform_samples],
            "eda": [sample.eda for sample in waveform_samples],
            "bvp_unit": "a.u." if is_e4 else "raw",
            "eda_unit": "µS" if is_e4 else "raw",
        }
    if snapshot.last_prediction is not None:
        payload["sensor_stress"] = bool(snapshot.smoothed_is_stressed)
        payload["sensor_stress_probability"] = float(snapshot.smoothed_probability)
        # The window end is a stable identifier for each new five-second
        # prediction.  The frontend uses it to distinguish a fresh result
        # from an ordinary status poll whose value happens to be unchanged.
        payload["sensor_prediction_window_end"] = float(
            snapshot.last_prediction.window_end_t
        )
        payload["sensor_smoothing_count"] = len(snapshot.recent_predictions)
        payload["sensor_smoothing_window"] = _c1_service.config.smoothing_window
    return payload


@app.get("/api/health")
def health() -> dict:
    """Liveness plus where each offloadable model is configured to run.

    Deliberately does no network I/O so the frontend's poll stays cheap; use
    ``/api/deployment`` to actually reach out and check a support node.
    """

    return {"ok": True}


@app.get("/api/deployment")
def deployment() -> dict:
    """Probe every configured support node and report whether it answers.

    The first stop when a reply comes back with ``source="template_fallback"``
    or voice mode fails: it distinguishes an unreachable node from a model
    error on a reachable one.
    """

    from remote_clients import node_health

    summary = services.deployment_summary()
    urls = {
        url
        for url in (
            settings.REMOTE_REPLY_URL,
            settings.REMOTE_STT_URL,
            settings.REMOTE_TTS_URL,
            settings.REMOTE_C2_URL,
        )
        if url
    }
    return {
        "services": summary,
        "nodes": [
            node_health(url, auth_token=settings.REMOTE_AUTH_TOKEN) for url in sorted(urls)
        ],
    }


@app.post("/api/sensor/start")
def start_sensor(payload: StartSensorRequest, request: Request) -> dict:
    global _c1_service, _sensor_skipped, _sensor_patient_id
    from c1 import C1ServiceConfig, SerialSensorService

    with _sensor_lock:
        _claim_sensor(request)
        if _c1_service is None:
            service = SerialSensorService(
                C1ServiceConfig(
                    port=payload.port,
                    smoothing_window=settings.C1_SMOOTHING_WINDOW,
                    smoothing_required=settings.C1_SMOOTHING_REQUIRED,
                )
            )
            try:
                service.start()
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            _c1_service = service
            _sensor_skipped = False
        return _sensor_status_payload()


@app.post("/api/sensor/upload")
def upload_sensor_recording(request: Request, files: list[UploadFile] = File(...)) -> dict:
    """Replay an uploaded recording instead of a live wearable.

    Accepts either one firmware ``DATA,<millis>,<ir>,<eda>`` capture file, or
    an Empatica E4 / WESAD-style session (a ``BVP.csv`` + ``EDA.csv`` pair,
    matched by filename; any other files in the selection, e.g. ``TEMP.csv``,
    ``HR.csv``, ``IBI.csv``, ``ACC.csv``, ``tags.csv``, ``info.txt``, are
    ignored since C1 only consumes BVP + EDA). Either way it runs the exact
    same calibration/warm-up/prediction timeline as a real sensor, paced in
    real time, so testing doesn't require wearing the device for every run.
    """

    global _c1_service, _sensor_skipped, _sensor_patient_id
    from c1 import (
        BVP_SOURCE_E4,
        BVP_SOURCE_MAX30102,
        C1ServiceConfig,
        ReplaySerialConnection,
        SerialSensorService,
        convert_e4_to_firmware_lines,
        parse_recording,
        recording_duration_seconds,
    )

    # Keep a list rather than a filename-keyed dict: two files from different
    # folders can share a name, and collapsing them used to turn two BVP files
    # into one apparent firmware capture with a misleading DATA-line error.
    contents = [
        (f.filename or f"file{i}", f.file.read()) for i, f in enumerate(files)
    ]
    bvp_files = [(name, raw) for name, raw in contents if "bvp" in name.lower()]
    eda_files = [(name, raw) for name, raw in contents if "eda" in name.lower()]

    try:
        if bvp_files or eda_files:
            if len(bvp_files) != 1 or len(eda_files) != 1:
                bvp_description = (
                    "no BVP file" if not bvp_files else f"{len(bvp_files)} BVP files"
                )
                eda_description = (
                    "no EDA file" if not eda_files else f"{len(eda_files)} EDA files"
                )
                raise ValueError(
                    "Upload exactly one BVP.csv and one EDA.csv file. "
                    f"Found {bvp_description} and {eda_description}."
                )
            lines = convert_e4_to_firmware_lines(
                bvp_files[0][1], eda_files[0][1]
            )
            label = "E4 BVP+EDA"
            # E4 BVP is band-passed and zero-centred; the MAX30102
            # finger-presence check would reject every window of it.
            bvp_source = BVP_SOURCE_E4
        elif len(contents) == 1:
            lines = parse_recording(contents[0][1])
            label = "firmware capture"
            bvp_source = BVP_SOURCE_MAX30102
        else:
            raise ValueError(
                "Select either one firmware DATA,... capture file, or a "
                "BVP.csv + EDA.csv pair"
            )
        duration = recording_duration_seconds(lines)
        required_duration = 360.0
        if duration < required_duration - 0.25:
            raise ValueError(
                f"The uploaded {label} contains only {duration:.1f} seconds of "
                "overlapping BVP and EDA. This C1 model requires at least 360 "
                "seconds: 300 seconds for personal calibration plus a fresh "
                "60-second prediction window."
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with _sensor_lock:
        _claim_sensor(request)
        if _c1_service is None:
            connection = ReplaySerialConnection(lines)
            service = SerialSensorService(
                C1ServiceConfig(
                    port=f"replay:{label}",
                    smoothing_window=settings.C1_SMOOTHING_WINDOW,
                    smoothing_required=settings.C1_SMOOTHING_REQUIRED,
                    bvp_source=bvp_source,
                ),
                serial_factory=lambda **_: connection,
            )
            service.start()
            _c1_service = service
            _sensor_skipped = False
        return _sensor_status_payload()


@app.post("/api/sensor/skip")
def skip_sensor(request: Request) -> dict:
    global _c1_service, _sensor_skipped, _sensor_patient_id
    with _sensor_lock:
        _claim_sensor(request)
        _sensor_skipped = True
        if _c1_service is not None:
            _c1_service.stop()
            _c1_service = None
        return _sensor_status_payload()


@app.post("/api/sensor/disconnect")
def disconnect_sensor(request: Request) -> dict:
    """Close the current stream and return to sensor setup for a clean retry."""

    global _c1_service, _sensor_skipped, _sensor_patient_id
    with _sensor_lock:
        _claim_sensor(request)
        if _c1_service is not None:
            _c1_service.stop()
        _c1_service = None
        _sensor_skipped = False
        _sensor_patient_id = None
        return _sensor_status_payload()


@app.post("/api/sensor/restart")
def restart_sensor(request: Request) -> dict:
    with _sensor_lock:
        _claim_sensor(request)
        if _c1_service is None:
            raise HTTPException(status_code=400, detail="Sensor is not running")
        _c1_service.restart_calibration()
        return _sensor_status_payload()


@app.get("/api/sensor/status")
def sensor_status(
    request: Request,
    session_id: str = DEFAULT_SESSION_ID,
    include_waveform: bool = False,
) -> dict:
    session_id = patient_id(request)
    with _sensor_lock:
        if _sensor_patient_id != session_id:
            return {"mode": "idle", "ready": False}
        payload = _sensor_status_payload(include_waveform=include_waveform)
    # The chat badge must never expose the raw five-second C1 result. Once C1
    # has a prediction, fuse that latest probability with this session's fixed
    # questionnaire result and return the combined value alongside the raw
    # fields retained for XAI.
    if payload.get("sensor_stress_probability") is not None:
        sensor_state = {
            "sensor_stress_available": True,
            "sensor_stress_probability": payload["sensor_stress_probability"],
        }
        multimodal_state, _ = _multimodal_turn_data(
            normalize_session_id(session_id), sensor_state
        )
        payload.update(multimodal_state)
    return payload


@app.get("/api/questionnaire/status")
def questionnaire_status(request: Request, session_id: str = DEFAULT_SESSION_ID) -> dict:
    session_id = patient_id(request)
    with _questionnaire_lock:
        result = _questionnaire_results.get(session_id)
    return _questionnaire_payload(result)


@app.post("/api/questionnaire")
def submit_questionnaire(payload: QuestionnaireRequest, request: Request) -> dict:
    try:
        result = score_questionnaire(
            payload.answers,
            threshold=settings.MULTIMODAL_STRESS_THRESHOLD,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    with _questionnaire_lock:
        _questionnaire_results[patient_id(request)] = result
    return _questionnaire_payload(result)


@app.get("/api/chat/history", response_model=list[ChatMessageOut], response_model_exclude_none=True)
def get_history(request: Request, session_id: str = DEFAULT_SESSION_ID) -> list[dict]:
    """Redraw a conversation the client no longer holds, after a page reload.

    Returns the whole retained session rather than the shorter window that is
    replayed into the prompt, so what is on screen is never *less* than what
    the model was given.
    """

    _store.purge_expired()
    messages = _store.transcript(patient_id(request))
    if current_user(request)["role"] == "patient":
        return [{"role": m["role"], "text": m["text"]} for m in messages]
    return messages


@app.post("/api/chat/reset")
def reset_chat(payload: ResetChatRequest, request: Request) -> dict:
    """Start a fresh conversation: forget the turns and the emotion baseline."""

    target = patient_id(request)
    _store.reset(target)
    with _turn_context_lock, _voice_audio_lock:
        for turn_id in [key for key, value in _turn_contexts.items() if value.get("patient_id") == target]:
            _turn_contexts.pop(turn_id, None)
            _voice_audio_outputs.pop(turn_id, None)
    return {"ok": True}


def _sensor_turn_data(session_id: str) -> tuple[dict, dict]:
    """Snapshot C1 once so inference and later XAI explain the same windows."""

    sensor_stress = False
    sensor_stress_probability = None
    sensor_available = False
    context = {
        "sensor_predictions": (),
        "sensor_smoothed_is_stressed": None,
        "sensor_calibration": None,
        "sensor_predictor": None,
    }
    if _c1_service is not None and _sensor_patient_id == session_id:
        from c1 import ServiceState

        snapshot = _c1_service.snapshot()
        context.update(
            {
                "sensor_predictions": snapshot.recent_predictions,
                "sensor_smoothed_is_stressed": snapshot.smoothed_is_stressed,
                "sensor_calibration": snapshot.calibration,
                "sensor_predictor": _c1_service.predictor,
            }
        )
        if (
            snapshot.state == ServiceState.RUNNING
            and not snapshot.stream_stalled
            and snapshot.last_prediction is not None
        ):
            sensor_stress = bool(snapshot.smoothed_is_stressed)
            sensor_stress_probability = float(snapshot.smoothed_probability)
            sensor_available = True
    return (
        {
            "sensor_stress": sensor_stress,
            "sensor_stress_probability": sensor_stress_probability,
            "sensor_stress_available": sensor_available,
        },
        context,
    )


def _questionnaire_payload(result: QuestionnaireResult | None) -> dict:
    if result is None:
        return {"completed": False}
    return {
        "completed": True,
        "stress": result.is_stressed,
        "stress_probability": result.probability,
    }


def _multimodal_turn_data(
    session_id: str,
    sensor_state: dict,
    *,
    voice_probability: float | None = None,
) -> tuple[dict, dict]:
    """Fuse the saved check-in with the inputs available for this turn."""

    with _questionnaire_lock:
        questionnaire = _questionnaire_results.get(session_id)
    sensor_probability = (
        sensor_state.get("sensor_stress_probability")
        if sensor_state.get("sensor_stress_available")
        else None
    )
    fused = fuse_multimodal_stress(
        questionnaire=questionnaire.probability if questionnaire else None,
        sensor=sensor_probability,
        voice=voice_probability,
        questionnaire_weight=settings.MULTIMODAL_QUESTIONNAIRE_WEIGHT,
        sensor_weight=settings.MULTIMODAL_SENSOR_WEIGHT,
        voice_weight=settings.MULTIMODAL_VOICE_WEIGHT,
        threshold=settings.MULTIMODAL_STRESS_THRESHOLD,
    )
    state = {
        "questionnaire_stress_available": questionnaire is not None,
        "questionnaire_stress": questionnaire.is_stressed if questionnaire else False,
        "questionnaire_stress_probability": (
            questionnaire.probability if questionnaire else None
        ),
        "multimodal_stress_available": fused.available,
        "multimodal_stress": fused.is_stressed,
        "multimodal_stress_probability": fused.probability,
        "multimodal_stress_weights": fused.normalized_weights,
        "multimodal_stress_inputs": fused.probabilities,
    }
    context = {
        "questionnaire": questionnaire,
        "multimodal_stress": fused,
    }
    return state, context


def _remember_turn_messages(
    session_id: str,
    user_text: str,
    result: dict,
    trace: dict | None = None,
) -> None:
    """Persist one completed exchange plus the next deviation baseline.

    A crisis turn short-circuits before the classifier runs, so it carries no
    ``current_emotion``; ``set_previous_emotion`` leaves the baseline alone in
    that case rather than clearing it.
    """

    _store.append(session_id, "user", user_text)
    _store.append(session_id, "assistant", result["reply"], trace=trace)
    _store.set_previous_emotion(session_id, result.get("current_emotion"))


def _deviation_payload(result: dict) -> dict:
    return {
        "previous_emotion": result.get("previous_emotion"),
        "deviation_level": result.get("deviation_level"),
        "deviation_score": result.get("deviation_score"),
        "support_contacts": result.get("support_contacts", []),
    }


def _remember_turn(turn_id: str, context: dict) -> None:
    with _turn_context_lock:
        _turn_contexts[turn_id] = context
        while len(_turn_contexts) > 50:
            _turn_contexts.pop(next(iter(_turn_contexts)))


@app.post("/api/chat")
def chat(payload: ChatRequest, request: Request) -> dict:
    text = payload.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    session_id = patient_id(request)
    with _sensor_lock:
        sensor_state, sensor_context = _sensor_turn_data(session_id)
    multimodal_state, multimodal_context = _multimodal_turn_data(
        session_id, sensor_state
    )
    with _pipeline_lock:
        state = {
            "user_text": text,
            # Two rows per exchange, so this is `history_turns` exchanges.
            "history": _store.history(session_id, settings.REPLY_HISTORY_TURNS * 2),
            "previous_emotion": _store.previous_emotion(session_id),
            **sensor_state,
            **multimodal_state,
        }
        result = get_workflow().invoke(state)

    trace = build_trace(
        result,
        mode="text",
        next_negative_min_confidence=settings.NEXT_NEGATIVE_MIN_CONFIDENCE,
    )
    turn_id = uuid.uuid4().hex
    trace["turn_id"] = turn_id
    _remember_turn_messages(session_id, text, result, trace)
    _remember_turn(
        turn_id,
        {
            "patient_id": session_id,
            "mode": "text",
            "message": text,
            **sensor_context,
            **multimodal_context,
        },
    )

    return _patient_response({
        "turn_id": turn_id,
        "reply": result["reply"],
        "route": result.get("reply_route"),
        "source": result.get("reply_source"),
        "current_emotion": result.get("current_emotion"),
        "next_emotion": result.get("next_emotion"),
        "next_emotion_confidence": result.get("next_emotion_confidence"),
        "next_intensity": result.get("next_intensity"),
        "next_intensity_probability": result.get("next_intensity_probability"),
        "fusion_method": result.get("text_signal_method"),
        "text_signal_probability": result.get("text_signal_probability"),
        "questionnaire_stress_probability": result.get(
            "questionnaire_stress_probability"
        ),
        "multimodal_stress": result.get("multimodal_stress"),
        "multimodal_stress_probability": result.get(
            "multimodal_stress_probability"
        ),
        "multimodal_stress_weights": result.get("multimodal_stress_weights", {}),
        "trace": trace,
        **_deviation_payload(result),
    })


@app.post("/api/voice/chat")
def voice_chat(
    request: Request,
    audio: UploadFile = File(...),
    session_id: str = Form(DEFAULT_SESSION_ID),
) -> dict:
    """Run C1 + C2 + STT + emotion graph + Qwen + Kokoro for one turn.

    C3 Stress and CBT predictors are neither loaded nor called by this route.
    """

    content_type = (audio.content_type or "").lower()
    if content_type and not (
        content_type.startswith("audio/") or content_type == "application/octet-stream"
    ):
        raise HTTPException(status_code=400, detail="Upload an audio recording")
    content = audio.file.read(settings.C2_MAX_UPLOAD_BYTES + 1)
    if len(content) > settings.C2_MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio exceeds the {settings.C2_MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )

    try:
        from c2 import decode_audio

        waveform = decode_audio(content)
        duration = waveform.size / 16_000
        if duration > settings.C2_MAX_AUDIO_SECONDS:
            raise ValueError(
                f"Audio is {duration:.1f}s; maximum is {settings.C2_MAX_AUDIO_SECONDS:.0f}s"
            )
        appraisal_analyzer, transcriber, synthesizer = get_voice_services()
        appraisal = appraisal_analyzer.predict(
            waveform,
            file_name=audio.filename or "voice_input",
        )
        transcription = transcriber.transcribe(waveform)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Voice analysis failed: {exc}") from exc

    session_id = patient_id(request)
    with _sensor_lock:
        sensor_state, sensor_context = _sensor_turn_data(session_id)
    multimodal_state, multimodal_context = _multimodal_turn_data(
        session_id,
        sensor_state,
        voice_probability=float(appraisal.voice_stress_score),
    )
    with _pipeline_lock:
        state = {
            "user_text": transcription.text,
            "history": _store.history(session_id, settings.REPLY_HISTORY_TURNS * 2),
            "previous_emotion": _store.previous_emotion(session_id),
            **sensor_state,
            **multimodal_state,
            "voice_stress": appraisal.voice_stress,
            "voice_stress_score": appraisal.voice_stress_score,
            "voice_appraisal_state": appraisal.dominant_state,
            "voice_appraisal_uncertain": appraisal.uncertain,
        }
        result = get_voice_workflow().invoke(state)
        try:
            reply_audio = synthesizer.synthesize(
                speakable(result["reply"], settings.SUPPORT_CONTACTS)
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Voice output failed: {exc}") from exc

    trace = build_trace(
        result,
        mode="voice",
        next_negative_min_confidence=settings.NEXT_NEGATIVE_MIN_CONFIDENCE,
    )
    turn_id = uuid.uuid4().hex
    trace["turn_id"] = turn_id
    _remember_turn_messages(session_id, transcription.text, result, trace)
    _remember_turn(
        turn_id,
        {
            "patient_id": session_id,
            "mode": "voice",
            "message": transcription.text,
            "voice_appraisal": appraisal.as_dict(),
            **sensor_context,
            **multimodal_context,
        },
    )
    with _voice_audio_lock:
        _voice_audio_outputs[turn_id] = reply_audio
        while len(_voice_audio_outputs) > 50:
            _voice_audio_outputs.pop(next(iter(_voice_audio_outputs)))

    return _patient_response({
        "turn_id": turn_id,
        "transcript": transcription.text,
        "transcription_language": transcription.language,
        "transcription_language_probability": transcription.language_probability,
        "duration_seconds": transcription.duration_seconds,
        "reply": result["reply"],
        "reply_audio_url": f"/api/voice/audio/{turn_id}",
        "route": result.get("reply_route"),
        "source": result.get("reply_source"),
        "current_emotion": result.get("current_emotion"),
        "next_emotion": result.get("next_emotion"),
        "next_emotion_confidence": result.get("next_emotion_confidence"),
        "next_intensity": result.get("next_intensity"),
        "next_intensity_probability": result.get("next_intensity_probability"),
        "voice_appraisal": appraisal.as_dict(),
        "questionnaire_stress_probability": result.get(
            "questionnaire_stress_probability"
        ),
        "multimodal_stress": result.get("multimodal_stress"),
        "multimodal_stress_probability": result.get(
            "multimodal_stress_probability"
        ),
        "multimodal_stress_weights": result.get("multimodal_stress_weights", {}),
        "trace": trace,
        **_deviation_payload(result),
    })


@app.get("/api/voice/audio/{turn_id}")
def voice_audio(turn_id: str, request: Request) -> Response:
    _authorized_turn(turn_id, current_user(request))
    with _voice_audio_lock:
        content = _voice_audio_outputs.get(turn_id)
    if content is None:
        raise HTTPException(status_code=404, detail="Voice reply expired")
    return Response(
        content=content,
        media_type="audio/wav",
        headers={"Cache-Control": "no-store"},
    )


_SPECIAL_TOKENS = {
    "[PAD]",
    "[CLS]",
    "[SEP]",
    "[UNK]",
    "<pad>",
    "<s>",
    "</s>",
    "<unk>",
}


def _display_token(token: str) -> str:
    """Make common WordPiece/SentencePiece markers readable in the browser."""

    if token.startswith("##"):
        return token[2:]
    return token.lstrip("▁Ġ")


def _serialize_token_attributions(items) -> list[dict[str, float | str]]:
    serialized = []
    for item in items:
        raw = str(item.token)
        if raw in _SPECIAL_TOKENS:
            continue
        token = _display_token(raw)
        if not token:
            continue
        serialized.append({"label": token, "score": float(item.score)})
    return serialized[:96]


def _serialize_mean_feature_attributions(groups) -> list[dict[str, float | str]]:
    """Average matching C1 feature attributions across the smoothed windows."""

    totals: dict[str, float] = {}
    count = 0
    for items in groups:
        count += 1
        for item in items:
            totals[str(item.feature)] = totals.get(str(item.feature), 0.0) + float(
                item.score
            )
    if count == 0:
        return []
    serialized = [
        {"label": feature, "score": score / count}
        for feature, score in totals.items()
    ]
    return sorted(serialized, key=lambda item: abs(float(item["score"])), reverse=True)


def _xai_coverage(mode: str, response: dict) -> list[dict[str, str]]:
    """State exactly what XAI covers, reuses, or intentionally cannot claim."""

    def item(component: str, status: str, detail: str) -> dict[str, str]:
        return {"component": component, "status": status, "detail": detail}

    shared = [
        item(
            "Personal check-in",
            "explained" if response.get("questionnaire") else "unavailable",
            "Each questionnaire answer is scored directly; no model is rerun."
            if response.get("questionnaire")
            else "No questionnaire result was saved for this turn.",
        ),
        item(
            "C1 wearable model",
            "explained" if response.get("sensor") else "unavailable",
            "Integrated Gradients explains the saved BVP/EDA prediction windows."
            if response.get("sensor")
            else "No calibrated wearable prediction was available for this turn.",
        ),
        item(
            "Wellbeing fusion",
            "explained" if response.get("combined") else "unavailable",
            "The displayed contribution is the actual 60% questionnaire + 40% wearable weighting."
            if response.get("combined")
            else "There were no available signals to combine for this turn.",
        ),
    ]
    if mode == "voice":
        return [
            item(
                "Speech recognition (STT)",
                "recorded",
                "The transcript is saved as the input. STT converts speech to words; it does not make a wellbeing prediction.",
            ),
            *shared,
            item(
                "C2 voice appraisal",
                "explained" if response.get("voice") else "unavailable",
                "Uses the arousal, dominance and valence contributions already saved during the voice turn."
                if response.get("voice")
                else "No voice appraisal was saved for this turn.",
            ),
            item(
                "C3 stress and thinking-pattern models",
                "not_used",
                "C3 belongs to the written-text path and is deliberately not run for a voice turn.",
            ),
            item(
                "Current emotion model",
                "explained" if response.get("current_emotion") else "unavailable",
                "Memory-bounded Integrated Gradients explains the transcript words.",
            ),
            item(
                "Next-turn intensity model",
                "explained" if response.get("next_emotion") else "unavailable",
                "Explains the differentiable RoBERTa branch of the forecast.",
            ),
            item(
                "Reply strategy and safety rules",
                "recorded",
                "The exact conditions that fired are shown in the saved History journey.",
            ),
            item(
                "Qwen reply wording",
                "boundary",
                "The route and prompt choice are recorded, but no exact token-level claim is made for free-form generated wording.",
            ),
            item(
                "Speech synthesis (TTS)",
                "output_only",
                "TTS reads the completed reply aloud and does not make a wellbeing prediction.",
            ),
        ]

    return [
        *shared,
        item(
            "C3 stress model",
            "explained" if response.get("stress") else "unavailable",
            "Memory-bounded Integrated Gradients explains influential message words.",
        ),
        item(
            "C3 thinking-pattern model",
            "explained" if response.get("cbt") else "unavailable",
            "Memory-bounded Integrated Gradients explains influential message words.",
        ),
        item(
            "Current emotion model",
            "explained" if response.get("current_emotion") else "unavailable",
            "Memory-bounded Integrated Gradients explains influential message words.",
        ),
        item(
            "Next-turn intensity model",
            "explained" if response.get("next_emotion") else "unavailable",
            "Explains the differentiable RoBERTa branch of the forecast.",
        ),
        item(
            "Reply strategy and safety rules",
            "recorded",
            "The exact conditions that fired are shown in the saved History journey.",
        ),
        item(
            "Qwen reply wording",
            "boundary",
            "The route and prompt choice are recorded, but no exact token-level claim is made for free-form generated wording.",
        ),
    ]


@app.post("/api/xai")
def explain_chat(payload: XAIRequest, request: Request) -> dict:
    """Compute classifier XAI on demand for one already-submitted message.

    Each section is isolated so, for example, a missing wearable prediction
    does not prevent the text explanations from being returned. Qwen itself
    is intentionally excluded: these attributions explain classifier targets,
    not free-form generated tokens.
    """

    if not payload.turn_id:
        raise HTTPException(400, "Select a saved patient turn to explain.")
    _authorized_turn(payload.turn_id, current_user(request))

    text = payload.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    response: dict = {
        "sensor": None,
        "questionnaire": None,
        "combined": None,
        "stress": None,
        "cbt": None,
        "voice": None,
        "current_emotion": None,
        "next_emotion": None,
        "errors": {},
        "note": (
            "Integrated Gradients shows influence on classifier targets; "
            "it does not explain the free-form Qwen reply."
        ),
    }

    turn_context = None
    if payload.turn_id:
        with _turn_context_lock:
            turn_context = _turn_contexts.get(payload.turn_id)
        if turn_context is None:
            raise HTTPException(status_code=404, detail="Explanation context expired")
        if turn_context["message"] != text:
            raise HTTPException(status_code=400, detail="turn_id does not match message")

    request_mode = turn_context.get("mode", "text") if turn_context else "text"
    if turn_context:
        questionnaire = turn_context.get("questionnaire")
        if questionnaire is not None:
            question_labels = {
                "manageable": "Demands feel manageable",
                "tense": "Feeling tense",
                "overwhelmed": "Feeling overwhelmed",
                "relaxed": "Able to relax",
                "worried": "Feeling worried",
                "in_control": "Feeling in control",
            }
            target_stressed = bool(questionnaire.is_stressed)
            response["questionnaire"] = {
                "target": "stressed" if target_stressed else "calm",
                "method": "Six-item momentary self-report, scored 0-1",
                "attributions": [
                    {
                        "label": question_labels[key],
                        "score": (
                            (score - 0.5) * 2
                            if target_stressed
                            else (0.5 - score) * 2
                        ),
                    }
                    for key, score in questionnaire.item_stress_scores.items()
                ],
            }

        fused = turn_context.get("multimodal_stress")
        if fused is not None and fused.available:
            target_stressed = bool(fused.is_stressed)
            source_labels = {
                "questionnaire": "Your check-in answers",
                "sensor": "Wearable reading",
                "voice": "Voice reading",
            }
            response["combined"] = {
                "target": "stressed" if target_stressed else "calm",
                "method": (
                    "Availability-normalized weighted late fusion "
                    "(questionnaire 60%, wearable 40%; voice kept separate)"
                ),
                "attributions": [
                    {
                        "label": source_labels[key],
                        "score": weight
                        * (
                            (fused.probabilities[key] - 0.5) * 2
                            if target_stressed
                            else (0.5 - fused.probabilities[key]) * 2
                        ),
                    }
                    for key, weight in fused.normalized_weights.items()
                ],
            }

    if request_mode == "voice" and turn_context:
        appraisal = turn_context.get("voice_appraisal")
        if appraisal:
            response["voice"] = {
                "target": (
                    "uncertain"
                    if appraisal.get("uncertain")
                    else appraisal.get("dominant_state", "unknown")
                ),
                "method": "C2 appraisal fuzzy-rule contribution",
                "member": "AudEERING ADV",
                "attributions": appraisal.get("contributions", []),
            }
            response["note"] = (
                "C2 contributions explain the appraisal-rule result; Integrated "
                "Gradients explains the transcript-based emotion targets. C3 is "
                "not part of the voice path, and Qwen text generation is not attributed."
            )

    with _pipeline_lock:
        components = (
            get_voice_components() if request_mode == "voice" else get_components()
        )

        if turn_context and turn_context["sensor_predictions"]:
            try:
                prediction_groups = [
                    turn_context["sensor_predictor"].explain(
                        prediction,
                        turn_context["sensor_calibration"],
                        n_steps=16,
                    )
                    for prediction in turn_context["sensor_predictions"]
                ]
                response["sensor"] = {
                    "target": (
                        "stressed"
                        if turn_context["sensor_smoothed_is_stressed"]
                        else "calm"
                    ),
                    "method": (
                        "Mean Integrated Gradients over "
                        f"{len(prediction_groups)} smoothed windows"
                    ),
                    "attributions": _serialize_mean_feature_attributions(
                        prediction_groups
                    ),
                }
            except Exception as exc:
                response["errors"]["sensor"] = str(exc)

        if request_mode == "text":
            try:
                prediction = components.stress_predictor.predict(text)
                response["stress"] = {
                    "target": prediction.label,
                    "method": "Ensemble-weighted Layer Integrated Gradients",
                    "member": "BERT 0.5 + DeBERTa-v3 0.5",
                    "attributions": _serialize_token_attributions(
                        components.stress_predictor.explain(
                            text,
                            is_stressed=prediction.is_stressed,
                            n_steps=8,
                        )
                    ),
                }
            except Exception as exc:
                response["errors"]["stress"] = str(exc)

            try:
                prediction = components.cbt_predictor.predict(text)
                response["cbt"] = {
                    "target": prediction.label,
                    "method": "Ensemble-weighted Layer Integrated Gradients",
                    "member": "BERT 0.1 + MentalBERT 0.4 + DeBERTa-v3 0.5",
                    "attributions": _serialize_token_attributions(
                        components.cbt_predictor.explain(
                            text,
                            has_distortion=prediction.has_distortion,
                            n_steps=8,
                        )
                    ),
                }
            except Exception as exc:
                response["errors"]["cbt"] = str(exc)

        try:
            emotion = components.emotion_chain.predict(text)
            current_items = components.emotion_chain.current_classifier.explain(
                text,
                target_label=emotion.current.label,
                n_steps=8,
            )
            next_items = components.emotion_chain.next_forecaster.explain(
                text,
                emotion.current,
                target_intensity=emotion.forecast.label,
                n_steps=8,
            )
            response["current_emotion"] = {
                "target": emotion.current.label,
                "method": "Layer Integrated Gradients",
                "attributions": _serialize_token_attributions(current_items),
            }
            # Only the neural half of the blend is differentiable in the
            # tokens, so the panel says so rather than implying the whole
            # forecaster is being explained.
            response["next_emotion"] = {
                "target": f"{emotion.forecast.label} intensity ({emotion.next.label})",
                "method": "Layer Integrated Gradients, frozen-RoBERTa branch only",
                "attributions": _serialize_token_attributions(next_items),
            }
        except Exception as exc:
            response["errors"]["emotion"] = str(exc)

    response["mode"] = request_mode
    response["coverage"] = _xai_coverage(request_mode, response)
    response["resource_note"] = (
        "Voice XAI reuses the saved C2 appraisal and does not rerun STT, TTS or "
        "Qwen. Only the shared current/next emotion models and an optional small "
        "sensor explainer perform extra bounded gradient work."
        if request_mode == "voice"
        else "Text XAI uses memory-bounded gradient batches for C3 and emotion "
        "models; Qwen generation is not rerun."
    )
    return response


# Portal access helpers. Legacy browser session ids are never used as authority.
def _claim_sensor(request: Request) -> None:
    global _sensor_patient_id
    target = patient_id(request)
    if _sensor_patient_id not in (None, target) and _c1_service is not None:
        raise HTTPException(409, "The wearable is in use for another patient. Disconnect that recording first.")
    _sensor_patient_id = target


def _authorized_turn(turn_id: str, user: dict) -> dict:
    with _turn_context_lock:
        context = _turn_contexts.get(turn_id)
    if not context or not accounts.can_access(user, context.get("patient_id", "")):
        raise HTTPException(404, "This turn is unavailable or its explanation has expired.")
    return context


def _patient_response(payload: dict) -> dict:
    return {key: payload[key] for key in (
        "turn_id", "reply", "support_contacts", "transcript", "reply_audio_url", "duration_seconds"
    ) if key in payload}


@app.get("/api/doctor/patients")
def doctor_patients(request: Request) -> list[dict]:
    result = []
    _store.purge_expired()
    for patient in accounts.patients(current_user(request)["id"]):
        messages = _store.transcript(patient["id"])
        result.append({**patient, "message_count": len(messages),
                       "latest_activity": messages[-1].get("created_at") if messages else None})
    return result


@app.get("/api/doctor/activity")
def doctor_activity(request: Request) -> list[dict]:
    _store.purge_expired()
    messages = _store.transcript(patient_id(request))
    with _turn_context_lock:
        available = set(_turn_contexts)
    for message in messages:
        turn_id = (message.get("trace") or {}).get("turn_id")
        message["explanation_available"] = bool(turn_id and turn_id in available)
    return messages


# Stop acquisition when its patient changes or revokes their care-team link.
def _release_patient_sensor(target: str) -> None:
    global _c1_service, _sensor_patient_id, _sensor_skipped
    with _sensor_lock:
        if _sensor_patient_id != target:
            return
        if _c1_service is not None:
            _c1_service.stop()
        _c1_service = None
        _sensor_patient_id = None
        _sensor_skipped = False


app.state.release_patient_sensor = _release_patient_sensor


@app.middleware("http")
async def private_responses(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response
