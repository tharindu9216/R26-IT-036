"""Loads ``config.yaml`` once and resolves every path in it.

Every entry point (``app.py``, ``cli.py``, ``supportive_app.py``, ``api.py``)
and the C1/C3 modules import their model paths and reply-generation knobs
from here instead of hardcoding them, so pointing any supportive module
(sensor, stress header, CBT header, current emotion, next intensity, reply
generation) at a different checkpoint is a one-line edit to ``config.yaml``,
not a code change.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from emotion_chain.fusion import InfluenceWeights
from emotion_chain.support_contacts import DEFAULT_SUPPORT_CONTACTS, SupportContact

BACKEND_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BACKEND_DIR / "config.yaml"
PROFILES_DIR = BACKEND_DIR / "profiles"

with CONFIG_PATH.open("r", encoding="utf-8") as _file:
    _raw: dict = yaml.safe_load(_file)


def _merge(base: dict, overlay: dict) -> dict:
    """Overlay one config section at a time, so a profile can stay tiny."""

    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge(current, value)
        else:
            merged[key] = value
    return merged


# ``config.yaml`` is the base and describes the single-machine deployment this
# project started as. A profile in ``profiles/`` overrides only the keys that
# differ on another machine -- devices, quantization, support-node URLs -- so
# the tuned values (fusion weights, thresholds, support contacts) live in one
# file and cannot drift between deployments.
#
# Pick one per machine, either way round:
#   config.yaml:  profile: split-main          <- set once, survives reboots
#   PowerShell:   $env:AGENT_PROFILE = "split-main"
#   bash:         export AGENT_PROFILE=split-main
#
# The environment variable wins when both are set, so a machine can keep its
# usual profile in the file and still be started with a different one for a
# single run. Either may be `none` (or empty) to force plain config.yaml even
# when the file names a profile.
_env_profile = os.getenv("AGENT_PROFILE")
_profile_name = _env_profile if _env_profile is not None else _raw.get("profile")
PROFILE_SOURCE = "AGENT_PROFILE" if _env_profile is not None else "config.yaml"
PROFILE = str(_profile_name or "").strip() or None
if PROFILE and PROFILE.lower() == "none":
    PROFILE = None
if PROFILE:
    _profile_path = Path(PROFILE)
    if not _profile_path.is_absolute():
        _profile_path = PROFILES_DIR / f"{PROFILE}.yaml"
    if not _profile_path.is_file():
        available = ", ".join(sorted(item.stem for item in PROFILES_DIR.glob("*.yaml")))
        raise FileNotFoundError(
            f"Profile {PROFILE!r} (from {PROFILE_SOURCE}) not found at "
            f"{_profile_path} (available: {available or 'none'})"
        )
    with _profile_path.open("r", encoding="utf-8") as _file:
        _raw = _merge(_raw, yaml.safe_load(_file) or {})
    # A profile must not select another profile -- one hop only, so the
    # active deployment is always readable from these two places alone.
    _raw.pop("profile", None)

MODEL_ROOT = (BACKEND_DIR / _raw["model_root"]).resolve()

# -- Runtime device placement --
C3_DEVICE = str(_raw.get("runtime_devices", {}).get("c3", "cpu"))
EMOTION_DEVICE = str(_raw.get("runtime_devices", {}).get("emotion", "cpu"))
QWEN_DEVICE = str(_raw.get("runtime_devices", {}).get("qwen", "auto"))
C2_DEVICE = str(_raw.get("runtime_devices", {}).get("c2", "cpu"))
STT_DEVICE = str(_raw.get("runtime_devices", {}).get("stt", "cpu"))
TTS_DEVICE = str(_raw.get("runtime_devices", {}).get("tts", "cpu"))

# -- Support-node offloading --
# Each entry is either null/empty (load the model in this process, the
# single-machine default) or the base URL of a ``../node`` support server on
# the LAN that hosts it instead, e.g. ``http://192.168.1.42:8010``. Only the
# four library-only services can be offloaded; C1, C3 and the emotion chain
# always run in-process. See ../DEPLOYMENT.md.
_remote_raw: dict = _raw.get("remote_services") or {}


def _remote_url(name: str) -> str | None:
    value = _remote_raw.get(name)
    if value is None:
        return None
    text = str(value).strip().rstrip("/")
    return text or None


REMOTE_REPLY_URL = _remote_url("reply")
REMOTE_STT_URL = _remote_url("stt")
REMOTE_TTS_URL = _remote_url("tts")
REMOTE_C2_URL = _remote_url("c2")
REMOTE_TIMEOUT = float(_remote_raw.get("timeout_seconds", 180))
_remote_token = _remote_raw.get("auth_token")
REMOTE_AUTH_TOKEN = str(_remote_token).strip() if _remote_token else None

# -- HTTP API --
# Extra browser origins allowed to call api.py. Add the LAN address the React
# dev server is served from (e.g. http://192.168.1.30:5173) when the UI is
# opened from another device.
API_CORS_ORIGINS: tuple[str, ...] = tuple(
    str(origin).strip()
    for origin in (
        _raw.get("api", {}).get(
            "cors_origins",
            [
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://localhost:4173",
                "http://127.0.0.1:4173",
            ],
        )
        or ()
    )
    if str(origin).strip()
)


def _model_path(*parts: str) -> Path:
    return MODEL_ROOT.joinpath(*parts)


# -- C1 sensor --
C1_ARTIFACT_DIR = _model_path(_raw["c1_sensor"]["artifact_dir"])
C1_DEFAULT_PORT = str(_raw["c1_sensor"]["default_port"])
C1_SMOOTHING_WINDOW = int(_raw["c1_sensor"].get("smoothing_window", 3))
C1_SMOOTHING_REQUIRED = int(_raw["c1_sensor"].get("smoothing_required", 2))

# -- C2 voice appraisal/stress --
C2_MODEL_DIR = _model_path(_raw["c2_voice"]["audeering_model_dir"])
C2_APPRAISAL_ARTIFACT = _model_path(_raw["c2_voice"]["appraisal_artifact"])
C2_MAX_UPLOAD_BYTES = int(_raw["c2_voice"].get("max_upload_mb", 25)) * 1024 * 1024
C2_MAX_AUDIO_SECONDS = float(_raw["c2_voice"].get("max_audio_seconds", 180))

# -- Offline voice I/O --
STT_MODEL_DIR = _model_path(_raw["speech_to_text"]["model_dir"])
STT_COMPUTE_TYPE = str(_raw["speech_to_text"].get("compute_type", "int8"))
STT_TASK = str(_raw["speech_to_text"].get("task", "transcribe"))
STT_LANGUAGE = _raw["speech_to_text"].get("language")
TTS_MODEL_DIR = _model_path(_raw["text_to_speech"]["model_dir"])
TTS_VOICE = str(_raw["text_to_speech"].get("voice", "af_heart"))
TTS_LANGUAGE_CODE = str(_raw["text_to_speech"].get("language_code", "a"))
TTS_SPEED = float(_raw["text_to_speech"].get("speed", 1.0))

# -- C3 stress / CBT (per-member weights stay in their own configs) --
C3_STRESS_DIR = _model_path(_raw["c3_stress"]["artifact_dir"])
C3_STRESS_BERTOPIC_DIR = _model_path(_raw["c3_stress_bertopic"]["artifact_dir"])
C3_CBT_DIR = _model_path(_raw["c3_cbt"]["artifact_dir"])
C3_BASE_MODELS_DIR = _model_path(_raw["c3_base_models"]["artifact_dir"])

# -- C3 Stress + CBT decision-level fusion --
FUSION_STRESS_WEIGHT = float(_raw["decision_fusion"]["stress_weight"])
FUSION_CBT_WEIGHT = float(_raw["decision_fusion"]["cbt_weight"])
FUSION_THRESHOLD = float(_raw["decision_fusion"]["threshold"])
FUSION_METHOD = str(_raw["decision_fusion"].get("method", "weighted_average"))
FUSION_STRESS_THRESHOLD = float(
    _raw["decision_fusion"].get("stress_threshold", 0.5)
)
FUSION_CBT_THRESHOLD = float(_raw["decision_fusion"].get("cbt_threshold", 0.37))

# -- How much each stress modality may sway the wording of a reply --
# Not a gate and not a threshold: the weight multiplies the source's own
# confidence and the product is stated in the prompt, so a low-trust source
# still reaches the model, just as weaker evidence. 0.0 hides it entirely.
_influence_raw = _raw.get("signal_influence", {})
SIGNAL_INFLUENCE = InfluenceWeights(
    text=float(_influence_raw.get("text_weight", 1.0)),
    sensor=float(_influence_raw.get("sensor_weight", 1.0)),
    voice=float(_influence_raw.get("voice_weight", 1.0)),
)

_multimodal_raw = _raw.get("multimodal_stress", {}) or {}
MULTIMODAL_QUESTIONNAIRE_WEIGHT = float(
    _multimodal_raw.get("questionnaire_weight", 0.6)
)
MULTIMODAL_SENSOR_WEIGHT = float(_multimodal_raw.get("sensor_weight", 0.4))
MULTIMODAL_VOICE_WEIGHT = float(_multimodal_raw.get("voice_weight", 0.0))
MULTIMODAL_STRESS_THRESHOLD = float(_multimodal_raw.get("threshold", 0.5))

# -- Reply routing --
# Confidence floor a negative forecast must clear before it counts as a routing
# reason. It still filters: the forecaster's decision threshold is 0.595, so a
# "low" call made just under it reports a confidence below 0.5 and is dropped,
# leaving the reason to fire only when the turn is genuinely expected to
# intensify. What it cannot do any more is change a route -- the forecaster
# keeps the current emotion's family, so it only names a negative state when
# the current emotion is already negative, which routes on its own. See
# backend/README.md.
NEXT_NEGATIVE_MIN_CONFIDENCE = float(
    _raw.get("routing", {}).get("next_negative_min_confidence", 0.5)
)

# -- Human contacts offered on crisis / extremely negative turns --
# An explicit empty list disables them; an absent key keeps the module default.
_support_contacts_raw = _raw.get("support_contacts")
SUPPORT_CONTACTS: tuple[SupportContact, ...] = (
    DEFAULT_SUPPORT_CONTACTS
    if _support_contacts_raw is None
    else tuple(
        SupportContact(
            name=str(entry["name"]),
            phone=str(entry["phone"]),
            note=str(entry.get("note", "")),
            spoken_name=str(entry.get("spoken_name", "")),
        )
        for entry in _support_contacts_raw
    )
)

# -- Current emotion / next intensity --
# The forecaster is one directory rather than a checkpoint plus a metadata
# file: its blend weights and threshold are part of the exported artifact, so
# there is nothing per-deployment left to select here.
CURRENT_EMOTION_DIR = _model_path(_raw["current_emotion"]["artifact_dir"])
NEXT_INTENSITY_DIR = _model_path(_raw["next_intensity"]["artifact_dir"])

# -- Reply generation (Qwen base + ESConv adapter) --
QWEN_MODEL_DIR = _model_path(_raw["reply_generation"]["qwen_base_dir"])
QWEN_ADAPTER_DIR = _model_path(_raw["reply_generation"]["qwen_adapter_dir"])
REPLY_USE_ADAPTER = bool(_raw["reply_generation"].get("use_adapter", True))
# ``none`` keeps the exported bf16/fp16 weights (~8 GB VRAM, needs a 10 GB+
# card); ``nf4`` loads the base model 4-bit through bitsandbytes (~3 GB) so it
# fits a 4 GB laptop GPU; ``int8`` sits between them (~5 GB).
REPLY_QUANTIZATION = str(_raw["reply_generation"].get("quantization", "none"))
REPLY_MAX_NEW_TOKENS = int(_raw["reply_generation"]["max_new_tokens"])
REPLY_TEMPERATURE = float(_raw["reply_generation"]["temperature"])
REPLY_TOP_P = float(_raw["reply_generation"]["top_p"])
REPLY_HISTORY_TURNS = int(_raw["reply_generation"].get("history_turns", 6))

# -- Chat history (per-session SQLite store used by api.py) --
# Read with .get() throughout so a profile written before this section existed
# still loads.
_chat_history = _raw.get("chat_history", {}) or {}
CHAT_DB_PATH = BACKEND_DIR / str(_chat_history.get("db_file", "chat_sessions.db"))
CHAT_HISTORY_TTL_SECONDS = float(_chat_history.get("ttl_seconds", 86_400))
CHAT_HISTORY_MAX_MESSAGES = int(_chat_history.get("max_messages", 200))
