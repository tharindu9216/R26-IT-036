"""One place that decides whether each heavy service is local or remote.

``api.py``, ``supportive_app.py``, ``app.py`` and ``cli.py`` used to repeat the
same eight-argument constructor calls; they now ask for a service here and get
either the in-process model or a support-node client, chosen by
``remote_services`` in the active config file. Nothing downstream changes:
the returned objects expose the same methods and return the same types, so the
LangGraph nodes, the XAI endpoint and the Streamlit pages are identical on a
single 12 GB machine and on a 4 GB laptop paired with a support node.

C1, C3 and the emotion chain are deliberately local-only -- see
``remote_clients`` for why.
"""

from __future__ import annotations

import logging

import settings
from emotion_chain.pipeline import EmotionChain
from emotion_chain.reply_generator import QwenReplyGenerator

LOGGER = logging.getLogger(__name__)


def build_emotion_chain(device: str | None = None) -> EmotionChain:
    """RoBERTa current emotion + the next-intensity hybrid. Always in-process.

    One RoBERTa serves both: the forecaster's fusion branch embeds with the
    classifier's own encoder, so the chain loads a single set of weights.
    """

    return EmotionChain(
        settings.CURRENT_EMOTION_DIR,
        settings.NEXT_INTENSITY_DIR,
        device=device or settings.EMOTION_DEVICE,
        next_negative_min_confidence=settings.NEXT_NEGATIVE_MIN_CONFIDENCE,
    )


def build_reply_generator(device: str | None = None) -> QwenReplyGenerator:
    """Qwen + the ESConv LoRA, loaded here or delegated to a support node.

    The generator keeps every routing/escalation decision locally either way;
    ``remote_url`` only moves the token generation. Weights stay lazy, so the
    first ``generate()`` call pays the load cost on whichever machine holds it.
    """

    return QwenReplyGenerator(
        settings.QWEN_MODEL_DIR,
        settings.QWEN_ADAPTER_DIR,
        device=device or settings.QWEN_DEVICE,
        use_adapter=settings.REPLY_USE_ADAPTER,
        max_new_tokens=settings.REPLY_MAX_NEW_TOKENS,
        temperature=settings.REPLY_TEMPERATURE,
        top_p=settings.REPLY_TOP_P,
        history_turns=settings.REPLY_HISTORY_TURNS,
        quantization=settings.REPLY_QUANTIZATION,
        influence_weights=settings.SIGNAL_INFLUENCE,
        support_contacts=settings.SUPPORT_CONTACTS,
        remote_url=settings.REMOTE_REPLY_URL,
        remote_timeout=settings.REMOTE_TIMEOUT,
        remote_auth_token=settings.REMOTE_AUTH_TOKEN,
    )


def build_stress_predictor():
    """C3 text-stress ensemble (BERT + DeBERTa-v3), loaded eagerly as before."""

    from c3.stress.stress_model import StressPredictor

    predictor = StressPredictor(device=settings.C3_DEVICE)
    predictor.load()
    return predictor


def build_cbt_predictor():
    """C3 cognitive-distortion ensemble (BERT + MentalBERT + DeBERTa-v3)."""

    from c3.CBT.cbt_model import CBTPredictor

    predictor = CBTPredictor(device=settings.C3_DEVICE)
    predictor.load()
    return predictor


def build_voice_appraisal_analyzer():
    if settings.REMOTE_C2_URL:
        from remote_clients import RemoteVoiceAppraisalAnalyzer

        return RemoteVoiceAppraisalAnalyzer(
            settings.REMOTE_C2_URL,
            timeout=settings.REMOTE_TIMEOUT,
            auth_token=settings.REMOTE_AUTH_TOKEN,
        )

    from c2 import VoiceAppraisalAnalyzer

    return VoiceAppraisalAnalyzer(
        settings.C2_MODEL_DIR,
        settings.C2_APPRAISAL_ARTIFACT,
        device=settings.C2_DEVICE,
    )


def build_transcriber():
    if settings.REMOTE_STT_URL:
        from remote_clients import RemoteVoiceTranscriber

        return RemoteVoiceTranscriber(
            settings.REMOTE_STT_URL,
            timeout=settings.REMOTE_TIMEOUT,
            auth_token=settings.REMOTE_AUTH_TOKEN,
        )

    from c2 import VoiceTranscriber

    return VoiceTranscriber(
        settings.STT_MODEL_DIR,
        device=settings.STT_DEVICE,
        compute_type=settings.STT_COMPUTE_TYPE,
        task=settings.STT_TASK,
        language=settings.STT_LANGUAGE,
    )


def build_synthesizer():
    if settings.REMOTE_TTS_URL:
        from remote_clients import RemoteKokoroSpeechSynthesizer

        return RemoteKokoroSpeechSynthesizer(
            settings.REMOTE_TTS_URL,
            voice=settings.TTS_VOICE,
            speed=settings.TTS_SPEED,
            timeout=settings.REMOTE_TIMEOUT,
            auth_token=settings.REMOTE_AUTH_TOKEN,
        )

    from c2 import KokoroSpeechSynthesizer

    return KokoroSpeechSynthesizer(
        settings.TTS_MODEL_DIR,
        voice=settings.TTS_VOICE,
        language_code=settings.TTS_LANGUAGE_CODE,
        device=settings.TTS_DEVICE,
        speed=settings.TTS_SPEED,
    )


def build_voice_services():
    """(C2 appraisal, Faster-Whisper, Kokoro) -- each local or remote."""

    return (
        build_voice_appraisal_analyzer(),
        build_transcriber(),
        build_synthesizer(),
    )


def deployment_summary() -> dict[str, str]:
    """Where every offloadable service runs, for /api/health and startup logs."""

    return {
        "profile": (
            f"{settings.PROFILE} (from {settings.PROFILE_SOURCE})"
            if settings.PROFILE
            else "none (config.yaml, all local)"
        ),
        "reply": settings.REMOTE_REPLY_URL
        or f"local ({settings.QWEN_DEVICE}, {settings.REPLY_QUANTIZATION})",
        "stt": settings.REMOTE_STT_URL or f"local ({settings.STT_DEVICE})",
        "tts": settings.REMOTE_TTS_URL or f"local ({settings.TTS_DEVICE})",
        "c2": settings.REMOTE_C2_URL or f"local ({settings.C2_DEVICE})",
        "c3": f"local ({settings.C3_DEVICE})",
        "emotion": f"local ({settings.EMOTION_DEVICE})",
    }


def log_deployment_summary() -> None:
    for service, location in deployment_summary().items():
        LOGGER.info("service %-8s -> %s", service, location)
