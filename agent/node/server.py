"""FastAPI support node: hosts the app's heavy models for a second machine.

Start it with ``run.ps1`` (or ``uvicorn server:app --host 0.0.0.0 --port 8010``)
and point the backend's ``remote_services`` at ``http://<this-ip>:8010``.

The node is a pure model server. It has no conversation state, no routing and
no crisis handling: the backend builds the prompt, decides the route and
appends escalation contacts, then asks only for the completion. That is what
keeps a 4 GB laptop paired with this node behaving identically to a single
12 GB machine running everything in one process.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

import models

LOGGER = logging.getLogger("node")
logging.basicConfig(level=logging.INFO)

NODE_DIR = Path(__file__).resolve().parent

_CONFIG_NAME = os.getenv("NODE_CONFIG", "config.yaml")
CONFIG_PATH = Path(_CONFIG_NAME)
if not CONFIG_PATH.is_absolute():
    CONFIG_PATH = NODE_DIR / CONFIG_PATH
CONFIG_PATH = CONFIG_PATH.resolve()
if not CONFIG_PATH.is_file():
    raise FileNotFoundError(f"Node config not found: {CONFIG_PATH}")

CONFIG: dict = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
MODEL_ROOT = (NODE_DIR / CONFIG["model_root"]).resolve()
ENABLED: dict = CONFIG.get("services", {})
AUTH_TOKEN = CONFIG.get("auth_token") or None
HOST = str(CONFIG.get("host", "0.0.0.0"))
PORT = int(CONFIG.get("port", 8010))


def _model_path(*parts: str) -> Path:
    return MODEL_ROOT.joinpath(*parts)


def _enabled(name: str) -> bool:
    return bool(ENABLED.get(name, False))


# --------------------------------------------------------------------------
# Service construction. Each enabled service is built (and its artifacts
# validated) at import time; the weights themselves stay lazy unless
# `preload: true`, so a missing file fails at startup, not mid-conversation.
# --------------------------------------------------------------------------

_reply_model: models.ReplyModel | None = None
_transcriber: models.Transcriber | None = None
_synthesizer: models.Synthesizer | None = None
_appraiser: models.AppraisalAnalyzer | None = None

if _enabled("reply"):
    _reply_config = CONFIG["reply"]
    _reply_model = models.ReplyModel(
        _model_path(_reply_config["qwen_base_dir"]),
        _model_path(_reply_config["qwen_adapter_dir"]),
        device=str(_reply_config.get("device", "auto")),
        quantization=str(_reply_config.get("quantization", "nf4")),
    )

if _enabled("stt"):
    _stt_config = CONFIG["speech_to_text"]
    _transcriber = models.Transcriber(
        _model_path(_stt_config["model_dir"]),
        device=str(_stt_config.get("device", "cpu")),
        compute_type=str(_stt_config.get("compute_type", "int8")),
        task=str(_stt_config.get("task", "transcribe")),
        language=_stt_config.get("language"),
    )

if _enabled("tts"):
    _tts_config = CONFIG["text_to_speech"]
    _synthesizer = models.Synthesizer(
        _model_path(_tts_config["model_dir"]),
        voice=str(_tts_config.get("voice", "af_heart")),
        language_code=str(_tts_config.get("language_code", "a")),
        device=str(_tts_config.get("device", "cpu")),
        speed=float(_tts_config.get("speed", 1.0)),
    )

if _enabled("c2"):
    _c2_config = CONFIG["c2"]
    _package_dir = _c2_config.get("appraisal_package_dir")
    _appraiser = models.AppraisalAnalyzer(
        _model_path(_c2_config["audeering_model_dir"]),
        _model_path(_c2_config["appraisal_artifact"]),
        device=str(_c2_config.get("device", "cpu")),
        package_dir=(NODE_DIR / _package_dir).resolve() if _package_dir else None,
    )

if not any((_reply_model, _transcriber, _synthesizer, _appraiser)):
    raise RuntimeError(
        f"No services enabled in {CONFIG_PATH}; the node would answer nothing"
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    for service, location in _deployment().items():
        LOGGER.info("hosting %-6s -> %s", service, location)
    if CONFIG.get("preload", False):
        for name, service in _services():
            LOGGER.info("preloading %s...", name)
            service.load()
        LOGGER.info("preload complete")
    else:
        LOGGER.info("preload disabled; models load on first request")
    yield


app = FastAPI(title="Supportive agent support node", lifespan=lifespan)


def _authorize(token: str | None) -> None:
    if AUTH_TOKEN and token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Auth-Token")


def _require(service, name: str):
    if service is None:
        raise HTTPException(
            status_code=503,
            detail=f"This node does not host '{name}' (see services in {CONFIG_PATH.name})",
        )
    return service


def _services() -> list[tuple[str, object]]:
    return [
        (name, service)
        for name, service in (
            ("reply", _reply_model),
            ("stt", _transcriber),
            ("tts", _synthesizer),
            ("c2", _appraiser),
        )
        if service is not None
    ]


def _vram() -> dict | None:
    """Current and peak GPU memory, so "will it fit in 4 GB?" is answerable.

    ``peak_mb`` is the high-water mark since the node started -- the number
    that matters, since it is reached mid-generation with a full history and
    is what an out-of-memory error would hit.
    """

    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return {
            "device": torch.cuda.get_device_name(0),
            "allocated_mb": round(torch.cuda.memory_allocated() / 1048576),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1048576),
            "peak_mb": round(torch.cuda.max_memory_reserved() / 1048576),
            "gpu_used_mb": round((total - free) / 1048576),
            "gpu_total_mb": round(total / 1048576),
        }
    except Exception:
        return None


def _deployment() -> dict[str, str]:
    entries: dict[str, str] = {}
    if _reply_model is not None:
        entries["reply"] = f"{_reply_model.device} / {_reply_model.quantization}"
    if _transcriber is not None:
        entries["stt"] = f"{_transcriber.device} / {_transcriber.compute_type}"
    if _synthesizer is not None:
        entries["tts"] = f"{_synthesizer.device} / {_synthesizer.voice}"
    if _appraiser is not None:
        entries["c2"] = _appraiser.device
    return entries


@app.get("/health")
def health(x_auth_token: str | None = Header(default=None)) -> dict:
    _authorize(x_auth_token)
    return {
        "ok": True,
        "config": CONFIG_PATH.name,
        "model_root": str(MODEL_ROOT),
        "hosts": _deployment(),
        "loaded": {name: service.is_loaded for name, service in _services()},
        "vram": _vram(),
    }


@app.post("/v1/admin/reset-peak")
def reset_peak(x_auth_token: str | None = Header(default=None)) -> dict:
    """Drop cached blocks and restart the peak counter.

    Sizing a card means asking "what does *this* setting cost", but the
    allocator's high-water mark only ever grows, so a second measurement in
    the same process reports the first one's peak. Call this between runs
    when deciding whether a GPU has room for a given history/quantization
    combination.
    """

    _authorize(x_auth_token)
    try:
        import torch

        if not torch.cuda.is_available():
            return {"ok": False, "detail": "no CUDA device"}
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"ok": True, "vram": _vram()}


class ChatMessage(BaseModel):
    role: str
    content: str


class ReplyRequest(BaseModel):
    """A prompt the backend has already assembled, plus its sampling knobs.

    ``use_adapter`` is the backend's routing decision made concrete: True
    means the ESConv LoRA answers this turn, False means the base model does.
    The node never decides this itself.
    """

    messages: list[ChatMessage] = Field(min_length=1)
    max_new_tokens: int = Field(default=160, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.8, gt=0.0, le=1.0)
    use_adapter: bool = False


@app.post("/v1/reply/generate")
def generate_reply(
    payload: ReplyRequest, x_auth_token: str | None = Header(default=None)
) -> dict:
    _authorize(x_auth_token)
    service: models.ReplyModel = _require(_reply_model, "reply")
    try:
        text = service.generate(
            [message.model_dump() for message in payload.messages],
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
            top_p=payload.top_p,
            use_adapter=payload.use_adapter,
        )
    except Exception as exc:
        LOGGER.exception("reply generation failed")
        raise HTTPException(status_code=500, detail=f"Reply generation failed: {exc}") from exc
    return {"text": text, "use_adapter": payload.use_adapter}


@app.post("/v1/stt/transcribe")
async def transcribe(
    request: Request,
    x_auth_token: str | None = Header(default=None),
) -> dict:
    """Transcribe raw little-endian float32 mono PCM at 16 kHz.

    The backend already decoded the browser recording to that form for C2, so
    sending it verbatim avoids a re-encode and keeps ffmpeg off this machine.
    """

    _authorize(x_auth_token)
    service: models.Transcriber = _require(_transcriber, "stt")
    try:
        waveform = models.pcm_to_waveform(await request.body())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        return service.transcribe(waveform)
    except Exception as exc:
        LOGGER.exception("transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1)
    voice: str | None = None
    speed: float | None = Field(default=None, gt=0.0, le=3.0)


@app.post("/v1/tts/synthesize")
def synthesize(
    payload: SynthesisRequest, x_auth_token: str | None = Header(default=None)
) -> Response:
    _authorize(x_auth_token)
    service: models.Synthesizer = _require(_synthesizer, "tts")
    try:
        audio = service.synthesize(
            payload.text, voice=payload.voice, speed=payload.speed
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("synthesis failed")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {exc}") from exc
    return Response(content=audio, media_type="audio/wav")


@app.post("/v1/c2/appraise")
async def appraise(
    request: Request,
    file_name: str = Query(default="voice_input"),
    x_auth_token: str | None = Header(default=None),
) -> dict:
    _authorize(x_auth_token)
    service: models.AppraisalAnalyzer = _require(_appraiser, "c2")
    try:
        waveform = models.pcm_to_waveform(await request.body())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        return service.predict(waveform, file_name=file_name)
    except Exception as exc:
        LOGGER.exception("appraisal failed")
        raise HTTPException(status_code=500, detail=f"C2 appraisal failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
