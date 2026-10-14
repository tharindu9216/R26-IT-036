"""Speech in and speech out for the C4 demo.

Both directions run on the CPU. That is a deliberate constraint rather than a
default: `config.py` measures the GPU budget to ~0.1 GB and low-VRAM mode leaves
roughly 0.8 GB of margin on a 4 GB card, so speech is only affordable if it does
not touch the GPU at all.

Nothing in `c4_pipeline/pipeline.py` knows this module exists. Speech is a
transport for the same strings the text chat already exchanges -- a transcript
goes in where `st.chat_input` would have put typed text, and the reply comes out
of `trace["supportive_response"]` -- so the pipeline stays modality-agnostic and
`smoke_test.py` is unaffected.

Both classes follow the convention the model wrappers in this package already
use: construction never raises, `available` says whether the real backend loaded,
and `load_error` carries the reason when it did not. A demo that loses its voice
should degrade in the sidebar, not crash mid-conversation.
"""

from __future__ import annotations

import io
import os
import platform
import re
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from config import (
    KOKORO_DIR,
    KOKORO_INTRA_OP_THREADS,
    KOKORO_MODEL_PREFERENCE,
    KOKORO_VOICES_FILE,
    STT_BEAM_SIZE,
    STT_COMPUTE_TYPE,
    STT_DEVICE,
    STT_LANGUAGE,
    STT_LOW_CONFIDENCE_LOGPROB,
    STT_MODEL,
    TTS_LANGUAGE,
    TTS_MAX_CHARS_PER_CHUNK,
    TTS_SAMPLE_WIDTH,
    TTS_SPEED,
    TTS_VOICE,
)

# Backend identifiers surfaced in the sidebar and stored on the trace.
STT_FASTER_WHISPER = "faster_whisper"
TTS_KOKORO = "kokoro"
TTS_SAPI = "windows_sapi"
TTS_NONE = "unavailable"

# Whisper checkpoints live beside the other vendored models, so an offline
# machine has everything under one directory.
WHISPER_DIR = KOKORO_DIR.parent / "faster-whisper"


def _vendored_or_hub(model_size: str) -> str:
    """Prefer `models/faster-whisper/<size>/`, else let faster-whisper download.

    Same resolution config.py uses for the transformers models, for the same
    reason: `vendor_models.py --voice` writes a plain CTranslate2 directory that
    WhisperModel accepts as a path, which is what makes an offline second machine
    possible. A size name is returned unchanged on a machine that never vendored.
    """
    local = WHISPER_DIR / model_size
    if (local / "model.bin").exists():
        return str(local)
    return model_size


# ===================================================================== speech in
@dataclass
class Transcript:
    """One transcribed recording.

    `avg_logprob` is Whisper's own mean token log-probability, duration-weighted
    across segments. It is kept because the demo submits transcripts without a
    confirmation step: the number is what lets the UI mark a turn as doubtful and
    lets the write-up report voice and text turns separately rather than assuming
    they are equivalent inputs.
    """

    text: str
    avg_logprob: float = 0.0
    duration: float = 0.0
    language: str = ""
    backend: str = STT_FASTER_WHISPER
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.text.strip()) and not self.error

    @property
    def low_confidence(self) -> bool:
        return self.ok and self.avg_logprob < STT_LOW_CONFIDENCE_LOGPROB


class SpeechTranscriber:
    """faster-whisper on the CPU, int8.

    Audio is decoded from an in-memory buffer and never written to disk. These
    are recordings of someone describing their distress; a temp WAV of one is not
    something a research prototype should leave behind in the temp directory. It
    is also why whisper.cpp was not used here -- its Windows path is a subprocess
    over a file, which costs that property for about 0.2 s.
    """

    def __init__(self, model_size: str = STT_MODEL) -> None:
        self.model_size = model_size
        self.model = None
        self.load_error: Optional[str] = None
        self.device = STT_DEVICE
        self.compute_type = STT_COMPUTE_TYPE

    @property
    def available(self) -> bool:
        return self.model is not None

    def load(self) -> "SpeechTranscriber":
        try:
            from faster_whisper import WhisperModel
        except Exception as error:  # noqa: BLE001 - absence is a supported state
            self.load_error = (
                f"faster-whisper is not installed ({type(error).__name__}: {error}). "
                "pip install faster-whisper"
            )
            return self
        try:
            self.model = WhisperModel(
                _vendored_or_hub(self.model_size),
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(WHISPER_DIR),
            )
        except Exception as error:  # noqa: BLE001
            self.load_error = f"{type(error).__name__}: {error}"
            self.model = None
        return self

    def transcribe(self, audio_bytes: bytes) -> Transcript:
        if not self.available:
            return Transcript(text="", error=self.load_error or "transcriber not loaded")
        if not audio_bytes:
            return Transcript(text="", error="empty recording")

        try:
            # faster-whisper decodes and resamples file-like input itself (PyAV),
            # so the recording stays a buffer from the browser to the model.
            segments, info = self.model.transcribe(
                io.BytesIO(audio_bytes),
                beam_size=STT_BEAM_SIZE,
                language=STT_LANGUAGE,
                vad_filter=True,
            )
            # `segments` is a generator; consume it before reading anything off it.
            collected = list(segments)
        except Exception as error:  # noqa: BLE001 - a bad clip must not kill the chat
            return Transcript(text="", error=f"{type(error).__name__}: {error}")

        text = re.sub(r"\s+", " ", " ".join(s.text.strip() for s in collected)).strip()

        # Duration-weighted, so one clipped half-second segment cannot drag down
        # the confidence of an otherwise clean sentence.
        total = sum(max(s.end - s.start, 0.0) for s in collected)
        if total > 0:
            avg_logprob = sum(
                s.avg_logprob * max(s.end - s.start, 0.0) for s in collected
            ) / total
        else:
            avg_logprob = 0.0

        return Transcript(
            text=text,
            avg_logprob=float(avg_logprob),
            duration=float(getattr(info, "duration", total) or total),
            language=str(getattr(info, "language", STT_LANGUAGE) or STT_LANGUAGE),
        )


# ==================================================================== speech out
def _split_for_synthesis(text: str, limit: int = TTS_MAX_CHARS_PER_CHUNK) -> List[str]:
    """Break text on sentence boundaries so no chunk exceeds `limit`.

    Kokoro degrades on long inputs. Replies are capped at four sentences upstream
    so this is usually a no-op; the fixed crisis text is the turn that can reach
    the limit, and that is precisely the turn that must not come out truncated.
    """
    text = (text or "").strip()
    if len(text) <= limit:
        return [text] if text else []

    chunks: List[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        # A single sentence over the limit is hard-split on whitespace; a seam
        # mid-sentence is better than silence.
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if not sentence:
            continue
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= limit:
            current = f"{current} {sentence}"
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


def _pcm_to_wav(samples, sample_rate: int) -> bytes:
    """Float32 samples in [-1, 1] to a 16-bit PCM WAV container, in memory."""
    import numpy as np

    array = np.asarray(samples, dtype=np.float32).flatten()
    # Clip before scaling: Kokoro occasionally overshoots 1.0 slightly, and
    # wrapping instead of clipping turns that into an audible crack.
    pcm = (np.clip(array, -1.0, 1.0) * 32767.0).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(TTS_SAMPLE_WIDTH)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _find_kokoro_files() -> Tuple[Optional[Path], Optional[Path]]:
    """Locate a Kokoro ONNX build and its voice pack under `models/kokoro/`.

    The release ships fp32, fp16 and int8 builds under different names, so
    `KOKORO_MODEL_PREFERENCE` is an order of preference rather than a required
    filename: whichever build was downloaded is the one that gets used.
    """
    if not KOKORO_DIR.is_dir():
        return None, None

    candidates = sorted(KOKORO_DIR.glob("*.onnx"))
    tagged = [t for t in KOKORO_MODEL_PREFERENCE if t]
    model: Optional[Path] = None
    for tag in KOKORO_MODEL_PREFERENCE:
        for path in candidates:
            # The empty tag means "the build with no precision suffix", i.e. fp32.
            if (tag and tag in path.name) or (
                not tag and not any(t in path.name for t in tagged)
            ):
                model = path
                break
        if model is not None:
            break
    if model is None and candidates:
        model = candidates[0]

    voices: Optional[Path] = KOKORO_DIR / KOKORO_VOICES_FILE
    if not voices.exists():
        found = sorted(KOKORO_DIR.glob("voices*.bin"))
        voices = found[0] if found else None
    return model, voices


class SpeechSynthesizer:
    """Kokoro-82M if it loads, Windows SAPI if it does not.

    The fallback exists because the realistic failure mode for neural TTS on
    Windows is not quality or speed, it is the phonemiser refusing to load on a
    machine that was never set up for it. SAPI is reached through a fresh
    PowerShell process rather than pyttsx3: the .NET speech stack is present on
    every Windows install, and a per-call subprocess cannot get wedged in the
    run-loop state an in-process engine can after a few dozen turns.
    """

    def __init__(self, voice: str = TTS_VOICE) -> None:
        self.voice = voice
        self.backend = TTS_NONE
        self.kokoro = None
        self.model_path: Optional[Path] = None
        self.load_error: Optional[str] = None
        self.fallback_error: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.backend != TTS_NONE

    @property
    def degraded(self) -> bool:
        """True when speech works, but not through the intended backend."""
        return self.backend == TTS_SAPI

    def load(self) -> "SpeechSynthesizer":
        self._load_kokoro()
        if self.backend == TTS_NONE:
            self._probe_sapi()
        return self

    # ------------------------------------------------------------------ tier 1
    def _load_kokoro(self) -> None:
        model, voices = _find_kokoro_files()
        if model is None or voices is None:
            self.load_error = (
                f"No Kokoro model in {KOKORO_DIR}. "
                "Run `python vendor_models.py --voice` to fetch it."
            )
            return
        try:
            import onnxruntime as ort
            from kokoro_onnx import Kokoro
        except Exception as error:  # noqa: BLE001
            self.load_error = (
                f"kokoro-onnx is not installed ({type(error).__name__}: {error}). "
                'pip install "kokoro-onnx>=0.4.9"'
            )
            return
        try:
            self.kokoro = self._build_kokoro(ort, Kokoro, model, voices)
            # Synthesise once at load, so a broken phonemiser surfaces in the
            # sidebar at startup rather than on the first spoken reply.
            self.kokoro.create("ready", voice=self.voice, speed=1.0, lang=TTS_LANGUAGE)
        except Exception as error:  # noqa: BLE001
            self.kokoro = None
            self.load_error = f"{type(error).__name__}: {error}"
            return
        self.model_path = model
        self.backend = TTS_KOKORO
        self.load_error = None

    @staticmethod
    def _build_kokoro(ort, Kokoro, model: Path, voices: Path):
        """Kokoro on an explicitly threaded onnxruntime session.

        The thread count is the reason this is not just `Kokoro(path, voices)`:
        onnxruntime's default of one thread per logical core is measurably worse
        than one per physical core on a hyperthreaded laptop (see
        KOKORO_INTRA_OP_THREADS). `from_session` is guarded because it is the
        newer of the two constructors -- an older kokoro-onnx still works, just
        at the default thread count.
        """
        if hasattr(Kokoro, "from_session"):
            options = ort.SessionOptions()
            options.intra_op_num_threads = KOKORO_INTRA_OP_THREADS
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(
                str(model), options, providers=["CPUExecutionProvider"]
            )
            return Kokoro.from_session(session, str(voices))
        return Kokoro(str(model), str(voices))

    # ------------------------------------------------------------------ tier 2
    def _probe_sapi(self) -> None:
        if platform.system() != "Windows":
            self.fallback_error = "The SAPI fallback is Windows-only."
            return
        if self._sapi_speak("ready"):
            self.backend = TTS_SAPI
        else:
            self.fallback_error = self.fallback_error or "SAPI produced no audio."

    def _sapi_speak(self, text: str) -> bytes:
        """Drive System.Speech through PowerShell and read back the WAV.

        This is the one place audio touches the filesystem. It is the assistant's
        reply rather than the user's recording, and the file is removed as soon as
        it has been read.
        """
        handle, path = tempfile.mkstemp(suffix=".wav")
        os.close(handle)
        # Single-quoted PowerShell literal: doubling is how a quote is escaped.
        escaped = text.replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.SetOutputToWaveFile('{path}'); "
            f"$s.Speak('{escaped}'); "
            "$s.Dispose()"
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                check=True,
                capture_output=True,
                timeout=60,
            )
            data = Path(path).read_bytes()
        except Exception as error:  # noqa: BLE001
            self.fallback_error = f"{type(error).__name__}: {error}"
            data = b""
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        return data

    # ---------------------------------------------------------------- synthesis
    def synthesize(self, text: str) -> bytes:
        """WAV bytes for `text`, or empty bytes when speech is unavailable."""
        text = (text or "").strip()
        if not text or not self.available:
            return b""
        if self.backend == TTS_SAPI:
            return self._sapi_speak(text)

        chunks = _split_for_synthesis(text)
        if not chunks:
            return b""
        try:
            import numpy as np

            pieces = []
            rate = 24000
            for chunk in chunks:
                samples, rate = self.kokoro.create(
                    chunk, voice=self.voice, speed=TTS_SPEED, lang=TTS_LANGUAGE
                )
                pieces.append(np.asarray(samples, dtype=np.float32).flatten())
            return _pcm_to_wav(np.concatenate(pieces), rate)
        except Exception as error:  # noqa: BLE001 - fall through, never kill the turn
            self.load_error = f"{type(error).__name__}: {error}"
            return b""

    def describe(self) -> str:
        if self.backend == TTS_KOKORO:
            name = self.model_path.name if self.model_path else "kokoro"
            return f"Kokoro-82M ({name}) - voice {self.voice}"
        if self.backend == TTS_SAPI:
            return "Windows SAPI (System.Speech) fallback"
        return "unavailable"
