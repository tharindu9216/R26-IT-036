"""Runtime services for the voice branch of the supportive agent.

The C2 appraisal model remains independent of the text-only C3 Stress and CBT
headers. One decoded waveform is shared by C2 and Faster-Whisper so browser
recordings do not need a format-specific conversion step.
"""

from __future__ import annotations

import io
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16_000
APPRAISAL_STATES = (
    "threat",
    "challenge",
    "withdrawal_distress",
    "negative_activation",
    "positive_activation",
    "calm_recovery",
)
VOICE_STRESS_STATES = {
    "threat",
    "challenge",
    "withdrawal_distress",
    "negative_activation",
}


def decode_audio(content: bytes, *, sampling_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode WAV/FLAC/MP3/M4A/OGG/WebM bytes to mono float32 PCM."""

    if not content:
        raise ValueError("audio file is empty")
    try:
        from faster_whisper.audio import decode_audio as whisper_decode_audio
    except ImportError as exc:
        raise RuntimeError(
            "Voice input requires faster-whisper; install backend/requirements.txt"
        ) from exc

    try:
        audio = whisper_decode_audio(io.BytesIO(content), sampling_rate=sampling_rate)
    except Exception as exc:
        raise ValueError(f"audio could not be decoded: {exc}") from exc
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("decoded audio does not contain a mono waveform")
    if not np.isfinite(audio).all():
        raise ValueError("decoded audio contains invalid samples")
    return audio


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None
    language_probability: float | None
    duration_seconds: float


@dataclass(frozen=True)
class AppraisalPrediction:
    arousal: float
    dominance: float
    valence: float
    dominant_state: str
    dominant_weight: float
    state_strengths: dict[str, float]
    interpretation_confidence: float
    appraisal_ambiguity: float
    reference_similarity: float
    uncertain: bool
    ood: bool
    low_confidence: bool
    voice_stress: bool
    voice_stress_score: float
    contributions: list[dict[str, float | str]]

    def as_dict(self) -> dict:
        return {
            "arousal": self.arousal,
            "dominance": self.dominance,
            "valence": self.valence,
            "dominant_state": self.dominant_state,
            "dominant_weight": self.dominant_weight,
            "state_strengths": dict(self.state_strengths),
            "interpretation_confidence": self.interpretation_confidence,
            "appraisal_ambiguity": self.appraisal_ambiguity,
            "reference_similarity": self.reference_similarity,
            "uncertain": self.uncertain,
            "ood": self.ood,
            "low_confidence": self.low_confidence,
            "voice_stress": self.voice_stress,
            "voice_stress_score": self.voice_stress_score,
            "contributions": list(self.contributions),
        }


class RegressionHead:
    """Factory namespace kept separate so torch remains a lazy dependency."""

    @staticmethod
    def build(config):
        import torch.nn as nn

        class _RegressionHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

            def forward(self, features):
                import torch

                value = self.dropout(features)
                value = torch.tanh(self.dense(value))
                return self.out_proj(self.dropout(value))

        return _RegressionHead()


def _emotion_model_class():
    """Build the AudEERING-compatible Wav2Vec2 regression architecture."""

    import torch
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        Wav2Vec2Model,
        Wav2Vec2PreTrainedModel,
    )

    class EmotionModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead.build(config)
            self.post_init()

        def forward(self, input_values):
            hidden_states = self.wav2vec2(input_values)[0]
            pooled = torch.mean(hidden_states, dim=1)
            return self.classifier(pooled)

    return EmotionModel


class VoiceAppraisalAnalyzer:
    """AudEERING ADV regression followed by the fitted fuzzy appraisal engine."""

    def __init__(
        self,
        model_dir: str | Path,
        appraisal_artifact: str | Path,
        *,
        device: str = "cpu",
    ) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.appraisal_artifact = Path(appraisal_artifact).expanduser().resolve()
        self.device = device
        self._feature_extractor = None
        self._model = None
        self._engine = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

        if not (self.model_dir / "config.json").is_file():
            raise FileNotFoundError(f"C2 AudEERING model not found: {self.model_dir}")
        if not self.appraisal_artifact.is_file():
            raise FileNotFoundError(
                f"C2 appraisal artifact not found: {self.appraisal_artifact}"
            )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._engine is not None

    def load(self) -> None:
        if self.is_loaded:
            return
        with self._load_lock:
            if self.is_loaded:
                return
            import joblib
            from transformers import Wav2Vec2FeatureExtractor

            model_class = _emotion_model_class()
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.model_dir,
                local_files_only=True,
            )
            model = model_class.from_pretrained(
                self.model_dir,
                local_files_only=True,
            )
            model.to(self.device).eval()
            self._feature_extractor = feature_extractor
            self._model = model
            self._engine = joblib.load(self.appraisal_artifact)

    def predict(self, audio: np.ndarray, *, file_name: str = "voice_input") -> AppraisalPrediction:
        import pandas as pd
        import torch

        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("C2 requires a non-empty mono waveform")
        self.load()

        with self._inference_lock, torch.inference_mode():
            inputs = self._feature_extractor(
                waveform,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
            )
            logits = self._model(inputs.input_values.to(self.device))[0]
            values = logits.detach().cpu().to(torch.float32).numpy()
            if values.shape[0] != 3:
                raise RuntimeError(f"C2 expected 3 ADV outputs, received {values.shape[0]}")
            arousal, dominance, valence = (float(value) for value in values)

            sample = pd.DataFrame(
                [
                    {
                        "file_name": file_name,
                        "actual_class": "unknown",
                        "arousal": arousal,
                        "dominance": dominance,
                        "valence": valence,
                    }
                ]
            )
            row = self._engine.transform(sample).iloc[0]

        strengths = {
            state: float(row[f"state_{state}"]) for state in APPRAISAL_STATES
        }
        dominant_state = str(row["dominant_appraisal_state"])
        uncertain = bool(row["interpretation_uncertain"])
        voice_stress_score = max(strengths[state] for state in VOICE_STRESS_STATES)
        contributions = sorted(
            (
                {
                    "label": state,
                    "score": float(row[f"xai_contribution_{state}"]),
                }
                for state in APPRAISAL_STATES
            ),
            key=lambda item: abs(float(item["score"])),
            reverse=True,
        )
        return AppraisalPrediction(
            arousal=arousal,
            dominance=dominance,
            valence=valence,
            dominant_state=dominant_state,
            dominant_weight=float(row["dominant_appraisal_weight"]),
            state_strengths=strengths,
            interpretation_confidence=float(row["interpretation_confidence"]),
            appraisal_ambiguity=float(row["appraisal_ambiguity"]),
            reference_similarity=float(row["reference_similarity"]),
            uncertain=uncertain,
            ood=bool(row["ood_flag"]),
            low_confidence=bool(row["low_interpretation_confidence"]),
            voice_stress=(not uncertain and dominant_state in VOICE_STRESS_STATES),
            voice_stress_score=voice_stress_score,
            contributions=contributions,
        )


class VoiceTranscriber:
    """Offline Faster-Whisper transcription using the local CTranslate2 model."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        task: str = "transcribe",
        language: str | None = None,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.device = device
        self.compute_type = compute_type
        self.task = task
        self.language = language
        self._model = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        if not (self.model_dir / "model.bin").is_file():
            raise FileNotFoundError(f"Faster-Whisper model not found: {self.model_dir}")

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "Voice transcription requires faster-whisper"
                ) from exc
            self._model = WhisperModel(
                str(self.model_dir),
                device=self.device,
                compute_type=self.compute_type,
                local_files_only=True,
            )

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("transcription requires a non-empty mono waveform")
        self.load()
        with self._inference_lock:
            segments, info = self._model.transcribe(
                waveform,
                language=self.language,
                task=self.task,
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=True,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
        if not text:
            raise ValueError("No speech was detected in the audio")
        probability = getattr(info, "language_probability", None)
        return TranscriptionResult(
            text=text,
            language=getattr(info, "language", None),
            language_probability=float(probability) if probability is not None else None,
            duration_seconds=float(waveform.size / SAMPLE_RATE),
        )


class KokoroSpeechSynthesizer:
    """Offline Kokoro synthesis using the local model and voice pack."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        voice: str = "af_heart",
        language_code: str = "a",
        device: str = "cpu",
        speed: float = 1.0,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.voice = voice
        self.language_code = language_code
        self.device = device
        self.speed = float(speed)
        self._pipeline = None
        self._voice_path = self.model_dir / "voices" / f"{voice}.pt"
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        if not (self.model_dir / "kokoro-v1_0.pth").is_file():
            raise FileNotFoundError(f"Kokoro model not found: {self.model_dir}")
        if not self._voice_path.is_file():
            raise FileNotFoundError(f"Kokoro voice not found: {self._voice_path}")

    def load(self) -> None:
        if self._pipeline is not None:
            return
        with self._load_lock:
            if self._pipeline is not None:
                return
            try:
                from kokoro import KModel, KPipeline
            except ImportError as exc:
                raise RuntimeError("Voice output requires kokoro") from exc
            model = KModel(
                repo_id="local/Kokoro-82M",
                config=str(self.model_dir / "config.json"),
                model=str(self.model_dir / "kokoro-v1_0.pth"),
            ).to(self.device).eval()
            self._pipeline = KPipeline(
                lang_code=self.language_code,
                repo_id="local/Kokoro-82M",
                model=model,
                device=self.device,
            )

    def synthesize(self, text: str) -> bytes:
        normalized = text.strip()
        if not normalized:
            raise ValueError("TTS text cannot be empty")
        self.load()
        with self._inference_lock:
            chunks = [
                np.asarray(result.audio, dtype=np.float32)
                for result in self._pipeline(
                    normalized,
                    voice=str(self._voice_path),
                    speed=self.speed,
                )
                if result.audio is not None and len(result.audio) > 0
            ]
        if not chunks:
            raise RuntimeError("Kokoro generated no audio")
        waveform = np.concatenate(chunks)
        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError("Kokoro WAV output requires soundfile") from exc
        output = io.BytesIO()
        sf.write(output, waveform, 24_000, format="WAV", subtype="PCM_16")
        return output.getvalue()
