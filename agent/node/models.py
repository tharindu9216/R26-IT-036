"""Standalone model runners for the support node.

Deliberately self-contained: nothing here imports from ``../backend``, so this
folder plus the matching subdirectories of ``../model`` is everything a second
machine needs. The node runs models and returns results -- it holds no routing,
safety, escalation or conversation state. Those decisions stay in the backend
so they cannot drift between a one-machine and a two-machine deployment.

The one exception is the C2 appraisal engine: its ``.joblib`` artifact is
pickled against the backend's own ``c2.appraisal_stress`` package, so hosting
C2 here also needs that package importable (see ``appraisal_package_dir`` in
config.yaml). Leave ``services.c2: false`` and C2 simply stays on the main
machine.
"""

from __future__ import annotations

import io
import sys
import threading
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


def pcm_to_waveform(payload: bytes) -> np.ndarray:
    """Decode the backend's raw little-endian float32 body into a waveform."""

    if not payload:
        raise ValueError("audio body is empty")
    if len(payload) % 4:
        raise ValueError("audio body is not a whole number of float32 samples")
    waveform = np.frombuffer(payload, dtype="<f4").astype(np.float32)
    if waveform.size == 0:
        raise ValueError("audio body decoded to zero samples")
    if not np.isfinite(waveform).all():
        raise ValueError("audio body contains invalid samples")
    return waveform


class ReplyModel:
    """Qwen3 base + the ESConv LoRA, switched per request.

    The adapter is attached once at load time and enabled or disabled for each
    call, exactly as the backend does when it runs the model itself -- one set
    of weights in VRAM, not two.
    """

    def __init__(
        self,
        base_model_dir: Path,
        adapter_dir: Path,
        *,
        device: str = "auto",
        quantization: str = "nf4",
    ) -> None:
        self.base_model_dir = Path(base_model_dir).expanduser().resolve()
        self.adapter_dir = Path(adapter_dir).expanduser().resolve()
        self.device = device
        self.quantization = str(quantization or "none").lower()
        self._model = None
        self._tokenizer = None
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()

        if not (self.base_model_dir / "config.json").is_file():
            raise FileNotFoundError(f"Qwen base model not found: {self.base_model_dir}")
        if not (self.adapter_dir / "adapter_config.json").is_file():
            raise FileNotFoundError(f"ESConv adapter not found: {self.adapter_dir}")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def _quantization_config(self):
        if self.quantization in {"", "none", "full", "bf16", "fp16"}:
            return None
        import torch
        from transformers import BitsAndBytesConfig

        if self.quantization in {"nf4", "4bit", "int4"}:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        if self.quantization in {"int8", "8bit"}:
            return BitsAndBytesConfig(load_in_8bit=True)
        raise ValueError(
            f"Unsupported quantization: {self.quantization!r} "
            "(expected none, nf4 or int8)"
        )

    def load(self) -> None:
        if self.is_loaded:
            return
        with self._load_lock:
            if self.is_loaded:
                return
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_dir, local_files_only=True
            )
            load_kwargs = {
                "local_files_only": True,
                "torch_dtype": "auto",
                "low_cpu_mem_usage": True,
                "device_map": "auto" if self.device == "auto" else {"": self.device},
            }
            quantization_config = self._quantization_config()
            if quantization_config is not None:
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["quantization_config"] = quantization_config
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_dir, **load_kwargs
            )
            model = PeftModel.from_pretrained(
                base_model,
                self.adapter_dir,
                adapter_name="supportive",
                is_trainable=False,
                local_files_only=True,
            )
            model.eval()
            self._tokenizer = tokenizer
            self._model = model

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        use_adapter: bool,
    ) -> str:
        from collections.abc import Mapping

        self.load()
        with self._generation_lock:
            encoded = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            input_device = self._model.get_input_embeddings().weight.device
            # transformers >= 5 returns a BatchEncoding (a UserDict) here; 4.x
            # returned the input_ids tensor directly.
            if isinstance(encoded, Mapping):
                model_inputs = {
                    key: value.to(input_device)
                    for key, value in encoded.items()
                    if key in {"input_ids", "attention_mask"}
                }
            else:
                model_inputs = {"input_ids": encoded.to(input_device)}
            input_ids = model_inputs["input_ids"]
            generation_kwargs = {
                "max_new_tokens": int(max_new_tokens),
                "do_sample": temperature > 0,
                "temperature": float(temperature),
                "top_p": float(top_p),
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
            }
            if use_adapter:
                self._model.set_adapter("supportive")
                output = self._model.generate(**model_inputs, **generation_kwargs)
            else:
                with self._model.disable_adapter():
                    output = self._model.generate(**model_inputs, **generation_kwargs)
            generated = output[0, input_ids.shape[1] :]
            return self._tokenizer.decode(generated, skip_special_tokens=True).strip()


class Transcriber:
    """Offline Faster-Whisper transcription on the local CTranslate2 model."""

    def __init__(
        self,
        model_dir: Path,
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

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                str(self.model_dir),
                device=self.device,
                compute_type=self.compute_type,
                local_files_only=True,
            )

    def transcribe(self, waveform: np.ndarray) -> dict:
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
        probability = getattr(info, "language_probability", None)
        # An empty transcript is returned rather than raised: the backend's
        # client turns it into the same "No speech was detected" ValueError the
        # local transcriber raises, so callers see one behaviour.
        return {
            "text": text,
            "language": getattr(info, "language", None),
            "language_probability": (
                float(probability) if probability is not None else None
            ),
            "duration_seconds": float(waveform.size / SAMPLE_RATE),
        }


class Synthesizer:
    """Offline Kokoro synthesis returning 24 kHz PCM-16 WAV bytes."""

    def __init__(
        self,
        model_dir: Path,
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
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        if not (self.model_dir / "kokoro-v1_0.pth").is_file():
            raise FileNotFoundError(f"Kokoro model not found: {self.model_dir}")
        if not self._voice_path(voice).is_file():
            raise FileNotFoundError(f"Kokoro voice not found: {self._voice_path(voice)}")

    def _voice_path(self, voice: str) -> Path:
        return self.model_dir / "voices" / f"{voice}.pt"

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def load(self) -> None:
        if self._pipeline is not None:
            return
        with self._load_lock:
            if self._pipeline is not None:
                return
            from kokoro import KModel, KPipeline

            model = (
                KModel(
                    repo_id="local/Kokoro-82M",
                    config=str(self.model_dir / "config.json"),
                    model=str(self.model_dir / "kokoro-v1_0.pth"),
                )
                .to(self.device)
                .eval()
            )
            self._pipeline = KPipeline(
                lang_code=self.language_code,
                repo_id="local/Kokoro-82M",
                model=model,
                device=self.device,
            )

    def synthesize(
        self, text: str, *, voice: str | None = None, speed: float | None = None
    ) -> bytes:
        normalized = text.strip()
        if not normalized:
            raise ValueError("TTS text cannot be empty")
        selected = voice or self.voice
        voice_path = self._voice_path(selected)
        if not voice_path.is_file():
            raise ValueError(f"Unknown Kokoro voice: {selected}")
        self.load()
        with self._inference_lock:
            chunks = [
                np.asarray(result.audio, dtype=np.float32)
                for result in self._pipeline(
                    normalized,
                    voice=str(voice_path),
                    speed=float(speed if speed is not None else self.speed),
                )
                if result.audio is not None and len(result.audio) > 0
            ]
        if not chunks:
            raise RuntimeError("Kokoro generated no audio")
        import soundfile as sf

        output = io.BytesIO()
        sf.write(output, np.concatenate(chunks), 24_000, format="WAV", subtype="PCM_16")
        return output.getvalue()


def _emotion_model_class():
    """Rebuild the AudEERING-compatible Wav2Vec2 ADV regression architecture."""

    import torch
    import torch.nn as nn
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        Wav2Vec2Model,
        Wav2Vec2PreTrainedModel,
    )

    class RegressionHead(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, features):
            value = self.dropout(features)
            value = torch.tanh(self.dense(value))
            return self.out_proj(self.dropout(value))

    class EmotionModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.post_init()

        def forward(self, input_values):
            hidden_states = self.wav2vec2(input_values)[0]
            return self.classifier(torch.mean(hidden_states, dim=1))

    return EmotionModel


class AppraisalAnalyzer:
    """AudEERING ADV regression followed by the fitted fuzzy appraisal engine."""

    def __init__(
        self,
        model_dir: Path,
        appraisal_artifact: Path,
        *,
        device: str = "cpu",
        package_dir: Path | None = None,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.appraisal_artifact = Path(appraisal_artifact).expanduser().resolve()
        self.device = device
        # The artifact unpickles as ``c2.appraisal_stress.engine.<class>``, so
        # the directory holding that package has to be importable before
        # joblib.load runs. Pointing at the backend folder is enough.
        self.package_dir = Path(package_dir).expanduser().resolve() if package_dir else None
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
        if self.package_dir is not None and not (self.package_dir / "c2").is_dir():
            raise FileNotFoundError(
                "c2.appraisal_stress package not found under "
                f"{self.package_dir}; set c2.appraisal_package_dir to the "
                "backend folder, or disable services.c2 and keep C2 on the "
                "main machine"
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

            if self.package_dir is not None:
                package_path = str(self.package_dir)
                if package_path not in sys.path:
                    sys.path.insert(0, package_path)

            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.model_dir, local_files_only=True
            )
            model = _emotion_model_class().from_pretrained(
                self.model_dir, local_files_only=True
            )
            model.to(self.device).eval()
            self._feature_extractor = feature_extractor
            self._model = model
            self._engine = joblib.load(self.appraisal_artifact)

    def predict(self, waveform: np.ndarray, *, file_name: str = "voice_input") -> dict:
        import pandas as pd
        import torch

        self.load()
        with self._inference_lock, torch.inference_mode():
            inputs = self._feature_extractor(
                waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt"
            )
            logits = self._model(inputs.input_values.to(self.device))[0]
            values = logits.detach().cpu().to(torch.float32).numpy()
            if values.shape[0] != 3:
                raise RuntimeError(
                    f"C2 expected 3 ADV outputs, received {values.shape[0]}"
                )
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

        strengths = {state: float(row[f"state_{state}"]) for state in APPRAISAL_STATES}
        dominant_state = str(row["dominant_appraisal_state"])
        uncertain = bool(row["interpretation_uncertain"])
        contributions = sorted(
            (
                {"label": state, "score": float(row[f"xai_contribution_{state}"])}
                for state in APPRAISAL_STATES
            ),
            key=lambda item: abs(float(item["score"])),
            reverse=True,
        )
        return {
            "arousal": arousal,
            "dominance": dominance,
            "valence": valence,
            "dominant_state": dominant_state,
            "dominant_weight": float(row["dominant_appraisal_weight"]),
            "state_strengths": strengths,
            "interpretation_confidence": float(row["interpretation_confidence"]),
            "appraisal_ambiguity": float(row["appraisal_ambiguity"]),
            "reference_similarity": float(row["reference_similarity"]),
            "uncertain": uncertain,
            "ood": bool(row["ood_flag"]),
            "low_confidence": bool(row["low_interpretation_confidence"]),
            "voice_stress": (not uncertain and dominant_state in VOICE_STRESS_STATES),
            "voice_stress_score": max(
                strengths[state] for state in VOICE_STRESS_STATES
            ),
            "contributions": contributions,
        }
