"""Current-emotion classification.

Wraps the fine-tuned RoBERTa checkpoint produced by ../../classification
(`models/optuna_comparison/roberta-base`, test accuracy 0.8214 / macro-F1
0.8162 on 5 classes). Falls back to a keyword matcher so the demo still runs
with the model files absent.

The loaded `model` and `tokenizer` stay as public attributes because the XAI
layer needs the embedding matrix and the raw token ids, not just the label.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception:  # pragma: no cover
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

from config import EMOTION_LABELS

MAX_LENGTH = 128  # matches MAX_LENGTH used during training


@dataclass
class ClassifierOutput:
    label: str
    confidence: float
    probabilities: Dict[str, float]
    label_id: int = 0
    source: str = "fallback"


class EmotionClassifier:
    def __init__(
        self,
        model_path: Path,
        labels: Optional[List[str]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.labels = labels or list(EMOTION_LABELS)
        self.fallback = True
        self.model = None
        self.tokenizer = None
        self.model_name = "keyword-fallback"
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.load_error: Optional[str] = None
        self.label_mapping = {index: label for index, label in enumerate(self.labels)}
        self._try_load_model()

    # ------------------------------------------------------------------ loading
    def _try_load_model(self) -> None:
        if AutoTokenizer is None or AutoModelForSequenceClassification is None:
            self.load_error = "transformers is not installed"
            return
        if not self._model_files_present():
            self.load_error = f"no model files under {self.model_path}"
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_path)
            )
            self.model.to(self.device)
            self.model.eval()
            self.fallback = False
            self.model_name = getattr(self.model.config, "_name_or_path", "") or "transformer"
            self.model_name = getattr(self.model.config, "model_type", self.model_name)
            # Trust the checkpoint's own id2label over the config default: if the
            # two ever disagree, the checkpoint is the one that is right.
            if getattr(self.model.config, "id2label", None):
                self.label_mapping = {
                    int(key): value for key, value in self.model.config.id2label.items()
                }
                self.labels = [self.label_mapping[i] for i in sorted(self.label_mapping)]
        except Exception as error:  # pragma: no cover - depends on local files
            self.model = None
            self.tokenizer = None
            self.fallback = True
            self.load_error = f"{type(error).__name__}: {error}"

    def _model_files_present(self) -> bool:
        if not self.model_path.exists():
            return False
        config_path = self.model_path / "config.json"
        weight_files = [
            self.model_path / "pytorch_model.bin",
            self.model_path / "model.safetensors",
        ]
        return config_path.exists() and any(path.exists() for path in weight_files)

    # ------------------------------------------------------------------ fallback
    def _keyword_emotion(self, text: str) -> ClassifierOutput:
        lowered = text.lower()
        keywords = {
            "joy": ["happy", "glad", "excited", "relieved", "better", "great"],
            "sadness": ["sad", "down", "depressed", "lonely", "hopeless"],
            "anger": ["angry", "mad", "annoyed", "frustrated", "rage"],
            "fear": ["scared", "afraid", "nervous", "anxious", "worried", "panic"],
        }
        for label, words in keywords.items():
            if label in self.labels and any(word in lowered for word in words):
                return self._make_output(label, 0.6, "fallback")
        return self._make_output("neutral", 0.4, "fallback")

    def _make_output(self, label: str, confidence: float, source: str) -> ClassifierOutput:
        label_id = self.labels.index(label) if label in self.labels else 0
        return ClassifierOutput(
            label=label,
            confidence=confidence,
            probabilities=self._build_probabilities(label, confidence),
            label_id=label_id,
            source=source,
        )

    def _build_probabilities(self, label: str, confidence: float) -> Dict[str, float]:
        remaining = max(0.0, 1.0 - confidence)
        other_labels = [item for item in self.labels if item != label]
        if not other_labels:
            return {label: 1.0}
        share = remaining / len(other_labels)
        probs = {item: share for item in other_labels}
        probs[label] = confidence
        return probs

    # ------------------------------------------------------------------ inference
    def encode(self, text: str):
        """Tokenised inputs on the model device. Used by predict and by the XAI layer."""
        return self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(self.device)

    def predict(self, text: str) -> ClassifierOutput:
        if self.fallback or self.model is None or self.tokenizer is None:
            return self._keyword_emotion(text)
        with torch.no_grad():
            inputs = self.encode(text)
            logits = self.model(**inputs).logits.squeeze(0)
            probs = torch.softmax(logits, dim=-1)
        top_index = int(torch.argmax(probs).item())
        probabilities = {
            self.label_mapping.get(i, str(i)): float(prob)
            for i, prob in enumerate(probs.tolist())
        }
        return ClassifierOutput(
            label=self.label_mapping.get(top_index, str(top_index)),
            confidence=float(probs[top_index].item()),
            probabilities=probabilities,
            label_id=top_index,
            source="model",
        )

    def probabilities_for_texts(self, texts: List[str]) -> torch.Tensor:
        """Batched softmax probabilities. Backs the occlusion explainer."""
        if self.fallback or self.model is None or self.tokenizer is None:
            raise RuntimeError("no trained classifier loaded")
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
            ).to(self.device)
            logits = self.model(**inputs).logits
            return torch.softmax(logits, dim=-1).cpu()

    # ------------------------------------------------------------------ status
    def status(self) -> Dict[str, object]:
        return {
            "available": not self.fallback,
            "fallback": self.fallback,
            "model_path": str(self.model_path),
            "model_name": self.model_name,
            "device": str(self.device),
            "labels": list(self.labels),
            "load_error": self.load_error,
        }
