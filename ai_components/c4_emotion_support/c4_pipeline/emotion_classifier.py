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


@dataclass
class ClassifierOutput:
    label: str
    confidence: float
    probabilities: Dict[str, float]


class EmotionClassifier:
    def __init__(self, model_path: Path, labels: Optional[List[str]] = None) -> None:
        self.model_path = Path(model_path)
        self.labels = labels or list(EMOTION_LABELS)
        self.fallback = True
        self.model = None
        self.tokenizer = None
        self.label_mapping = {index: label for index, label in enumerate(self.labels)}
        self._try_load_model()

    def _try_load_model(self) -> None:
        if AutoTokenizer is None or AutoModelForSequenceClassification is None:
            return
        if not self._model_files_present():
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_path)
            )
            self.model.eval()
            self.fallback = False
            if hasattr(self.model.config, "id2label") and self.model.config.id2label:
                self.label_mapping = {
                    int(key): value
                    for key, value in self.model.config.id2label.items()
                }
                self.labels = [self.label_mapping[i] for i in sorted(self.label_mapping)]
        except Exception:
            self.model = None
            self.tokenizer = None
            self.fallback = True

    def _model_files_present(self) -> bool:
        if not self.model_path.exists():
            return False
        config_path = self.model_path / "config.json"
        weight_files = [
            self.model_path / "pytorch_model.bin",
            self.model_path / "model.safetensors",
        ]
        return config_path.exists() and any(path.exists() for path in weight_files)

    def _keyword_emotion(self, text: str) -> ClassifierOutput:
        lowered = text.lower()
        keywords = {
            "joy": ["happy", "glad", "excited", "relieved", "better", "great"],
            "sadness": ["sad", "down", "depressed", "lonely", "hopeless"],
            "anger": ["angry", "mad", "annoyed", "frustrated", "rage"],
            "fear": ["scared", "afraid", "nervous", "anxious", "worried", "panic"],
            "disgust": ["disgust", "gross", "sick", "repulsed"],
            "surprise": ["surprised", "shocked", "sudden", "unexpected"],
        }
        for label, words in keywords.items():
            if any(word in lowered for word in words):
                confidence = 0.6
                return ClassifierOutput(
                    label=label,
                    confidence=confidence,
                    probabilities=self._build_probabilities(label, confidence),
                )
        label = "neutral"
        confidence = 0.4
        return ClassifierOutput(
            label=label,
            confidence=confidence,
            probabilities=self._build_probabilities(label, confidence),
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

    def predict(self, text: str) -> ClassifierOutput:
        if self.fallback or self.model is None or self.tokenizer is None:
            return self._keyword_emotion(text)
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )
            outputs = self.model(**inputs)
            logits = outputs.logits.squeeze(0)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            top_index = int(torch.argmax(probs).item())
            label = self.label_mapping.get(top_index, self.labels[top_index])
            confidence = float(probs[top_index].item())
            probabilities = {
                self.label_mapping.get(i, self.labels[i]): float(prob)
                for i, prob in enumerate(probs.tolist())
            }
            return ClassifierOutput(
                label=label, confidence=confidence, probabilities=probabilities
            )

    def status(self) -> Dict[str, str]:
        return {
            "available": str(not self.fallback),
            "fallback": str(self.fallback),
            "model_path": str(self.model_path),
        }
