"""Next-turn emotion forecasting.

Loads one of the checkpoints trained by ../../../forcasting/train_deep.py:

    textcnn_forecast.pt      test acc 0.7217  macro-F1 0.7003   (0.6 MB)
    bilstm_forecast.pt       test acc 0.7164  macro-F1 0.6903   (5.6 MB)
    distilbert_forecast.pt   test acc 0.7177  macro-F1 0.7025   (265 MB)

Model input is the dialogue-context string (previous k turns + the current
utterance) plus the speaker's *current* emotion as an auxiliary embedded
feature. The current emotion arrives from the RoBERTa classifier in a different
label space, so it goes through `label_mapping.aux_emotion_id` first.

Honest caveat kept visible in the trace: the checkpoints were trained on a
synthetic, template-generated corpus whose vocabulary is only 528 tokens, so
free-text demo input hits `<unk>` often. `unk_rate` is reported per prediction
and the UI warns when it is high.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from config import (
    DEFAULT_FORECASTER,
    DISTILBERT_BASE,
    FORECAST_CONTEXT_TURNS,
    FORECAST_LABELS,
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
)
from .forecast_models import (
    UNK,
    BiLSTMAttention,
    DistilBertClassifier,
    TextCNN,
    build_context_text,
    encode,
    simple_tokenize,
)
from .label_mapping import UNKNOWN_AUX_ID, aux_emotion_id

CHECKPOINT_FILES = {
    "textcnn": "textcnn_forecast.pt",
    "bilstm": "bilstm_forecast.pt",
    "distilbert": "distilbert_forecast.pt",
}


class EmotionForecaster:
    def __init__(
        self,
        model_path: Path,
        labels: Optional[List[str]] = None,
        force_fallback: bool = False,
        architecture: str = DEFAULT_FORECASTER,
        device: Optional[str] = None,
        context_turns: int = FORECAST_CONTEXT_TURNS,
    ) -> None:
        self.model_path = Path(model_path)
        self.labels = labels or list(FORECAST_LABELS)
        self.force_fallback = force_fallback
        self.architecture = architecture
        self.context_turns = context_turns
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = None
        self.vocab: Dict[str, int] = {}
        self.tokenizer = None          # only for distilbert
        self.params: Dict[str, object] = {}
        self.max_len = 64
        self.model_type = "rule"
        self.fallback = True
        self.load_error: Optional[str] = None

        self._load_meta()
        self._try_load_model()

    # ------------------------------------------------------------------ loading
    def _load_meta(self) -> None:
        """Prefer the label order recorded at training time over the config default."""
        meta_file = self.model_path / "meta_forecast.json"
        if not meta_file.exists():
            return
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            label2id = meta.get("label2id") or {}
            if label2id:
                self.labels = [
                    label for label, _ in sorted(label2id.items(), key=lambda kv: kv[1])
                ]
            self.context_turns = int(meta.get("context", self.context_turns))
        except Exception as error:  # pragma: no cover
            self.load_error = f"meta_forecast.json unreadable: {error}"

    def _try_load_model(self) -> None:
        if self.force_fallback:
            self.load_error = "fallback forced by user"
            return
        filename = CHECKPOINT_FILES.get(self.architecture)
        if filename is None:
            self.load_error = f"unknown architecture '{self.architecture}'"
            return
        checkpoint_path = self.model_path / filename
        if not checkpoint_path.exists():
            self.load_error = f"missing checkpoint {checkpoint_path}"
            return

        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            self.params = dict(checkpoint.get("params", {}))
            self.max_len = int(self.params.get("max_len", 64))
            n_classes = len(self.labels)
            n_emo = len(self.labels)

            if self.architecture == "distilbert":
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_BASE)
                model = DistilBertClassifier(
                    DISTILBERT_BASE,
                    n_classes,
                    n_emo,
                    dropout=float(self.params.get("dropout", 0.2)),
                )
            else:
                self.vocab = checkpoint["vocab"]
                if self.architecture == "bilstm":
                    model = BiLSTMAttention(
                        len(self.vocab), n_classes, n_emo,
                        emb_dim=int(self.params.get("emb_dim", 200)),
                        hidden=int(self.params.get("hidden", 192)),
                        layers=int(self.params.get("layers", 1)),
                        dropout=float(self.params.get("dropout", 0.3)),
                    )
                else:
                    model = TextCNN(
                        len(self.vocab), n_classes, n_emo,
                        emb_dim=int(self.params.get("emb_dim", 200)),
                        n_filters=int(self.params.get("n_filters", 128)),
                        dropout=float(self.params.get("dropout", 0.4)),
                    )

            # strict=True on purpose: a silently partial load would produce
            # confident nonsense rather than an error we can show in the UI.
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.to(self.device)
            model.eval()
            self.model = model
            self.model_type = self.architecture
            self.fallback = False
            self.load_error = None
        except Exception as error:
            self.model = None
            self.fallback = True
            self.load_error = f"{type(error).__name__}: {error}"

    # ------------------------------------------------------------------ encoding
    def build_context(
        self,
        current_message: str,
        history: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> str:
        return build_context_text(history or [], current_message, self.context_turns)

    def encode_context(self, context_text: str):
        """Token ids / attention mask for the loaded architecture."""
        if self.architecture == "distilbert":
            enc = self.tokenizer(
                context_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt",
            )
            return (
                enc["input_ids"].to(self.device),
                enc["attention_mask"].to(self.device),
            )
        ids, length = encode(context_text, self.vocab, self.max_len)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        lens = torch.tensor([max(length, 1)], dtype=torch.long, device=self.device)
        return x, lens

    def unk_rate(self, context_text: str) -> float:
        """Share of tokens that fall outside the training vocabulary."""
        if self.architecture == "distilbert" or not self.vocab:
            return 0.0
        tokens = simple_tokenize(context_text)
        if not tokens:
            return 0.0
        unknown = sum(1 for token in tokens if self.vocab.get(token, UNK) == UNK)
        return unknown / len(tokens)

    # ------------------------------------------------------------------ inference
    def _model_probabilities(self, context_text: str, aux_id: int) -> Dict[str, float]:
        aux = torch.tensor([aux_id], dtype=torch.long, device=self.device)
        with torch.no_grad():
            if self.architecture == "distilbert":
                input_ids, attention = self.encode_context(context_text)
                logits = self.model(input_ids, attention, aux)
            else:
                x, lens = self.encode_context(context_text)
                logits = self.model(x, lens, aux)
            probs = torch.softmax(logits.float().squeeze(0), dim=-1)
        return {label: float(probs[i]) for i, label in enumerate(self.labels)}

    def predict(
        self,
        current_emotion: str,
        history: List[str],
        deviation_level: str,
        previous_emotion: Optional[str] = None,
        current_message: str = "",
        dialogue_history: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> Dict[str, object]:
        """Forecast the user's next-turn emotion.

        `history` is the list of past *emotion labels* (used by the rule
        fallback); `dialogue_history` is the list of (speaker, text) turns the
        trained model actually consumes.
        """
        if self.fallback or self.model is None:
            result = self._rule_based_forecast(current_emotion, history, deviation_level)
            result["context_text"] = ""
            result["unk_rate"] = 0.0
            result["aux_emotion_id"] = UNKNOWN_AUX_ID
            return result

        context_text = self.build_context(current_message, dialogue_history)
        aux_id = aux_emotion_id(current_emotion)
        probabilities = self._model_probabilities(context_text, aux_id)
        label = max(probabilities, key=probabilities.__getitem__)
        return {
            "label": label,
            "confidence": probabilities[label],
            "probabilities": probabilities,
            "source": self.architecture,
            "context_text": context_text,
            "unk_rate": self.unk_rate(context_text),
            "aux_emotion_id": aux_id,
            "aux_emotion_label": (
                self.labels[aux_id] if aux_id < len(self.labels) else "unknown"
            ),
        }

    def counterfactual_by_current_emotion(
        self,
        context_text: str,
    ) -> Dict[str, Dict[str, object]]:
        """Re-run the forecast under every possible current emotion.

        The aux feature is the single strongest signal this model has, so
        sweeping it shows how much of the forecast is driven by the classifier's
        output versus by the text itself.
        """
        if self.fallback or self.model is None:
            return {}
        results: Dict[str, Dict[str, object]] = {}
        for aux_id, aux_label in enumerate(self.labels):
            probabilities = self._model_probabilities(context_text, aux_id)
            top = max(probabilities, key=probabilities.__getitem__)
            results[aux_label] = {
                "label": top,
                "confidence": probabilities[top],
                "probabilities": probabilities,
            }
        return results

    # ------------------------------------------------------------------ fallback
    def _rule_based_forecast(
        self, current_emotion: str, history: List[str], deviation_level: str
    ) -> Dict[str, object]:
        """Persistence-style heuristic used when no checkpoint is available.

        Deliberately close to `baseline_persistence` from the training pipeline
        (test macro-F1 0.7098) -- emotion has strong inertia, so "the same as
        now" is a genuinely strong guess.
        """
        if current_emotion in POSITIVE:
            label, confidence = "happy", 0.70
        elif current_emotion in NEUTRAL:
            label, confidence = "neutral", 0.60
        elif current_emotion in NEGATIVE:
            if deviation_level == "High":
                label, confidence = "anxious", 0.65
            else:
                label = {
                    "sadness": "sad",
                    "sad": "sad",
                    "anger": "angry",
                    "angry": "angry",
                    "fear": "anxious",
                    "anxious": "anxious",
                    "stressed": "stressed",
                }.get(current_emotion, "stressed")
                confidence = 0.70
        else:
            label, confidence = "neutral", 0.55

        return {
            "label": label,
            "confidence": confidence,
            "probabilities": self._build_probabilities(label, confidence),
            "source": "rule",
        }

    def _build_probabilities(self, label: str, confidence: float) -> Dict[str, float]:
        other_labels = [item for item in self.labels if item != label]
        if not other_labels:
            return {label: 1.0}
        share = max(0.0, 1.0 - confidence) / len(other_labels)
        probs = {item: share for item in other_labels}
        probs[label] = confidence
        return probs

    # ------------------------------------------------------------------ status
    def status(self) -> Dict[str, object]:
        return {
            "available": not self.fallback,
            "fallback": self.fallback,
            "model_path": str(self.model_path),
            "model_type": self.model_type,
            "architecture": self.architecture,
            "device": str(self.device),
            "labels": list(self.labels),
            "vocab_size": len(self.vocab) if self.vocab else None,
            "max_len": self.max_len,
            "context_turns": self.context_turns,
            "load_error": self.load_error,
        }
