"""Next-turn emotion-STATE forecasting.

Loads a checkpoint trained by `../../../emotion_forecasting_pipeline`:

    textcnn_state_forecast.pt
    bilstm_state_forecast.pt
    bigru_state_forecast.pt
    cnn_bilstm_state_forecast.pt

What this model predicts
------------------------
Not "which emotion comes next" -- the previous forecaster did that, and a
persistence baseline matched it, because an emotion label barely moves from one
turn to the next. This one predicts the next emotional *state*: the emotion
together with where its intensity is going (`high_sadness`, `low_joy`,
`neutral`, ...). `low_sadness` and `high_sadness` are the same emotion and
opposite situations, and that difference is the point.

Model input
-----------
The current utterance plus the speaker's current emotion, and nothing else.
The training corpus also ships a `context_text` column; it was generated from
the target state and a TF-IDF model trained on it scores 1.0000 test accuracy.
It is a leak, not a feature, and this class never touches it. The retrained
model therefore does not read dialogue history at all -- `dialogue_history` is
still accepted so existing callers do not break, and is ignored by design.

The current emotion arrives from the RoBERTa classifier in the *same* five
labels the forecaster was conditioned on, so no projection is needed; it is
embedded as an auxiliary feature via `label_mapping.aux_emotion_id`.

Transition constraint
---------------------
The corpus transition rules are deterministic: from `sadness` the only
reachable next states are `low_sadness`, `high_sadness` and `neutral`. The
forecast distribution is masked to the reachable set, so the pipeline can never
be handed `high_joy` for a user who is currently sad. See
`config.FORECAST_CONSTRAIN_TRANSITIONS`.

Honest caveat, kept visible in the trace rather than buried
-----------------------------------------------------------
The `next_emotion_state` labels are synthetic and rule-constrained. Measured on
the held-out split, the utterance text carries no usable signal about which of
the reachable states follows: a TF-IDF model on the text alone scores at or
below the majority baseline within every current-emotion group. What the model
reliably learns is the transition structure. So a forecast here is a calibrated
statement about *which states are possible and their base rates*, not evidence
that this particular sentence predicts escalation. `trained_signal` in
`status()` reports that, and the UI says it out loud.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from config import (
    DEFAULT_FORECASTER,
    FORECAST_CONSTRAIN_TRANSITIONS,
    FORECAST_LABELS,
)
from .forecast_models import (
    UNK,
    build_model,
    encode,
    simple_tokenize,
)
from .label_mapping import (
    UNKNOWN_AUX_ID,
    aux_emotion_id,
    is_deteriorating,
    reachable_states,
    state_base_emotion,
    state_intensity,
    trajectory,
)

CHECKPOINT_FILES = {
    "textcnn": "textcnn_state_forecast.pt",
    "bilstm": "bilstm_state_forecast.pt",
    "bigru": "bigru_state_forecast.pt",
    "cnn_bilstm": "cnn_bilstm_state_forecast.pt",
}


class EmotionForecaster:
    def __init__(
        self,
        model_path: Path,
        labels: Optional[List[str]] = None,
        force_fallback: bool = False,
        architecture: str = DEFAULT_FORECASTER,
        device: Optional[str] = None,
        constrain_transitions: bool = FORECAST_CONSTRAIN_TRANSITIONS,
    ) -> None:
        self.model_path = Path(model_path)
        self.labels = labels or list(FORECAST_LABELS)
        self.force_fallback = force_fallback
        self.architecture = architecture
        self.constrain_transitions = constrain_transitions
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = None
        self.vocab: Dict[str, int] = {}
        self.params: Dict[str, object] = {}
        self.max_len = 120
        self.model_type = "rule"
        self.fallback = True
        self.load_error: Optional[str] = None
        self.meta: Dict[str, object] = {}

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
            self.meta = meta
            label2id = meta.get("label2id") or {}
            if label2id:
                self.labels = [
                    label for label, _ in sorted(label2id.items(), key=lambda kv: kv[1])
                ]
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
            self.max_len = int(checkpoint.get("max_len", 120))
            self.vocab = checkpoint["vocab"]

            label2id = checkpoint.get("label2id") or {}
            if label2id:
                self.labels = [
                    label for label, _ in sorted(label2id.items(), key=lambda kv: kv[1])
                ]

            model = build_model(
                self.architecture,
                vocab_size=len(self.vocab),
                n_classes=len(self.labels),
                n_emo=UNKNOWN_AUX_ID,
                params=self.params,
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
            self.checkpoint_metrics = checkpoint.get("test_metrics", {})
        except Exception as error:
            self.model = None
            self.fallback = True
            self.load_error = f"{type(error).__name__}: {error}"

    # ------------------------------------------------------------------ encoding
    def encode_context(self, text: str):
        """Token ids / lengths for the loaded architecture.

        Named `encode_context` for continuity with the previous forecaster's
        interface (xai.py calls it), but the text is a single utterance now,
        not a dialogue context window.
        """
        ids, length = encode(text, self.vocab, self.max_len)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        lens = torch.tensor([max(length, 1)], dtype=torch.long, device=self.device)
        return x, lens

    def unk_rate(self, text: str) -> float:
        """Share of tokens that fall outside the training vocabulary."""
        if not self.vocab:
            return 0.0
        tokens = simple_tokenize(text)
        if not tokens:
            return 0.0
        unknown = sum(1 for token in tokens if self.vocab.get(token, UNK) == UNK)
        return unknown / len(tokens)

    # ------------------------------------------------------------------ inference
    def _model_probabilities(
        self, text: str, aux_id: int, current_emotion: Optional[str] = None
    ) -> Dict[str, float]:
        aux = torch.tensor([aux_id], dtype=torch.long, device=self.device)
        with torch.no_grad():
            x, lens = self.encode_context(text)
            logits = self.model(x, lens, aux).float().squeeze(0)
            if self.constrain_transitions and current_emotion is not None:
                logits = self._mask_logits(logits, current_emotion)
            probs = torch.softmax(logits, dim=-1)
        return {label: float(probs[i]) for i, label in enumerate(self.labels)}

    def _mask_logits(self, logits: torch.Tensor, current_emotion: str) -> torch.Tensor:
        """-inf everything the transition rules make unreachable.

        Masking the logits rather than zeroing probabilities keeps the result a
        proper softmax over the reachable set, so `confidence` stays comparable
        across turns.
        """
        allowed = set(reachable_states(current_emotion))
        mask = torch.tensor(
            [label in allowed for label in self.labels],
            dtype=torch.bool,
            device=logits.device,
        )
        if not bool(mask.any()):
            return logits
        return logits.masked_fill(~mask, float("-inf"))

    def predict(
        self,
        current_emotion: str,
        history: List[str],
        deviation_level: str,
        previous_emotion: Optional[str] = None,
        current_message: str = "",
        dialogue_history: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> Dict[str, object]:
        """Forecast the user's next-turn emotional state.

        `history` (past emotion labels) is used only by the rule fallback.
        `dialogue_history` is accepted and ignored: the retrained model's
        leakage-safe feature set is the current utterance plus the current
        emotion, and nothing else. See the module docstring.
        """
        if self.fallback or self.model is None:
            result = self._rule_based_forecast(current_emotion, history, deviation_level)
            result["input_text"] = current_message
            result["unk_rate"] = 0.0
            result["aux_emotion_id"] = UNKNOWN_AUX_ID
            result["aux_emotion_label"] = current_emotion or "unknown"
            return self._decorate(result, current_emotion)

        aux_id = aux_emotion_id(current_emotion)
        probabilities = self._model_probabilities(
            current_message, aux_id, current_emotion
        )
        label = max(probabilities, key=probabilities.__getitem__)
        result = {
            "label": label,
            "confidence": probabilities[label],
            "probabilities": probabilities,
            "source": self.architecture,
            "input_text": current_message,
            "unk_rate": self.unk_rate(current_message),
            "aux_emotion_id": aux_id,
            "aux_emotion_label": (
                self.labels[aux_id] if aux_id < UNKNOWN_AUX_ID else "unknown"
            ),
        }
        return self._decorate(result, current_emotion)

    def _decorate(
        self, result: Dict[str, object], current_emotion: Optional[str]
    ) -> Dict[str, object]:
        """Attach the decomposition that makes a state more than a label."""
        label = str(result["label"])
        result["aux_emotion_label"] = current_emotion or "unknown"
        result["base_emotion"] = state_base_emotion(label)
        result["intensity"] = state_intensity(label)
        result["trajectory"] = trajectory(current_emotion, label)
        result["deteriorating"] = is_deteriorating(current_emotion, label)
        result["reachable_states"] = reachable_states(current_emotion)
        result["constrained"] = bool(
            self.constrain_transitions and not self.fallback
        )
        return result

    def counterfactual_by_current_emotion(
        self,
        text: str,
    ) -> Dict[str, Dict[str, object]]:
        """Re-run the forecast under every possible current emotion.

        The aux feature is the strongest signal this model has, so sweeping it
        shows how much of the forecast is driven by the classifier's output
        versus by the text. With the transition mask on, the sweep also makes
        the constraint visible: each assumed emotion admits a different set of
        states.
        """
        if self.fallback or self.model is None:
            return {}
        from config import FORECAST_CURRENT_EMOTIONS

        results: Dict[str, Dict[str, object]] = {}
        for emotion in FORECAST_CURRENT_EMOTIONS:
            probabilities = self._model_probabilities(
                text, aux_emotion_id(emotion), emotion
            )
            top = max(probabilities, key=probabilities.__getitem__)
            results[emotion] = {
                "label": top,
                "confidence": probabilities[top],
                "probabilities": probabilities,
                "trajectory": trajectory(emotion, top),
            }
        return results

    # ------------------------------------------------------------------ fallback
    def _rule_based_forecast(
        self, current_emotion: str, history: List[str], deviation_level: str
    ) -> Dict[str, object]:
        """Persistence heuristic used when no checkpoint is available.

        Deliberately close to the `baseline_prior_by_current_emotion` reference
        in the training pipeline: stay in the current emotion, and let the
        deviation level decide whether it is escalating or easing. On this
        corpus that baseline is genuinely competitive, which is a fact about
        the labels rather than a compliment to the heuristic.
        """
        if current_emotion == "neutral" or current_emotion is None:
            label, confidence = "neutral", 0.55
        elif current_emotion in ("joy", "sadness", "anger", "fear"):
            intensity = "high" if deviation_level == "High" else "low"
            label = f"{intensity}_{current_emotion}"
            confidence = 0.60
        else:
            label, confidence = "neutral", 0.45

        return {
            "label": label,
            "confidence": confidence,
            "probabilities": self._build_probabilities(
                label, confidence, current_emotion
            ),
            "source": "rule",
        }

    def _build_probabilities(
        self, label: str, confidence: float, current_emotion: Optional[str] = None
    ) -> Dict[str, float]:
        """Spread the remaining mass over the states that are actually reachable."""
        allowed = set(reachable_states(current_emotion)) & set(self.labels)
        if label not in allowed:
            allowed = set(self.labels)
        others = [item for item in allowed if item != label]
        probs = {item: 0.0 for item in self.labels}
        probs[label] = confidence
        if others:
            share = max(0.0, 1.0 - confidence) / len(others)
            for item in others:
                probs[item] = share
        else:
            probs[label] = 1.0
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
            "constrain_transitions": self.constrain_transitions,
            "load_error": self.load_error,
            "target": "next_emotion_state",
            "trained_signal": (
                "transition structure only -- the utterance text does not "
                "separate the reachable states on this synthetic corpus"
            ),
        }
