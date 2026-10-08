"""Current-emotion -> next-intensity inference pipeline.

The forecaster is the locked hybrid trained in ``model/forcast``: a sparse
TF-IDF + emotion logistic model and a three-seed MLP ensemble over frozen
mean-pooled embeddings from the current-emotion RoBERTa, blended 50/50 and cut
at a validation-tuned threshold. It answers one question -- will the user's
next turn be more or less intense -- rather than naming the next emotion, so
the family is carried over from the current emotion and the pair is written
back into the same 13-label state vocabulary the rest of the app reads (see
``core.intensity_state``).

Both branches consume the current-emotion model's own output, and the fusion
branch embeds with that same encoder, so ``CurrentEmotionClassifier`` is a
dependency of the forecaster rather than merely its predecessor: one RoBERTa is
loaded, and it is the one the features were built from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from xai import TokenAttribution, explain_tokens, pad_preserved_baseline

from .core import (
    CURRENT_EMOTIONS,
    INTENSITY_LABELS,
    intensity_probability_map,
    intensity_state,
    normalize_current_emotion,
    normalize_intensity,
    select_reply_route,
    validate_forecaster_metadata,
)
from .models import build_fusion_mlp


def mean_pool(
    last_hidden: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Average the encoder states over real tokens only.

    Identical to ``model/forcast/model_train/fusion.py``; padding must not
    dilute the vector or inference would not reproduce the training features.
    """

    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


@dataclass(frozen=True)
class EmotionPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class IntensityForecast:
    """The forecaster's own output, before it is named as a state label."""

    label: str
    probability_high: float
    threshold: float
    # Each branch's P(high) before blending, kept for the trace and the XAI
    # panel: the two disagree often enough that showing only the blend hides
    # which half of the model drove a turn.
    baseline_probability: float
    fusion_probability: float

    @property
    def confidence(self) -> float:
        """The model's probability for the class it reported, not for ``high``.

        The decision threshold is tuned (0.595, not 0.5), so a ``low`` call
        made just under it carries a confidence below 0.5. That is the honest
        reading -- the turn was called low because the threshold said so, not
        because the model was sure -- and it is why the routing floor treats
        such a forecast as no evidence at all.
        """

        return (
            self.probability_high
            if self.label == "high"
            else 1.0 - self.probability_high
        )

    @property
    def is_high(self) -> bool:
        return self.label == "high"

    def as_dict(self) -> dict[str, float | str]:
        return {
            "intensity": self.label,
            "probability_high": self.probability_high,
            "threshold": self.threshold,
            "baseline_probability": self.baseline_probability,
            "fusion_probability": self.fusion_probability,
        }


@dataclass(frozen=True)
class ChainedPrediction:
    current: EmotionPrediction
    next: EmotionPrediction
    forecast: IntensityForecast
    reply_route: str

    @property
    def intensifies(self) -> bool:
        return self.forecast.is_high


class CurrentEmotionClassifier:
    def __init__(self, model_dir: str | Path, device: torch.device) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"Current-emotion model not found: {self.model_dir}")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, local_files_only=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir, local_files_only=True
        ).to(self.device)
        self.model.eval()
        self.id_to_label = {
            int(index): str(label).lower()
            for index, label in self.model.config.id2label.items()
        }
        # The bare transformer under the classification head, i.e. what the
        # fusion branch's training script mean-pooled. Kept as an attribute so
        # the forecaster embeds with this exact encoder instead of loading a
        # second copy of the same weights.
        self.encoder = getattr(self.model, self.model.base_model_prefix)

    def probability_vector(self, prediction: EmotionPrediction) -> list[float]:
        """The 5 class probabilities in the order both branches were fitted on."""

        return [
            float(prediction.probabilities[label]) for label in CURRENT_EMOTIONS
        ]

    def embed(self, texts: Sequence[str], max_length: int) -> np.ndarray:
        """Mask-aware mean-pooled encoder states, as used to fit the fusion MLP."""

        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        with torch.inference_mode():
            hidden = self.encoder(**encoded).last_hidden_state
            pooled = mean_pool(hidden, encoded["attention_mask"])
        return pooled.cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def predict(self, text: str) -> EmotionPrediction:
        if not text.strip():
            raise ValueError("text cannot be empty")
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        logits = self.model(**encoded).logits[0]
        probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
        probability_map = {
            self.id_to_label[index]: float(value)
            for index, value in enumerate(probabilities)
        }
        best_index = int(torch.argmax(logits).item())
        label = self.id_to_label[best_index]
        normalize_current_emotion(label)
        return EmotionPrediction(label, probability_map[label], probability_map)

    def explain(
        self, text: str, *, target_label: str | None = None, n_steps: int = 32
    ) -> list[TokenAttribution]:
        """Integrated Gradients token attribution for the current-emotion label."""

        if not text.strip():
            raise ValueError("text cannot be empty")
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        if target_label is None:
            with torch.no_grad():
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                target_idx = int(torch.argmax(logits, dim=-1).item())
        else:
            label_to_id = {label: index for index, label in self.id_to_label.items()}
            target_idx = label_to_id[normalize_current_emotion(target_label)]

        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            input_ids[0].tolist(), already_has_special_tokens=True
        )
        baseline_ids = pad_preserved_baseline(
            input_ids, self.tokenizer.pad_token_id, special_tokens_mask
        )

        def forward_func(ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            logits = self.model(input_ids=ids, attention_mask=mask).logits
            return torch.softmax(logits, dim=-1)[:, target_idx]

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        return explain_tokens(
            forward_func=forward_func,
            embedding_layer=self.model.get_input_embeddings(),
            input_ids=input_ids,
            baseline_ids=baseline_ids,
            tokens=tokens,
            additional_forward_args=(attention_mask,),
            n_steps=n_steps,
        )


class NextIntensityForecaster:
    """The locked hybrid: sparse logistic + frozen-RoBERTa MLP ensemble.

    Loads the trained artifacts described by ``model/forcast/metadata.json``.
    Both blend weights and the decision threshold come from that manifest,
    which records the values selected on the validation split -- nothing is
    retuned here.
    """

    def __init__(
        self,
        artifact_dir: str | Path,
        classifier: CurrentEmotionClassifier,
        device: torch.device,
    ) -> None:
        self.artifact_dir = Path(artifact_dir).expanduser().resolve()
        metadata_path = self.artifact_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Next-intensity metadata not found: {metadata_path}"
            )
        self.device = device
        self.classifier = classifier
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        validate_forecaster_metadata(self.metadata)

        blend = self.metadata["blend"]
        self.baseline_weight = float(blend["baseline_weight"])
        self.fusion_weight = float(blend["fusion_weight"])
        self.threshold = float(blend["threshold"])

        fusion = self.metadata["fusion"]
        self.max_length = int(fusion["max_length"])
        self.baseline = self._load_baseline()
        self.scaler_mean, self.scaler_std = self._load_scaler(fusion["scaler"])
        self.fusion_models = self._load_fusion_models(fusion["checkpoints"])

    # -- artifact loading ---------------------------------------------------

    def _load_baseline(self):
        path = self.artifact_dir / self.metadata["baseline"]["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Sparse baseline not found: {path}")
        return joblib.load(path)["pipeline"]

    def _load_scaler(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """The train-split mean/std the fusion features were standardized with.

        Skipping this would feed the MLP raw embeddings, which produces
        confident nonsense rather than an error, so the file is required.
        """

        path = self.artifact_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Fusion feature scaler not found: {path}")
        with np.load(path, allow_pickle=False) as scaler:
            mean = torch.as_tensor(scaler["mean"], dtype=torch.float32)
            std = torch.as_tensor(scaler["std"], dtype=torch.float32)
        return mean.to(self.device), std.to(self.device)

    def _load_fusion_models(self, names: Sequence[str]) -> list:
        models = []
        for name in names:
            path = self.artifact_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"Fusion seed checkpoint not found: {path}")
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if int(checkpoint["input_dim"]) != len(self.scaler_mean):
                raise ValueError(
                    f"{name} expects {checkpoint['input_dim']} features but the "
                    f"scaler describes {len(self.scaler_mean)}"
                )
            model = build_fusion_mlp(checkpoint)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            models.append(model.to(self.device).eval())
        if not models:
            raise ValueError("metadata.json lists no fusion checkpoints")
        return models

    # -- the two branches ---------------------------------------------------

    def _baseline_probability(self, text: str, current: EmotionPrediction) -> float:
        """P(high) from the TF-IDF + emotion logistic pipeline.

        The pipeline is a ``ColumnTransformer``, so it is fed the same named
        columns the trainer used rather than a positional array.
        """

        row = {
            "current_user_text": text,
            "current_emotion_pred": current.label,
            "current_emotion_confidence": float(current.confidence),
        }
        for label, value in zip(
            CURRENT_EMOTIONS,
            self.classifier.probability_vector(current),
            strict=True,
        ):
            row[f"current_prob_{label}"] = value
        return float(self.baseline.predict_proba(pd.DataFrame([row]))[0, 1])

    def _standardize(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.scaler_mean) / self.scaler_std

    def _fusion_features(self, text: str, current: EmotionPrediction) -> torch.Tensor:
        embedding = self.classifier.embed([text], self.max_length)
        numeric = np.asarray(
            [[float(current.confidence), *self.classifier.probability_vector(current)]],
            dtype=np.float32,
        )
        stacked = np.concatenate((embedding, numeric), axis=1).astype(np.float32)
        return self._standardize(torch.as_tensor(stacked, device=self.device))

    def _fusion_probability_from(self, features: torch.Tensor) -> torch.Tensor:
        """Mean of every seed's sigmoid -- the ensemble the threshold was tuned on."""

        members = [torch.sigmoid(model(features)) for model in self.fusion_models]
        return torch.stack(members, dim=0).mean(dim=0)

    # -- inference ----------------------------------------------------------

    def predict(self, text: str, current: EmotionPrediction) -> IntensityForecast:
        if not text.strip():
            raise ValueError("text cannot be empty")
        normalize_current_emotion(current.label)
        baseline_probability = self._baseline_probability(text, current)
        with torch.inference_mode():
            features = self._fusion_features(text, current)
            fusion_probability = float(self._fusion_probability_from(features).item())
        blended = (
            self.baseline_weight * baseline_probability
            + self.fusion_weight * fusion_probability
        )
        label = INTENSITY_LABELS[1] if blended >= self.threshold else INTENSITY_LABELS[0]
        return IntensityForecast(
            label=label,
            probability_high=blended,
            threshold=self.threshold,
            baseline_probability=baseline_probability,
            fusion_probability=fusion_probability,
        )

    def explain(
        self,
        text: str,
        current: EmotionPrediction,
        *,
        target_intensity: str | None = None,
        n_steps: int = 32,
    ) -> list[TokenAttribution]:
        """Integrated Gradients token attribution for the forecast intensity.

        Only the fusion branch is differentiable in the tokens, so this
        attributes that half of the blend; the emotion probability block is
        held fixed, the same way the retired forecaster held its categorical
        emotion input fixed. The sparse branch is linear in its own n-gram
        features and is not represented here, so callers should say which half
        of the model they are showing.
        """

        if not text.strip():
            raise ValueError("text cannot be empty")
        level = (
            self.predict(text, current).label
            if target_intensity is None
            else normalize_intensity(target_intensity)
        )
        encoded = self.classifier.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        numeric = torch.as_tensor(
            [[float(current.confidence), *self.classifier.probability_vector(current)]],
            dtype=torch.float32,
            device=self.device,
        )

        def forward_func(
            ids: torch.Tensor, mask: torch.Tensor, emotion: torch.Tensor
        ) -> torch.Tensor:
            hidden = self.classifier.encoder(
                input_ids=ids, attention_mask=mask
            ).last_hidden_state
            pooled = mean_pool(hidden, mask)
            features = self._standardize(
                torch.cat((pooled, emotion.expand(pooled.shape[0], -1)), dim=1)
            )
            probability = self._fusion_probability_from(features)
            return probability if level == "high" else 1.0 - probability

        special_tokens_mask = self.classifier.tokenizer.get_special_tokens_mask(
            input_ids[0].tolist(), already_has_special_tokens=True
        )
        baseline_ids = pad_preserved_baseline(
            input_ids, self.classifier.tokenizer.pad_token_id, special_tokens_mask
        )
        tokens = self.classifier.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        return explain_tokens(
            forward_func=forward_func,
            embedding_layer=self.classifier.model.get_input_embeddings(),
            input_ids=input_ids,
            baseline_ids=baseline_ids,
            tokens=tokens,
            additional_forward_args=(attention_mask, numeric),
            n_steps=n_steps,
        )
class EmotionChain:
    """Load the current-emotion RoBERTa and the next-intensity hybrid once."""

    def __init__(
        self,
        current_model_dir: str | Path,
        forecast_artifact_dir: str | Path,
        device: str | None = None,
        next_negative_min_confidence: float = 0.0,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.current_classifier = CurrentEmotionClassifier(
            current_model_dir, self.device
        )
        # The forecaster reuses the classifier's encoder and its probability
        # output, so it is constructed from the classifier rather than from a
        # second model directory.
        self.next_forecaster = NextIntensityForecaster(
            forecast_artifact_dir, self.current_classifier, self.device
        )
        self.next_negative_min_confidence = float(next_negative_min_confidence)

    def _next_state(
        self, current: EmotionPrediction, forecast: IntensityForecast
    ) -> EmotionPrediction:
        """Name the forecast as a state label the rest of the app already reads.

        ``confidence`` stays the confidence of the low/high call rather than
        the mass on the named state, so a neutral turn -- where the label space
        offers no low/high state to name -- still reports what the forecaster
        was actually sure of.
        """

        label = intensity_state(current.label, forecast.label)
        probabilities = intensity_probability_map(
            current.label, forecast.probability_high
        )
        return EmotionPrediction(label, forecast.confidence, probabilities)

    def predict(self, current_text: str) -> ChainedPrediction:
        current = self.current_classifier.predict(current_text)
        forecast = self.next_forecaster.predict(current_text, current)
        next_prediction = self._next_state(current, forecast)
        return ChainedPrediction(
            current=current,
            next=next_prediction,
            forecast=forecast,
            reply_route=select_reply_route(
                current.label,
                next_prediction.label,
                next_emotion_confidence=next_prediction.confidence,
                next_negative_min_confidence=self.next_negative_min_confidence,
            ),
        )

    def explain(
        self,
        current_text: str,
        *,
        n_steps: int = 32,
    ) -> tuple[list[TokenAttribution], list[TokenAttribution]]:
        """Convenience: (current-emotion, next-intensity) attributions for one
        message, both against the labels this chain actually predicted."""

        current = self.current_classifier.predict(current_text)
        forecast = self.next_forecaster.predict(current_text, current)
        current_attribution = self.current_classifier.explain(
            current_text, target_label=current.label, n_steps=n_steps
        )
        next_attribution = self.next_forecaster.explain(
            current_text,
            current,
            target_intensity=forecast.label,
            n_steps=n_steps,
        )
        return current_attribution, next_attribution
