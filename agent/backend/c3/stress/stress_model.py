from __future__ import annotations

import json
from dataclasses import dataclass

import torch
import torch.nn as nn

from xai import TokenAttribution, explain_tokens, pad_preserved_baseline

from .config import (
    DEFAULT_NUM_SUBREDDITS,
    DEVICE,
    METADATA_PATH,
    STRESS_DECISION_THRESHOLD,
    STRESS_CALIBRATION_PATH,
    STRESS_ENSEMBLE_MEMBERS,
    validate_config,
)
from ..calibration import TemperatureCalibrator
from ..ensemble_xai import combine_member_attributions
from .ensembles import (
    EnsembleMemberResources,
    EnsembleMemberSpec,
    WeightedEnsemblePredictor,
)
from .model import DualHeadStressModel
from .preprocessing import preprocess_for_model


@dataclass
class PredictionResult:
    """Result of a stress prediction, including label, confidence, and probabilities."""

    label: str
    is_stressed: bool
    confidence: float
    probabilities: dict[str, float]


def _resolve_device(requested: str | torch.device | None = None) -> torch.device:
    selected = str(requested) if requested is not None else DEVICE
    if selected != "auto":
        return torch.device(selected)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StressPredictor:
    """Equal-weight BERT + DeBERTa-v3 probability ensemble for stress.

    This is the research ensemble produced by the Stress training pipeline.
    See STRESS_ENSEMBLE_MEMBERS / STRESS_DECISION_THRESHOLD in config.py.
    """

    def __init__(self, device: str | torch.device | None = None):
        validate_config()
        self._device = _resolve_device(device)
        self._num_subreddits = DEFAULT_NUM_SUBREDDITS
        self._labels = {"0": "Not Stressed", "1": "Stressed"}
        self._load_metadata()
        calibrator = TemperatureCalibrator.from_json(STRESS_CALIBRATION_PATH)

        members = [
            EnsembleMemberSpec(
                name=name,
                checkpoint_path=spec["checkpoint"],
                model_dir=spec["model_dir"],
                max_len=spec["max_len"],
                weight=spec["weight"],
            )
            for name, spec in STRESS_ENSEMBLE_MEMBERS.items()
        ]
        self._ensemble = WeightedEnsemblePredictor(
            members=members,
            model_factory=self._build_model,
            logits_fn=lambda model, ids, mask: model(input_ids=ids, attention_mask=mask)[0],
            head_weight_key="head_1a.fc1.weight",
            labels=self._labels,
            threshold=STRESS_DECISION_THRESHOLD,
            device=self._device,
            preprocess_fn=preprocess_for_model,
            calibrator=calibrator,
        )

    def _build_model(self, model_dir: str, intermediate: int) -> nn.Module:
        return DualHeadStressModel(
            model_dir,
            num_subreddit_labels=self._num_subreddits,
            intermediate=intermediate,
        )

    def _load_metadata(self) -> None:
        if not METADATA_PATH.exists():
            return
        with METADATA_PATH.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        self._num_subreddits = int(metadata.get("num_labels", {}).get("head1b", self._num_subreddits))
        self._labels = metadata.get("labels", {}).get("head1a", self._labels)

    def load(self) -> None:
        self._ensemble.load()

    @property
    def is_loaded(self) -> bool:
        return self._ensemble.is_loaded

    @property
    def checkpoint_paths(self) -> dict[str, str]:
        return {name: str(spec["checkpoint"]) for name, spec in STRESS_ENSEMBLE_MEMBERS.items()}

    @property
    def calibration(self) -> dict | None:
        return self._ensemble.calibration_summary

    def xai_resources(
        self,
        member_name: str = "DeBERTa-v3",
    ) -> EnsembleMemberResources:
        """Expose one trained member for an explicitly labelled XAI request."""
        return self._ensemble.member_resources(member_name)

    def xai_ensemble_resources(self) -> tuple[EnsembleMemberResources, ...]:
        return self._ensemble.all_member_resources()

    def predict(self, text: str) -> PredictionResult:
        result = self._ensemble.predict(text)
        return PredictionResult(
            label=result.label,
            is_stressed=result.is_positive,
            confidence=result.confidence,
            probabilities={
                "not_stressed": result.probabilities.get("Not Stressed", 0.0),
                "stressed": result.probabilities.get("Stressed", 0.0),
            },
        )

    def explain(
        self,
        text: str,
        *,
        is_stressed: bool | None = None,
        n_steps: int = 32,
    ) -> list[TokenAttribution]:
        """Weight-normalized Integrated Gradients for the actual ensemble."""

        if is_stressed is None:
            is_stressed = self.predict(text).is_stressed
        target_idx = int(is_stressed)
        member_results = []
        for resources in self.xai_ensemble_resources():
            model = resources.model
            tokenizer = resources.tokenizer
            device = resources.device
            model_text = preprocess_for_model(text, resources.name)
            encoded = tokenizer(
                model_text,
                max_length=resources.max_len,
                padding=False,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            special_tokens_mask = tokenizer.get_special_tokens_mask(
                input_ids[0].tolist(), already_has_special_tokens=True
            )
            baseline_ids = pad_preserved_baseline(
                input_ids,
                tokenizer.pad_token_id,
                special_tokens_mask,
            )

            def forward_func(ids, mask, current_model=model):
                logits = current_model(input_ids=ids, attention_mask=mask)[0]
                return torch.softmax(logits, dim=-1)[:, target_idx]

            attributions = explain_tokens(
                forward_func=forward_func,
                embedding_layer=model.encoder.get_input_embeddings(),
                input_ids=input_ids,
                baseline_ids=baseline_ids,
                tokens=tokenizer.convert_ids_to_tokens(input_ids[0].tolist()),
                additional_forward_args=(attention_mask,),
                n_steps=n_steps,
            )
            member_results.append((resources.name, resources.weight, attributions))
        return combine_member_attributions(member_results)
