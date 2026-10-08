"""Public predictor for the C3 cognitive-distortion ensemble."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from xai import TokenAttribution, explain_tokens, pad_preserved_baseline

from .config import (
    CBT_CALIBRATION_PATH,
    CBT_DIR,
    CBT_MODEL_DIRS,
    DEVICE,
    load_binary_config,
    validate_config,
)
from ..calibration import TemperatureCalibrator
from ..ensemble_xai import combine_member_attributions
from .ensembles import (
    EnsembleMemberResources,
    EnsembleMemberSpec,
    WeightedEnsemblePredictor,
)
from .model import BinaryCDTModel
from .preprocessing import preprocess_for_model


@dataclass
class CBTPredictionResult:
    label: str
    has_distortion: bool
    confidence: float
    probabilities: dict[str, float]


def _resolve_device(requested: str | torch.device | None = None) -> torch.device:
    selected = str(requested) if requested is not None else DEVICE
    if selected != "auto":
        return torch.device(selected)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CBTPredictor:
    """BERT + MentalBERT + DeBERTa-v3 weighted soft-vote predictor."""

    def __init__(self, device: str | torch.device | None = None) -> None:
        validate_config()
        self._device = _resolve_device(device)
        config = load_binary_config()
        labels = config.get("task", {"0": "No Distortion", "1": "Distortion"})
        if "labels" in labels:
            labels = labels["labels"]
        self._labels = {str(key): value for key, value in labels.items()}
        deployment = config["deployment"]
        weights = deployment["weights"]
        calibrator = TemperatureCalibrator.from_json(CBT_CALIBRATION_PATH)

        members = [
            EnsembleMemberSpec(
                name=name,
                checkpoint_path=CBT_DIR / spec["checkpoint_file"],
                model_dir=CBT_MODEL_DIRS[name],
                max_len=int(spec["max_length"]),
                weight=float(weights[name]),
            )
            for name, spec in config["models"].items()
        ]
        self._ensemble = WeightedEnsemblePredictor(
            members=members,
            model_factory=self._build_model,
            logits_fn=lambda model, ids, mask: model(
                input_ids=ids, attention_mask=mask
            ),
            head_weight_key="binary_head.fc1.weight",
            labels=self._labels,
            threshold=float(deployment["threshold"]),
            device=self._device,
            preprocess_fn=preprocess_for_model,
            calibrator=calibrator,
        )

    @staticmethod
    def _build_model(model_dir: str, intermediate: int) -> nn.Module:
        return BinaryCDTModel(model_dir, intermediate=intermediate)

    def load(self) -> None:
        self._ensemble.load()

    @property
    def is_loaded(self) -> bool:
        return self._ensemble.is_loaded

    @property
    def calibration(self) -> dict | None:
        return self._ensemble.calibration_summary

    @property
    def checkpoint_paths(self) -> dict[str, str]:
        return {
            member.name: str(member.checkpoint_path)
            for member in self._ensemble.members
        }

    def xai_resources(
        self, member_name: str = "DeBERTa-v3"
    ) -> EnsembleMemberResources:
        return self._ensemble.member_resources(member_name)

    def xai_ensemble_resources(self) -> tuple[EnsembleMemberResources, ...]:
        return self._ensemble.all_member_resources()

    def predict(self, text: str) -> CBTPredictionResult:
        result = self._ensemble.predict(text)
        return CBTPredictionResult(
            label=result.label,
            has_distortion=result.is_positive,
            confidence=result.confidence,
            probabilities={
                "no_distortion": result.probabilities.get("No Distortion", 0.0),
                "distortion": result.probabilities.get("Distortion", 0.0),
            },
        )

    def explain(
        self,
        text: str,
        *,
        has_distortion: bool | None = None,
        n_steps: int = 32,
    ) -> list[TokenAttribution]:
        """Weight-normalized Integrated Gradients for the actual ensemble."""

        if has_distortion is None:
            has_distortion = self.predict(text).has_distortion
        target_idx = int(has_distortion)
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
                logits = current_model(input_ids=ids, attention_mask=mask)
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
