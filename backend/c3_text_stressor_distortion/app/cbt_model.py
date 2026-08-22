from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import (
    CBT_BINARY_CONFIG_PATH,
    CBT_MODEL_DIR,
    CBT_RUNTIME_HF_ID_OVERRIDES,
    CBT_TRAIN_MODEL_PATH,
    DEVICE,
)
from .ensembles import (
    EnsembleMemberResources,
    EnsembleMemberSpec,
    WeightedEnsemblePredictor,
)


def _binary_model_class():
    """Import BinaryCDTModel from the canonical training source instead of
    duplicating the architecture (same trick as the existing CBT XAI loader:
    ai_components/.../CBT header/xai/model_loader.py)."""
    module_name = "backend_cbt_training_model"
    if module_name in sys.modules:
        return sys.modules[module_name].BinaryCDTModel
    spec = importlib.util.spec_from_file_location(module_name, CBT_TRAIN_MODEL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load CBT model architecture: {CBT_TRAIN_MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.BinaryCDTModel


@dataclass
class CBTPredictionResult:
    """Result of a cognitive-distortion prediction."""

    label: str
    has_distortion: bool
    confidence: float
    probabilities: dict[str, float]


def _resolve_device() -> torch.device:
    if DEVICE != "auto":
        return torch.device(DEVICE)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CBTPredictor:
    """Weighted 3-transformer ensemble (BERT + MentalBERT + DeBERTa-v3) for
    cognitive-distortion detection. Weights and threshold come from the
    existing OOF-selected deployment config (binary_config.json) rather than
    being re-derived here.
    """

    def __init__(self):
        self._device = _resolve_device()
        self._labels = {"0": "No Distortion", "1": "Distortion"}
        self._ensemble = self._build_ensemble()

    def _build_ensemble(self) -> WeightedEnsemblePredictor:
        if not CBT_BINARY_CONFIG_PATH.exists():
            raise FileNotFoundError(f"Missing CBT config: {CBT_BINARY_CONFIG_PATH}")
        config = json.loads(CBT_BINARY_CONFIG_PATH.read_text(encoding="utf-8"))
        model_specs = config["models"]
        deployment = config["deployment"]
        weights = deployment["weights"]
        self._labels = {
            str(index): label
            for index, label in sorted(
                config.get("task", self._labels).items(), key=lambda item: int(item[0])
            )
        }

        members = [
            EnsembleMemberSpec(
                name=name,
                checkpoint_path=CBT_MODEL_DIR / spec["checkpoint_file"],
                hf_id=spec["hf_id"],
                runtime_hf_id=CBT_RUNTIME_HF_ID_OVERRIDES.get(name, spec["hf_id"]),
                max_len=int(spec["max_length"]),
                weight=float(weights[name]),
            )
            for name, spec in model_specs.items()
            if name in weights
        ]

        binary_model_class = _binary_model_class()

        def model_factory(runtime_hf_id: str, intermediate: int) -> nn.Module:
            return binary_model_class(runtime_hf_id, intermediate=intermediate)

        return WeightedEnsemblePredictor(
            members=members,
            model_factory=model_factory,
            logits_fn=lambda model, ids, mask: model(input_ids=ids, attention_mask=mask),
            head_weight_key="binary_head.fc1.weight",
            labels=self._labels,
            threshold=float(deployment["threshold"]),
            device=self._device,
        )

    def load(self) -> None:
        self._ensemble.load()

    @property
    def is_loaded(self) -> bool:
        return self._ensemble.is_loaded

    def xai_resources(
        self,
        member_name: str = "DeBERTa-v3",
    ) -> EnsembleMemberResources:
        """Expose one trained member for an explicitly labelled XAI request."""
        return self._ensemble.member_resources(member_name)

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
