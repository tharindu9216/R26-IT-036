from __future__ import annotations

import json
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModel

from .config import (
    DEFAULT_NUM_SUBREDDITS,
    DEVICE,
    METADATA_PATH,
    STRESS_DECISION_THRESHOLD,
    STRESS_ENSEMBLE_MEMBERS,
)
from .ensembles import (
    EnsembleMemberResources,
    EnsembleMemberSpec,
    WeightedEnsemblePredictor,
)


class ClassificationHead(nn.Module):
    """Simple feedforward head with one hidden layer and dropout."""

    def __init__(self, hidden_size: int, intermediate: int, num_classes: int, head_dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(head_dropout)
        self.fc2 = nn.Linear(intermediate, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class DualHeadStressModel(nn.Module):
    """Dual-head model for stress detection and subreddit classification."""

    def __init__(
        self,
        hf_id: str,
        num_subreddit_labels: int = 10,
        num_binary_labels: int = 2,
        dropout: float = 0.3,
        head_dropout: float = 0.1,
        intermediate: int = 256,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(hf_id)
        hidden = self.encoder.config.hidden_size
        self.layer_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.head_1a = ClassificationHead(hidden, intermediate, num_binary_labels, head_dropout)
        self.head_1b = ClassificationHead(hidden, intermediate, num_subreddit_labels, head_dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        cls = self.layer_norm(cls)
        cls = self.dropout(cls)
        return self.head_1a(cls), self.head_1b(cls)


@dataclass
class PredictionResult:
    """Result of a stress prediction, including label, confidence, and probabilities."""

    label: str
    is_stressed: bool
    confidence: float
    probabilities: dict[str, float]


def _resolve_device() -> torch.device:
    if DEVICE != "auto":
        return torch.device(DEVICE)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StressPredictor:
    """Equal-weight BERT + DeBERTa-v3 probability ensemble for stress.

    This is the research ensemble produced by the Stress training pipeline.
    See STRESS_ENSEMBLE_MEMBERS / STRESS_DECISION_THRESHOLD in config.py.
    """

    def __init__(self):
        self._device = _resolve_device()
        self._num_subreddits = DEFAULT_NUM_SUBREDDITS
        self._labels = {"0": "Not Stressed", "1": "Stressed"}
        self._load_metadata()

        members = [
            EnsembleMemberSpec(
                name=name,
                checkpoint_path=spec["checkpoint"],
                hf_id=spec["hf_id"],
                runtime_hf_id=spec["runtime_hf_id"],
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
        )

    def _build_model(self, runtime_hf_id: str, intermediate: int) -> nn.Module:
        return DualHeadStressModel(
            runtime_hf_id,
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

    def xai_resources(
        self,
        member_name: str = "DeBERTa-v3",
    ) -> EnsembleMemberResources:
        """Expose one trained member for an explicitly labelled XAI request."""
        return self._ensemble.member_resources(member_name)

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
