"""Generic weighted soft-vote ensemble over per-member transformer checkpoints.

Shared by the Stress header (equal-weight BERT + DeBERTa-v3) and the CBT
header (BERT + MentalBERT + DeBERTa-v3 with OOF-selected weights). Loading and
averaging logic is written once; each header supplies its model architecture
and head-logits extraction.
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


@dataclass
class EnsembleMemberSpec:
    name: str
    checkpoint_path: Any
    hf_id: str
    runtime_hf_id: str
    max_len: int
    weight: float


@dataclass
class EnsemblePrediction:
    label: str
    is_positive: bool
    confidence: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class EnsembleMemberResources:
    """Loaded resources for one ensemble member used by on-demand XAI."""

    name: str
    model: nn.Module
    tokenizer: Any
    max_len: int
    weight: float
    decision_threshold: float
    device: torch.device


def load_state_dict(checkpoint_path, device: torch.device) -> dict[str, torch.Tensor]:
    """Load a checkpoint's state dict, unwrapping common save-wrapper shapes."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint content: {type(checkpoint).__name__}")
    return {key.removeprefix("module."): value for key, value in checkpoint.items()}


class WeightedEnsemblePredictor:
    """Loads N transformer checkpoints and averages weight_i * softmax(logits_i).

    `model_factory(runtime_hf_id, intermediate) -> nn.Module` builds the
    architecture for one member. `logits_fn(model, input_ids, attention_mask)`
    extracts the relevant binary-head logits tensor from that architecture's
    forward output (different headers return different output shapes).
    `head_weight_key` is the state_dict key used to infer the head's
    `intermediate` size, so no per-member hyperparameters need to be tracked.
    """

    def __init__(
        self,
        members: list[EnsembleMemberSpec],
        model_factory: Callable[[str, int], nn.Module],
        logits_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
        head_weight_key: str,
        labels: dict[str, str],
        threshold: float = 0.5,
        device: str | torch.device | None = None,
    ):
        self._members = members
        self._model_factory = model_factory
        self._logits_fn = logits_fn
        self._head_weight_key = head_weight_key
        self._labels = labels
        self._threshold = threshold
        self._device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._lock = Lock()
        self._loaded = False
        self._loaded_models: dict[str, nn.Module] = {}
        self._loaded_tokenizers: dict[str, Any] = {}

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def member_names(self) -> tuple[str, ...]:
        return tuple(member.name for member in self._members)

    def load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            for member in self._members:
                if not member.checkpoint_path.exists():
                    raise FileNotFoundError(f"Missing checkpoint: {member.checkpoint_path}")

                state_dict = load_state_dict(member.checkpoint_path, self._device)
                head_weight = state_dict.get(self._head_weight_key)
                if head_weight is None:
                    raise KeyError(
                        f"{member.name} checkpoint missing {self._head_weight_key!r}"
                    )
                intermediate = int(head_weight.shape[0])

                tokenizer = AutoTokenizer.from_pretrained(member.runtime_hf_id)
                model = self._model_factory(member.runtime_hf_id, intermediate)
                model.load_state_dict(state_dict, strict=True)
                model.to(self._device)
                model.eval()

                self._loaded_tokenizers[member.name] = tokenizer
                self._loaded_models[member.name] = model

            self._loaded = True

    def member_resources(self, member_name: str) -> EnsembleMemberResources:
        """Return one loaded member without exposing mutable ensemble mappings."""
        self.load()
        member = next(
            (item for item in self._members if item.name == member_name),
            None,
        )
        if member is None:
            choices = ", ".join(self.member_names)
            raise ValueError(
                f"Unknown ensemble member {member_name!r}; choose from: {choices}"
            )
        return EnsembleMemberResources(
            name=member.name,
            model=self._loaded_models[member.name],
            tokenizer=self._loaded_tokenizers[member.name],
            max_len=member.max_len,
            weight=member.weight,
            decision_threshold=self._threshold,
            device=self._device,
        )

    def predict_proba(self, text: str) -> torch.Tensor:
        # Models must be constructed outside inference mode. On-demand
        # Integrated Gradients reuses a loaded ensemble member and needs its
        # parameters to remain compatible with autograd. Creating the models
        # inside inference mode permanently turns their tensors into inference
        # tensors, which makes Captum's backward pass fail.
        self.load()
        with torch.inference_mode():
            total = None
            for member in self._members:
                tokenizer = self._loaded_tokenizers[member.name]
                model = self._loaded_models[member.name]
                encoded = tokenizer(
                    text,
                    max_length=member.max_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(self._device)
                attention_mask = encoded["attention_mask"].to(self._device)

                logits = self._logits_fn(model, input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu()
                weighted = probs * member.weight
                total = weighted if total is None else total + weighted
        return total

    def predict(self, text: str) -> EnsemblePrediction:
        probs = self.predict_proba(text)
        positive_prob = float(probs[1].item())
        pred_idx = int(positive_prob >= self._threshold)
        label = self._labels.get(str(pred_idx), str(pred_idx))
        confidence = positive_prob if pred_idx == 1 else 1.0 - positive_prob
        return EnsemblePrediction(
            label=label,
            is_positive=pred_idx == 1,
            confidence=confidence,
            probabilities={
                self._labels.get("0", "0"): float(probs[0].item()),
                self._labels.get("1", "1"): float(probs[1].item()),
            },
        )
