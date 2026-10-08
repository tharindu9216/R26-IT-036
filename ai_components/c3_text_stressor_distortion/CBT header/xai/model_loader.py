"""Load trained CBT binary checkpoints for prediction and XAI."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer


XAI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = XAI_DIR.parents[3]
MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "c3_text_stressor_distortion"
    / "CBT header"
)
CONFIG_PATH = MODEL_DIR / "binary_config.json"
TRAIN_MODEL_PATH = XAI_DIR.parent / "train" / "model.py"


class DualOutputBinaryAdapter(nn.Module):
    """Expose BinaryCDTModel through the dual-output interface shared by XAI."""

    def __init__(self, binary_model: nn.Module):
        super().__init__()
        self.binary_model = binary_model

    @property
    def encoder(self):
        return self.binary_model.encoder

    def forward(self, input_ids, attention_mask, global_attention_mask=None):
        logits = self.binary_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        )
        return logits, None


@dataclass
class CBTXAIResources:
    model_name: str
    model: nn.Module
    tokenizer: Any
    labels: dict[str, str]
    max_length: int
    decision_threshold: float
    checkpoint_path: Path
    device: torch.device


def load_binary_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing CBT binary config: {CONFIG_PATH}")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _binary_model_class():
    """Import the canonical training architecture without duplicating it."""
    module_name = "cbt_xai_training_model"
    if module_name in sys.modules:
        return sys.modules[module_name].BinaryCDTModel
    spec = importlib.util.spec_from_file_location(module_name, TRAIN_MODEL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load CBT model architecture: {TRAIN_MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.BinaryCDTModel


def _state_dict(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:  # PyTorch versions before weights_only was introduced.
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint content: {type(checkpoint).__name__}")
    return {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
    }


def load_cbt_xai_resources(
    model_name: str = "DeBERTa-v3",
    device: str | torch.device | None = None,
) -> CBTXAIResources:
    config = load_binary_config()
    model_configs = config.get("models", {})
    if model_name not in model_configs:
        choices = ", ".join(model_configs)
        raise ValueError(f"Unknown CBT model {model_name!r}; choose from: {choices}")

    spec = model_configs[model_name]
    # MentalBERT is gated, but its architecture and tokenizer vocabulary are
    # BERT-base-uncased compatible. The fine-tuned checkpoint below contains
    # the complete encoder state, so using the public runtime scaffold does
    # not replace any trained MentalBERT weights.
    runtime_hf_id = spec.get("runtime_hf_id", spec["hf_id"])
    checkpoint_path = MODEL_DIR / spec["checkpoint_file"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing CBT checkpoint: {checkpoint_path}")
    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    state_dict = _state_dict(checkpoint_path, resolved_device)
    head_weight = state_dict.get("binary_head.fc1.weight")
    if head_weight is None:
        raise KeyError("Checkpoint does not contain binary_head.fc1.weight")

    auth_token = os.getenv("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(
        runtime_hf_id,
        token=auth_token,
    )
    model_class = _binary_model_class()
    binary_model = model_class(
        runtime_hf_id,
        dropout=float(spec.get("dropout", 0.3)),
        head_dropout=float(spec.get("head_dropout", 0.1)),
        intermediate=int(head_weight.shape[0]),
    )
    binary_model.load_state_dict(state_dict, strict=True)
    binary_model.to(resolved_device).eval()

    labels = {
        str(index): label
        for index, label in sorted(
            config.get(
                "task",
                {"0": "No Distortion", "1": "Distortion"},
            ).items(),
            key=lambda item: int(item[0]),
        )
    }
    return CBTXAIResources(
        model_name=model_name,
        model=DualOutputBinaryAdapter(binary_model).to(resolved_device).eval(),
        tokenizer=tokenizer,
        labels=labels,
        max_length=int(spec["max_length"]),
        decision_threshold=float(spec["threshold"]),
        checkpoint_path=checkpoint_path,
        device=resolved_device,
    )
