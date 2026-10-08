"""Inference architecture matching the saved binary CBT checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class ClassificationHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate: int,
        num_classes: int,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(head_dropout)
        self.fc2 = nn.Linear(intermediate, num_classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(values))))


class BinaryCDTModel(nn.Module):
    """Shared transformer encoder with a two-class distortion head."""

    def __init__(
        self,
        model_dir: str | Path,
        dropout: float = 0.3,
        head_dropout: float = 0.1,
        intermediate: int = 256,
    ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        self.encoder = AutoModel.from_config(config)
        hidden = int(config.hidden_size)
        self.layer_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.binary_head = ClassificationHead(hidden, intermediate, 2, head_dropout)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(self.layer_norm(output.last_hidden_state[:, 0, :]))
        return self.binary_head(cls)
