"""Inference architecture matching the saved dual-head stress checkpoints."""

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


class DualHeadStressModel(nn.Module):
    """Shared encoder with binary-stress and subreddit classification heads."""

    def __init__(
        self,
        model_dir: str | Path,
        num_subreddit_labels: int = 10,
        num_binary_labels: int = 2,
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
        self.head_1a = ClassificationHead(
            hidden, intermediate, num_binary_labels, head_dropout
        )
        self.head_1b = ClassificationHead(
            hidden, intermediate, num_subreddit_labels, head_dropout
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(self.layer_norm(output.last_hidden_state[:, 0, :]))
        return self.head_1a(cls), self.head_1b(cls)
