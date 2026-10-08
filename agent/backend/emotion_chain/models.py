"""Forecaster architectures matching the exported checkpoints.

``FusionMLP`` is the live one: the head of the next-intensity hybrid, run over
frozen mean-pooled current-emotion embeddings concatenated with that model's
probability vector. It must stay layer-for-layer identical to
``model/forcast/model_train/fusion.py``, because the exported seed
checkpoints are plain ``state_dict`` saves against it.

The four 13-state architectures below belong to the retired next-emotion
forecaster in ``model/next_emotion_forecaster/``. Nothing in the running app
loads them any more -- ``EmotionChain`` forecasts intensity now -- but the
checkpoints are still on disk, so their loader is kept here rather than
deleted.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class FusionMLP(nn.Module):
    """Compact head over [frozen text embedding | emotion probabilities]."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.35):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def build_fusion_mlp(checkpoint: Mapping[str, Any]) -> FusionMLP:
    """Construct the head declared inside one exported fusion seed checkpoint."""

    return FusionMLP(
        int(checkpoint["input_dim"]),
        int(checkpoint["hidden_dim"]),
        float(checkpoint["dropout"]),
    )


def _masked_attention_pool(
    sequence: torch.Tensor,
    attention: nn.Linear,
    mask: torch.Tensor,
) -> torch.Tensor:
    scores = attention(sequence).squeeze(-1)
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=1)
    return torch.sum(sequence * weights.unsqueeze(-1), dim=1)


class EmotionConditionedHead(nn.Module):
    def __init__(
        self,
        text_feature_size: int,
        class_count: int = 13,
        emotion_count: int = 6,
        emotion_dim: int = 16,
    ) -> None:
        super().__init__()
        self.aux_emb = nn.Embedding(emotion_count, emotion_dim)
        self.fc = nn.Linear(text_feature_size + emotion_dim, class_count)

    def forward(
        self, text_features: torch.Tensor, emotion_ids: torch.Tensor
    ) -> torch.Tensor:
        emotion_features = self.aux_emb(emotion_ids)
        return self.fc(torch.cat((text_features, emotion_features), dim=1))


class BiGRUStateForecaster(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.grus = nn.ModuleList()
        input_size = embedding_dim
        for _ in range(layers):
            self.grus.append(
                nn.GRU(input_size, hidden, batch_first=True, bidirectional=True)
            )
            input_size = hidden * 2
        self.attn = nn.Linear(hidden * 2, 1)
        self.head = EmotionConditionedHead(hidden * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, token_ids: torch.Tensor, emotion_ids: torch.Tensor
    ) -> torch.Tensor:
        mask = token_ids.ne(0)
        sequence = self.dropout(self.emb(token_ids))
        for gru in self.grus:
            sequence, _ = gru(sequence)
            sequence = self.dropout(sequence)
        pooled = _masked_attention_pool(sequence, self.attn, mask)
        return self.head(self.dropout(pooled), emotion_ids)


class BiLSTMStateForecaster(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstms = nn.ModuleList()
        input_size = embedding_dim
        for _ in range(layers):
            self.lstms.append(
                nn.LSTM(input_size, hidden, batch_first=True, bidirectional=True)
            )
            input_size = hidden * 2
        self.attn = nn.Linear(hidden * 2, 1)
        self.head = EmotionConditionedHead(hidden * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, token_ids: torch.Tensor, emotion_ids: torch.Tensor
    ) -> torch.Tensor:
        mask = token_ids.ne(0)
        sequence = self.dropout(self.emb(token_ids))
        for lstm in self.lstms:
            sequence, _ = lstm(sequence)
            sequence = self.dropout(sequence)
        pooled = _masked_attention_pool(sequence, self.attn, mask)
        return self.head(self.dropout(pooled), emotion_ids)


class CNNBiLSTMStateForecaster(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        filters: int,
        kernel_size: int,
        hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv = nn.Conv1d(
            embedding_dim,
            filters,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.lstm = nn.LSTM(
            filters,
            hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = nn.Linear(hidden * 2, 1)
        self.head = EmotionConditionedHead(hidden * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, token_ids: torch.Tensor, emotion_ids: torch.Tensor
    ) -> torch.Tensor:
        mask = token_ids.ne(0)
        embedded = self.dropout(self.emb(token_ids)).transpose(1, 2)
        sequence = F.relu(self.conv(embedded)).transpose(1, 2)
        sequence, _ = self.lstm(self.dropout(sequence))
        pooled = _masked_attention_pool(sequence, self.attn, mask)
        return self.head(self.dropout(pooled), emotion_ids)


class TextCNNStateForecaster(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        filters: int,
        kernel_sizes: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            nn.Conv1d(embedding_dim, filters, size) for size in kernel_sizes
        )
        self.head = EmotionConditionedHead(filters * len(kernel_sizes))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, token_ids: torch.Tensor, emotion_ids: torch.Tensor
    ) -> torch.Tensor:
        embedded = self.dropout(self.emb(token_ids)).transpose(1, 2)
        pooled = [
            F.adaptive_max_pool1d(F.relu(convolution(embedded)), 1).squeeze(2)
            for convolution in self.convs
        ]
        return self.head(self.dropout(torch.cat(pooled, dim=1)), emotion_ids)


def build_state_forecaster(checkpoint: Mapping[str, Any]) -> nn.Module:
    """Construct the architecture declared inside a state-forecast checkpoint."""

    arch = str(checkpoint["arch"])
    params = checkpoint["params"]
    vocab_size = len(checkpoint["vocab"])
    embedding_dim = int(params["emb_dim"])
    dropout = float(params.get("dropout", 0.0))
    if arch == "bigru":
        return BiGRUStateForecaster(
            vocab_size,
            embedding_dim,
            hidden=int(params["hidden"]),
            layers=int(params.get("layers", 1)),
            dropout=dropout,
        )
    if arch == "bilstm":
        return BiLSTMStateForecaster(
            vocab_size,
            embedding_dim,
            hidden=int(params["hidden"]),
            layers=int(params.get("layers", 1)),
            dropout=dropout,
        )
    if arch == "cnn_bilstm":
        return CNNBiLSTMStateForecaster(
            vocab_size,
            embedding_dim,
            filters=int(params["n_filters"]),
            kernel_size=int(params["kernel_size"]),
            hidden=int(params["hidden"]),
            dropout=dropout,
        )
    if arch == "textcnn":
        return TextCNNStateForecaster(
            vocab_size,
            embedding_dim,
            filters=int(params["n_filters"]),
            kernel_sizes=tuple(int(value) for value in params["kernel_sizes"]),
            dropout=dropout,
        )
    raise ValueError(f"Unsupported next-emotion architecture: {arch}")
