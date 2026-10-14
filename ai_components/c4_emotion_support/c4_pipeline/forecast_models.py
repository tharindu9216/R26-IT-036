"""Architectures and text encoding for the next-emotion-STATE forecaster.

GENERATED FILE -- do not edit here.

Copied verbatim from `emotion_forecasting_pipeline/state_forecast_models.py` by
`emotion_forecasting_pipeline/export_to_c4.py`. Edit the training-side file and
re-run the export; editing this copy makes the checkpoint and the serving code
disagree, which shows up as a `load_state_dict` failure at best and silently
degraded predictions at worst.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, UNK = 0, 1

# Current-emotion vocabulary of the forecasting dataset. This is deliberately
# the *same* five labels the C4 current-emotion classifier emits, so the two
# models share a label space and the old lossy 5 -> 8 projection disappears.
CURRENT_EMOTIONS: List[str] = ["neutral", "anger", "fear", "joy", "sadness"]
CURRENT_EMOTION_TO_ID: Dict[str, int] = {
    label: index for index, label in enumerate(CURRENT_EMOTIONS)
}
# Extra final row of `aux_emb`: "unknown / unmappable current emotion".
UNKNOWN_EMOTION_ID = len(CURRENT_EMOTIONS)

# ------------------------------------------------------------------ text utils
_URL = re.compile(r"https?://\S+|www\.\S+")
_WS = re.compile(r"\s+")
_TOK = re.compile(r"[a-z0-9']+|[<\[][a-z]+[>\]]|[!?.,]")


def clean_text(text: str) -> str:
    value = str(text)
    value = _URL.sub(" <url> ", value)
    value = re.sub(r"(.)\1{3,}", r"\1\1\1", value)  # loooove -> loove
    return _WS.sub(" ", value).strip()


def simple_tokenize(text: str) -> List[str]:
    return _TOK.findall(clean_text(text).lower())


def encode(text: str, vocab: Dict[str, int], max_len: int) -> Tuple[List[int], int]:
    """Truncate to max_len, right-pad with PAD. Returns (ids, true_length)."""
    ids = [vocab.get(token, UNK) for token in simple_tokenize(text)][:max_len]
    return ids + [PAD] * (max_len - len(ids)), min(len(ids), max_len)


def build_vocab(texts: Sequence[str], min_freq: int = 2, max_size: int = 30000) -> Dict[str, int]:
    """Vocabulary from TRAINING TEXT ONLY -- never fit on val/test."""
    from collections import Counter

    counter: Counter = Counter()
    for text in texts:
        counter.update(simple_tokenize(text))
    vocab: Dict[str, int] = {"<pad>": PAD, "<unk>": UNK}
    for token, count in counter.most_common():
        if count < min_freq or len(vocab) >= max_size:
            continue
        vocab[token] = len(vocab)
    return vocab


def emotion_id(label: str) -> int:
    """Aux id for a current emotion, falling back to the unknown bucket."""
    return CURRENT_EMOTION_TO_ID.get(str(label).strip().lower(), UNKNOWN_EMOTION_ID)


# ------------------------------------------------------------------ shared bits
class _AuxHead(nn.Module):
    """Concat [pooled text features | aux emotion embedding] -> logits."""

    def __init__(self, feature_dim: int, n_classes: int, n_emo: int,
                 aux_dim: int = 16, dropout: float = 0.3):
        super().__init__()
        self.aux_emb = nn.Embedding(n_emo + 1, aux_dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(feature_dim + aux_dim, n_classes)

    def forward(self, pooled: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        h = torch.cat([pooled, self.aux_emb(aux)], dim=-1)
        return self.fc(self.drop(h))


def _attention_pool(out: torch.Tensor, attn: nn.Linear, mask: torch.Tensor) -> torch.Tensor:
    scores = attn(out).squeeze(-1).masked_fill(~mask, -1e4)
    weights = torch.softmax(scores, dim=1).unsqueeze(-1)
    return (out * weights).sum(1)


# ------------------------------------------------------------------ models
class TextCNN(nn.Module):
    def __init__(self, vocab_size, n_classes, n_emo, emb_dim=200, n_filters=128,
                 kernel_sizes=(2, 3, 4, 5), dropout=0.4, aux_dim=16):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, n_filters, k, padding=k - 1) for k in kernel_sizes])
        self.drop = nn.Dropout(dropout)
        self.head = _AuxHead(n_filters * len(kernel_sizes), n_classes, n_emo,
                             aux_dim, dropout)

    def forward(self, x, lens, aux):
        return self.forward_from_embeddings(self.emb(x), x, lens, aux)

    def forward_from_embeddings(self, embedded, x, lens, aux):
        e = self.drop(embedded).transpose(1, 2)
        feats = [F.relu(conv(e)).max(dim=2).values for conv in self.convs]
        return self.head(torch.cat(feats, dim=-1), aux)


class BiLSTMAttention(nn.Module):
    """Stacked BiLSTM with additive attention pooling.

    Separate single-layer LSTMs with hand-applied inter-layer dropout rather
    than a fused `nn.LSTM(num_layers=n, dropout=p)`: the state_dict keys depend
    on this, so do not "simplify" it.
    """

    def __init__(self, vocab_size, n_classes, n_emo, emb_dim=200, hidden=192,
                 layers=1, dropout=0.3, aux_dim=16):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.lstms = nn.ModuleList([
            nn.LSTM(emb_dim if i == 0 else hidden * 2, hidden, num_layers=1,
                    batch_first=True, bidirectional=True)
            for i in range(layers)])
        self.attn = nn.Linear(hidden * 2, 1)
        self.drop = nn.Dropout(dropout)
        self.head = _AuxHead(hidden * 2, n_classes, n_emo, aux_dim, dropout)

    def _encode(self, embedded, x, lens):
        packed = nn.utils.rnn.pack_padded_sequence(
            self.drop(embedded), lens.cpu(), batch_first=True, enforce_sorted=False)
        for i, lstm in enumerate(self.lstms):
            if i > 0:
                packed = packed._replace(data=self.drop(packed.data))
            packed, _ = lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed, batch_first=True, total_length=x.size(1))
        return out

    def forward(self, x, lens, aux):
        return self.forward_from_embeddings(self.emb(x), x, lens, aux)

    def forward_from_embeddings(self, embedded, x, lens, aux):
        out = self._encode(embedded, x, lens)
        pooled = _attention_pool(out, self.attn, x != PAD)
        return self.head(pooled, aux)

    def attention_weights(self, x, lens):
        out = self._encode(self.emb(x), x, lens)
        mask = (x != PAD)
        scores = self.attn(out).squeeze(-1).masked_fill(~mask, -1e4)
        return torch.softmax(scores, dim=1)


class BiGRUAttention(nn.Module):
    """Same shape as BiLSTMAttention with GRU cells -- fewer gates, less to fit."""

    def __init__(self, vocab_size, n_classes, n_emo, emb_dim=200, hidden=192,
                 layers=1, dropout=0.3, aux_dim=16):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.grus = nn.ModuleList([
            nn.GRU(emb_dim if i == 0 else hidden * 2, hidden, num_layers=1,
                   batch_first=True, bidirectional=True)
            for i in range(layers)])
        self.attn = nn.Linear(hidden * 2, 1)
        self.drop = nn.Dropout(dropout)
        self.head = _AuxHead(hidden * 2, n_classes, n_emo, aux_dim, dropout)

    def _encode(self, embedded, x, lens):
        packed = nn.utils.rnn.pack_padded_sequence(
            self.drop(embedded), lens.cpu(), batch_first=True, enforce_sorted=False)
        for i, gru in enumerate(self.grus):
            if i > 0:
                packed = packed._replace(data=self.drop(packed.data))
            packed, _ = gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed, batch_first=True, total_length=x.size(1))
        return out

    def forward(self, x, lens, aux):
        return self.forward_from_embeddings(self.emb(x), x, lens, aux)

    def forward_from_embeddings(self, embedded, x, lens, aux):
        out = self._encode(embedded, x, lens)
        pooled = _attention_pool(out, self.attn, x != PAD)
        return self.head(pooled, aux)

    def attention_weights(self, x, lens):
        out = self._encode(self.emb(x), x, lens)
        mask = (x != PAD)
        scores = self.attn(out).squeeze(-1).masked_fill(~mask, -1e4)
        return torch.softmax(scores, dim=1)


class CNNBiLSTM(nn.Module):
    """Conv1d -> max-pool/2 -> BiLSTM -> attention.

    The pooling halves the sequence, so the packing lengths and the attention
    mask are recomputed on the pooled grid rather than reused from the input.
    """

    def __init__(self, vocab_size, n_classes, n_emo, emb_dim=200, n_filters=128,
                 kernel_size=3, hidden=96, dropout=0.3, aux_dim=16):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.conv = nn.Conv1d(emb_dim, n_filters, kernel_size,
                              padding=kernel_size // 2)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(n_filters, hidden, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden * 2, 1)
        self.drop = nn.Dropout(dropout)
        self.head = _AuxHead(hidden * 2, n_classes, n_emo, aux_dim, dropout)

    def forward(self, x, lens, aux):
        return self.forward_from_embeddings(self.emb(x), x, lens, aux)

    def forward_from_embeddings(self, embedded, x, lens, aux):
        e = self.drop(embedded).transpose(1, 2)          # [B, emb, T]
        c = self.pool(F.relu(self.conv(e))).transpose(1, 2)  # [B, T//2, filters]

        pooled_total = c.size(1)
        pooled_lens = torch.clamp(torch.div(lens, 2, rounding_mode="floor"),
                                  min=1, max=pooled_total)
        packed = nn.utils.rnn.pack_padded_sequence(
            c, pooled_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed, batch_first=True, total_length=pooled_total)

        positions = torch.arange(pooled_total, device=c.device).unsqueeze(0)
        mask = positions < pooled_lens.unsqueeze(1)
        return self.head(_attention_pool(out, self.attn, mask), aux)


ARCHITECTURES = {
    "textcnn": TextCNN,
    "bilstm": BiLSTMAttention,
    "bigru": BiGRUAttention,
    "cnn_bilstm": CNNBiLSTM,
}


def build_model(name: str, vocab_size: int, n_classes: int, n_emo: int,
                params: Dict[str, object]) -> nn.Module:
    """Instantiate `name` with only the kwargs its constructor accepts."""
    import inspect

    cls = ARCHITECTURES[name]
    accepted = set(inspect.signature(cls.__init__).parameters)
    kwargs = {k: v for k, v in params.items() if k in accepted}
    return cls(vocab_size, n_classes, n_emo, **kwargs)
