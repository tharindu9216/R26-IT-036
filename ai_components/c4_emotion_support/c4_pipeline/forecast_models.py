"""Architectures and text encoding for the next-emotion forecaster.

These are byte-for-byte compatible re-declarations of the classes in
`../../../forcasting/train_deep.py` and the text helpers in
`../../../forcasting/preprocess.py`. They are duplicated here rather than
imported so the demo stays self-contained (the training tree lives outside this
repository), but the two must be kept in sync: any change to layer names or to
the tokenizer regex silently breaks `load_state_dict` or, worse, degrades
predictions without erroring.
"""

import re
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, UNK = 0, 1

# ------------------------------------------------------------------ text utils
_URL = re.compile(r"https?://\S+|www\.\S+")
_WS = re.compile(r"\s+")
_TOK = re.compile(r"[a-z0-9']+|[<\[][a-z]+[>\]]|[!?.,]")


def clean_text(text: str) -> str:
    """Mirrors preprocess.clean_text."""
    value = str(text)
    value = _URL.sub(" <url> ", value)
    value = re.sub(r"(.)\1{3,}", r"\1\1\1", value)  # loooove -> loove
    return _WS.sub(" ", value).strip()


def simple_tokenize(text: str) -> List[str]:
    """Mirrors train_deep.simple_tokenize."""
    return _TOK.findall(str(text).lower())


def encode(text: str, vocab: Dict[str, int], max_len: int) -> Tuple[List[int], int]:
    """Mirrors train_deep.encode: truncate to max_len, right-pad with PAD."""
    ids = [vocab.get(token, UNK) for token in simple_tokenize(text)][:max_len]
    return ids + [PAD] * (max_len - len(ids)), min(len(ids), max_len)


def build_context_text(
    history: Sequence[Tuple[str, str]],
    current_message: str,
    context_turns: int = 3,
) -> str:
    """Rebuild the `context_text` field the forecaster was trained on.

    `history` is an ordered sequence of (speaker, text) for the turns that came
    before `current_message`; speakers must be "user" or "supporter", matching
    the training corpus. Format (from preprocess.build_context):

        [spk] t-3 [spk] t-2 [spk] t-1 [SEP] [user] current

    With no history the string is just `[user] current`, exactly as the first
    turn of every training conversation was encoded.
    """
    current = f"[user] {clean_text(current_message)}"
    if context_turns <= 0 or not history:
        return current
    recent = list(history)[-context_turns:]
    prefix = " ".join(f"[{speaker}] {clean_text(text)}" for speaker, text in recent)
    return f"{prefix} [SEP] {current}".strip()


# ------------------------------------------------------------------ models
class BiLSTMAttention(nn.Module):
    """Stacked BiLSTM with additive attention pooling.

    Separate single-layer LSTMs with hand-applied inter-layer dropout, not a
    fused `nn.LSTM(num_layers=2, dropout=p)` -- see the note in train_deep.py.
    The state_dict keys depend on this, so do not "simplify" it.
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
        self.aux_emb = nn.Embedding(n_emo + 1, aux_dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2 + aux_dim, n_classes)

    def forward(self, x, lens, aux):
        return self.forward_from_embeddings(self.emb(x), x, lens, aux)

    def forward_from_embeddings(self, embedded, x, lens, aux):
        """Split out so Integrated Gradients can inject interpolated embeddings."""
        mask = (x != PAD)
        e = self.drop(embedded)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lens.cpu(), batch_first=True, enforce_sorted=False)
        for i, lstm in enumerate(self.lstms):
            if i > 0:
                packed = packed._replace(data=self.drop(packed.data))
            packed, _ = lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed, batch_first=True,
                                                  total_length=x.size(1))
        scores = self.attn(out).squeeze(-1).masked_fill(~mask, -1e4)
        w = torch.softmax(scores, dim=1).unsqueeze(-1)
        ctx = (out * w).sum(1)
        h = torch.cat([ctx, self.aux_emb(aux)], dim=-1)
        return self.fc(self.drop(h))

    def attention_weights(self, x, lens):
        """Attention distribution over tokens -- a free, native explanation."""
        mask = (x != PAD)
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lens.cpu(), batch_first=True, enforce_sorted=False)
        for lstm in self.lstms:
            packed, _ = lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed, batch_first=True,
                                                  total_length=x.size(1))
        scores = self.attn(out).squeeze(-1).masked_fill(~mask, -1e4)
        return torch.softmax(scores, dim=1)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, n_classes, n_emo, emb_dim=200, n_filters=128,
                 kernel_sizes=(2, 3, 4, 5), dropout=0.4, aux_dim=16):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, n_filters, k, padding=k - 1) for k in kernel_sizes])
        self.aux_emb = nn.Embedding(n_emo + 1, aux_dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes) + aux_dim, n_classes)

    def forward(self, x, lens, aux):
        return self.forward_from_embeddings(self.emb(x), x, lens, aux)

    def forward_from_embeddings(self, embedded, x, lens, aux):
        e = self.drop(embedded).transpose(1, 2)
        feats = [F.relu(conv(e)).max(dim=2).values for conv in self.convs]
        h = torch.cat(feats + [self.aux_emb(aux)], dim=-1)
        return self.fc(self.drop(h))


class DistilBertClassifier(nn.Module):
    def __init__(self, model_name, n_classes, n_emo, dropout=0.2, aux_dim=16):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name)
        hid = self.encoder.config.hidden_size
        self.aux_emb = nn.Embedding(n_emo + 1, aux_dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hid + aux_dim, n_classes)

    def forward(self, input_ids, attn, aux):
        out = self.encoder(input_ids=input_ids, attention_mask=attn)
        cls = out.last_hidden_state[:, 0]
        h = torch.cat([cls, self.aux_emb(aux)], dim=-1)
        return self.fc(self.drop(h))

    def forward_from_embeddings(self, embedded, attn, aux):
        out = self.encoder(inputs_embeds=embedded, attention_mask=attn)
        cls = out.last_hidden_state[:, 0]
        h = torch.cat([cls, self.aux_emb(aux)], dim=-1)
        return self.fc(self.drop(h))
