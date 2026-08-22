"""Model architectures for transformer-based cognitive distortion detection.

This module defines a hierarchical transformer model for CDT (Cognitive
Distortion Detection). A shared encoder feeds two classification heads:

* binary head: No Distortion vs Distortion
* type head: one of 10 distortion types, conditional on Distortion

It also defines a layer-wise optimizer so pretrained transformer layers are
updated slowly and the newly added classification head learns faster.
"""

import re

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModel


# Two-Layer Classification Head (spec: Linear→GELU→Dropout→Linear)
class ClassificationHead(nn.Module):
    """
    768 → 256 → GELU → Dropout(0.1) → num_classes

    GELU is smoother than ReLU — better gradient flow for transformers.
    """
    def __init__(self, hidden_size, intermediate, num_classes,
                 head_dropout=0.1):
        super().__init__()
        self.fc1     = nn.Linear(hidden_size, intermediate)
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(head_dropout)
        self.fc2     = nn.Linear(intermediate, num_classes)

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


# Hierarchical CDT Model
class CDTModel(nn.Module):
    """
    Shared encoder + LayerNorm + Dropout → two classification heads.
    """

    def __init__(self, hf_id,
                 num_types=10,
                 dropout=0.3,
                 head_dropout=0.1,
                 intermediate=256):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(hf_id)
        hidden          = self.encoder.config.hidden_size

        self.layer_norm = nn.LayerNorm(hidden)
        self.dropout    = nn.Dropout(dropout)

        self.binary_head = ClassificationHead(
            hidden, intermediate, 2, head_dropout)
        self.type_head = ClassificationHead(
            hidden, intermediate, num_types, head_dropout)

    def forward(self, input_ids, attention_mask, global_attention_mask=None):
        if global_attention_mask is not None:
            # Longformer only — other encoders don't accept this argument.
            out = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask,
                               global_attention_mask=global_attention_mask)
        else:
            out = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]   # CLS / <s> token
        cls = self.layer_norm(cls)             # LayerNorm
        cls = self.dropout(cls)                # Dropout
        return self.binary_head(cls), self.type_head(cls)


class BinaryCDTModel(nn.Module):
    """Dedicated Distortion vs No-Distortion transformer.

    The distortion-type decision is deliberately kept outside this model.
    This prevents the sparse 10-class objective from competing with the
    binary objective in a shared encoder.
    """

    def __init__(self, hf_id, dropout=0.3, head_dropout=0.1,
                 intermediate=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(hf_id)
        hidden = self.encoder.config.hidden_size
        self.layer_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.binary_head = ClassificationHead(
            hidden, intermediate, 2, head_dropout)

    def forward(self, input_ids, attention_mask, global_attention_mask=None):
        if global_attention_mask is not None:
            out = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
            )
        else:
            out = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        cls = out.last_hidden_state[:, 0, :]
        cls = self.dropout(self.layer_norm(cls))
        return self.binary_head(cls)


# Layer-wise LR Decay Optimizer (spec: layer_lr = base_lr × 0.9^(12-i))
def get_layerwise_optimizer(model, base_lr, lr_decay=0.9,
                             head_lr_mult=10.0, weight_decay=0.01):
    """
    Lower layers  → small LR  (general language features)
    Higher layers → base_lr   (task-specific features)
    Head          → base_lr × head_lr_mult  (trained from scratch)

    No weight decay on: bias, LayerNorm.weight, layer_norm.weight

    Works for BERT/MentalBERT/DeBERTa-v3. These expose `.encoder.layer`
    (BERT-style naming) or `.encoder.layers` (DeBERTa-v3).
    """
    num_layers = getattr(model.encoder.config, 'num_hidden_layers', 12)
    grouped = {}

    def learning_rate_for(name):
        if name.startswith(('binary_head.', 'type_head.', 'layer_norm.')):
            return base_lr * head_lr_mult
        if name.startswith('encoder.embeddings.'):
            return base_lr * (lr_decay ** num_layers)

        layer_match = re.match(
            r'^encoder\.encoder\.layers?\.(\d+)\.',
            name,
        )
        if layer_match:
            layer_index = int(layer_match.group(1))
            depth = max(0, num_layers - layer_index - 1)
            return base_lr * (lr_decay ** depth)

        # Poolers, relative-position embeddings, and any future encoder-level
        # trainable parameters receive the base encoder LR instead of being
        # silently omitted from the optimizer.
        return base_lr

    def uses_weight_decay(name):
        lowered = name.lower()
        return not (
            name.endswith('.bias')
            or 'layernorm.weight' in lowered
            or 'layer_norm.weight' in lowered
        )

    trainable = {
        id(parameter): (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    for name, parameter in (
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad):
        lr = learning_rate_for(name)
        decay = weight_decay if uses_weight_decay(name) else 0.0
        grouped.setdefault((lr, decay), []).append(parameter)

    groups = [
        {'params': parameters, 'lr': lr, 'weight_decay': decay}
        for (lr, decay), parameters in grouped.items()
    ]
    grouped_ids = [
        id(parameter)
        for group in groups
        for parameter in group['params']
    ]
    missing_ids = set(trainable) - set(grouped_ids)
    duplicate_count = len(grouped_ids) - len(set(grouped_ids))
    if missing_ids or duplicate_count:
        missing_names = [trainable[item][0] for item in missing_ids]
        raise RuntimeError(
            'Optimizer parameter coverage failed: '
            f'missing={missing_names}, duplicates={duplicate_count}'
        )

    return AdamW(groups, lr=base_lr)
