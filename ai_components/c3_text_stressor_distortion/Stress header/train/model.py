
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModel


# =============================================================================
# Two-Layer Classification Head (spec: Linear→GELU→Dropout→Linear)
# =============================================================================
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


# =============================================================================
# Dual-Head Stress Model
# =============================================================================
class DualHeadStressModel(nn.Module):
    """
    Shared encoder + LayerNorm + Dropout → two ClassificationHeads.

    Head 1A → Stress binary    (0=Not Stressed, 1=Stressed)
    Head 1B → Subreddit class  (0-9)

    Loss = 0.6 × loss_1a  +  0.4 × loss_1b
    """

    def __init__(self, hf_id,
                 num_subreddit_labels=10,
                 num_binary_labels=2,
                 dropout=0.3,
                 head_dropout=0.1,
                 intermediate=256):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(hf_id)
        hidden          = self.encoder.config.hidden_size

        self.layer_norm = nn.LayerNorm(hidden)
        self.dropout    = nn.Dropout(dropout)

        self.head_1a    = ClassificationHead(
            hidden, intermediate, num_binary_labels,    head_dropout)
        self.head_1b    = ClassificationHead(
            hidden, intermediate, num_subreddit_labels, head_dropout)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]   # CLS token
        cls = self.layer_norm(cls)             # LayerNorm
        cls = self.dropout(cls)                # Dropout
        return self.head_1a(cls), self.head_1b(cls)


# =============================================================================
# Layer-wise LR Decay Optimizer (spec: layer_lr = base_lr × 0.9^(12-i))
# =============================================================================
def get_layerwise_optimizer(model, base_lr, lr_decay=0.9,
                             head_lr_mult=10.0, weight_decay=0.01):
    """
    Lower layers  → small LR  (general language features)
    Higher layers → base_lr   (task-specific features)
    Heads         → base_lr × head_lr_mult  (trained from scratch)

    No weight decay on: bias, LayerNorm.weight, layer_norm.weight
    """
    NO_DECAY = {'bias', 'LayerNorm.weight', 'layer_norm.weight'}
    encoder  = model.encoder
    groups   = []

    try:
        num_layers = encoder.config.num_hidden_layers
    except AttributeError:
        num_layers = 12

    # Embeddings — lowest LR
    try:
        emb_params = list(encoder.embeddings.named_parameters())
        groups += [
            {'params': [p for n, p in emb_params if not any(nd in n for nd in NO_DECAY)],
             'lr': base_lr * (lr_decay ** num_layers), 'weight_decay': weight_decay},
            {'params': [p for n, p in emb_params if     any(nd in n for nd in NO_DECAY)],
             'lr': base_lr * (lr_decay ** num_layers), 'weight_decay': 0.0},
        ]
    except AttributeError:
        pass

    # Transformer layers — layer_lr = base_lr × lr_decay^(num_layers - i)
    try:
        layers = encoder.encoder.layer      # BERT / MentalBERT
    except AttributeError:
        try:
            layers = encoder.encoder.layers # DeBERTa-v3
        except AttributeError:
            layers = []

    for i, layer in enumerate(layers):
        layer_lr     = base_lr * (lr_decay ** (num_layers - i - 1))
        layer_params = list(layer.named_parameters())
        groups += [
            {'params': [p for n, p in layer_params if not any(nd in n for nd in NO_DECAY)],
             'lr': layer_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in layer_params if     any(nd in n for nd in NO_DECAY)],
             'lr': layer_lr, 'weight_decay': 0.0},
        ]

    # Pooler
    try:
        groups.append({
            'params': list(encoder.pooler.parameters()),
            'lr': base_lr, 'weight_decay': weight_decay})
    except AttributeError:
        pass

    # model LayerNorm + heads — highest LR
    head_params = (list(model.layer_norm.named_parameters()) +
                   list(model.head_1a.named_parameters()) +
                   list(model.head_1b.named_parameters()))
    groups += [
        {'params': [p for n, p in head_params if not any(nd in n for nd in NO_DECAY)],
         'lr': base_lr * head_lr_mult, 'weight_decay': weight_decay},
        {'params': [p for n, p in head_params if     any(nd in n for nd in NO_DECAY)],
         'lr': base_lr * head_lr_mult, 'weight_decay': 0.0},
    ]

    return AdamW(groups)
