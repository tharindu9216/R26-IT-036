"""Dataset and DataLoader utilities for CBT cognitive-distortion (CDT) classification.

This module converts preprocessed text data into PyTorch Dataset and DataLoader
objects for model training, validation, and testing.

Each sample is tokenized using the selected transformer tokenizer and returns
three targets: the original end-to-end 11-class label, a binary
No-Distortion/Distortion label, and a conditional 10-class distortion-type
label. The type target uses -100 for No Distortion so it can be excluded from
the conditional loss.

Text augmentation is applied only to the training data when enabled in Config.
Validation and test data are always kept unchanged.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset

from config import Config
from augmentation import synonym_replace


class CDTDataset(Dataset):
    """
    Reads exactly what cbt_preprocessing.ipynb saved:
        text_for_bert        → BERT
        text_for_mentalbert  → MentalBERT
        text_for_deberta     → DeBERTa-v3
        Patient Question     → ML Baselines (raw)
        label                → end-to-end target (0-10)

    Derived targets:
        binary_label         → 0=No Distortion, 1=Distortion
        type_label           → 0-9 for original labels 1-10; -100 otherwise
    """

    def __init__(self, df, tokenizer, text_col, max_len,
                 augment=False, needs_global_attention=False):
        self.texts    = df[text_col].fillna('').tolist()
        self.labels   = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.augment   = augment
        self.needs_global_attention = needs_global_attention

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.augment:
            text = synonym_replace(
                text,
                probability=Config.AUGMENTATION.get('synonym_prob', 0.15),
                max_replacements=Config.AUGMENTATION.get('max_replacements', 1),
            )

        enc = self.tokenizer(
            text,
            max_length     = self.max_len,
            padding        = 'max_length',
            truncation     = True,
            return_tensors = 'pt',
        )
        label = int(self.labels[idx])
        item = {
            'input_ids'      : enc['input_ids'].squeeze(0),
            'attention_mask' : enc['attention_mask'].squeeze(0),
            'label'          : torch.tensor(label, dtype=torch.long),
            'binary_label'   : torch.tensor(
                int(label != 0), dtype=torch.long),
            'type_label'     : torch.tensor(
                label - 1 if label != 0 else -100, dtype=torch.long),
        }

        if self.needs_global_attention:
            # Global attention on the first token ([CLS]/<s>) only — the
            # standard recipe for Longformer sequence classification.
            global_mask    = torch.zeros_like(item['input_ids'])
            global_mask[0] = 1
            item['global_attention_mask'] = global_mask

        return item


def make_loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=2, pin_memory=True)


def get_dataloaders(train_df, val_df, test_df,
                    tokenizer, text_col, max_len, batch_size,
                    needs_global_attention=False):
    augment = Config.AUGMENTATION.get('enabled', False)
    return (
        make_loader(CDTDataset(train_df, tokenizer, text_col, max_len,
                               augment=augment,
                               needs_global_attention=needs_global_attention),
                    batch_size, shuffle=True),
        make_loader(CDTDataset(val_df, tokenizer, text_col, max_len,
                               needs_global_attention=needs_global_attention),
                    batch_size, shuffle=False),
        make_loader(CDTDataset(test_df, tokenizer, text_col, max_len,
                               needs_global_attention=needs_global_attention),
                    batch_size, shuffle=False),
    )


def get_fold_loaders(full_dataset, train_idx, val_idx, batch_size):
    return (
        make_loader(Subset(full_dataset, train_idx), batch_size, shuffle=True),
        make_loader(Subset(full_dataset, val_idx),   batch_size, shuffle=False),
    )
