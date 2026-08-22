
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from config import Config
from augmentation import synonym_replace


class StressDataset(Dataset):
    """
    Reads exactly what preprocessing saved:
        text_for_bert        → BERT
        text_for_mentalbert  → MentalBERT
        text_for_deberta     → DeBERTa-v3
        text                 → ML Baselines (raw)
        label                → Head 1A  (0=Not Stressed, 1=Stressed)
        subreddit_label      → Head 1B  (0-9 subreddit category)
    """

    def __init__(self, df, tokenizer, text_col, max_len, augment=False):
        self.texts     = df[text_col].fillna('').tolist()
        self.labels_1a = df['label'].tolist()
        self.labels_1b = df['subreddit_label'].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.augment   = augment

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
        return {
            'input_ids'      : enc['input_ids'].squeeze(0),
            'attention_mask' : enc['attention_mask'].squeeze(0),
            'label_1a'       : torch.tensor(self.labels_1a[idx], dtype=torch.long),
            'label_1b'       : torch.tensor(self.labels_1b[idx], dtype=torch.long),
        }


def make_loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=2, pin_memory=True)


def get_dataloaders(train_df, val_df, test_df,
                    tokenizer, text_col, max_len, batch_size):
    augment = Config.AUGMENTATION.get('enabled', False)
    return (
        make_loader(StressDataset(train_df, tokenizer, text_col, max_len,
                                  augment=augment),
                    batch_size, shuffle=True),
        make_loader(StressDataset(val_df,   tokenizer, text_col, max_len),
                    batch_size, shuffle=False),
        make_loader(StressDataset(test_df,  tokenizer, text_col, max_len),
                    batch_size, shuffle=False),
    )


def get_fold_loaders(full_dataset, train_idx, val_idx, batch_size):
    return (
        make_loader(Subset(full_dataset, train_idx), batch_size, shuffle=True),
        make_loader(Subset(full_dataset, val_idx),   batch_size, shuffle=False),
    )
