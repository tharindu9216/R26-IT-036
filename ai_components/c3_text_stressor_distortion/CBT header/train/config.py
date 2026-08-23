"""
Configuration settings for the CBT cognitive-distortion (CDT) training pipeline.

This module stores all shared settings used across the project, including
dataset paths, output paths, model names, hyperparameters, augmentation
settings, Optuna settings, K-Fold settings, TF-IDF baseline parameters, and
device configuration.

The active Binary V4 pipeline predicts only whether a distortion is present.
TF-IDF Logistic Regression and Linear SVM provide traditional baselines.
BERT, MentalBERT, and DeBERTa-v3 are trained independently and combined with
an out-of-fold validated weighted soft-voting deployment ensemble.
"""
import os
from pathlib import Path
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[4]


class Config:

    # Paths (Modal overrides these via env vars)
    DATA_DIR = os.getenv(
        'DATA_DIR',
        str(PROJECT_ROOT / 'data/CBT header/processed')
    )
    OUTPUT_DIR = os.getenv(
        'OUTPUT_DIR',
        str(PROJECT_ROOT / 'models/c3_text_stressor_distortion/CBT header/binary_v4')
    )

    #  Reproducibility
    SEED = 42

    # Active transformer models. Longformer was intentionally removed from
    # the training pipeline because of its substantially higher runtime.
    TRANSFORMERS = ['BERT', 'MentalBERT', 'DeBERTa-v3']
    ML_BASELINES = ['LR', 'SVM']

    # None of the active models require a global attention mask.
    GLOBAL_ATTENTION_MODELS = set()

    #  Hyperparameters
    # Optuna will override per model — these are fallback defaults
    BASE_HYPERPARAMS = {
        'num_epochs'       : 10,
        'min_epochs'       : 5,
        'batch_size'       : 16,      # fallback outside Optuna
        'learning_rate'    : 1.5e-5,
        'dropout'          : 0.3,
        'head_dropout'     : 0.1,     # two-layer head inner dropout
        'intermediate'     : 256,     # two-layer head hidden size
        'label_smoothing'  : 0.1,
        'warmup_ratio'     : 0.1,
        'accum_steps'      : 4,       # gradient accumulation
        'lr_decay'         : 0.9,     # layer-wise LR decay factor
        'patience'         : 3,
        'weight_decay'     : 0.01,
        'max_grad_norm'    : 1.0,
        'head_lr_mult'     : 10.0,
        'type_loss_weight' : 1.0,
        'binary_threshold' : 0.5,
        'selection_metric' : 'f1_macro',
        'MAX_LEN'          : 192,     # used if metadata.json max_len not found
    }

    #  Text augmentation
    # Applied only to training splits, never validation/test.
    AUGMENTATION = {
        'enabled'          : True,
        'synonym_prob'     : 0.15,
        'max_replacements' : 1,
    }

    # Optional reviewed synthetic examples for the conditional type model.
    # The loader accepts only labels 1-10 and requires human_reviewed=True.
    # These rows are appended to type training only; validation/test and the
    # binary transformer always remain real-data-only.
    SYNTHETIC_DATA_PATH = os.getenv(
        'SYNTHETIC_DATA_PATH',
        os.path.join(DATA_DIR, 'cbt_synthetic_reviewed.csv'),
    )

    #  Optuna
    N_OPTUNA_TRIALS = 10
    OPTUNA_EPOCHS   = 2     # fast proxy — 2 epochs per trial
    # Optuna runs separately per model inside
    # binary_classification/train.py.

    #  K-Fold
    N_FOLDS = 5

    #  TF-IDF + ML Baseline settings
    TFIDF_PARAMS = {
        'max_features': 50000,
        'ngram_range' : (1, 2),
        'min_df'      : 2,
        'max_df'      : 0.95,
        'sublinear_tf': True,
        'lowercase'   : True,
    }
    LR_PARAMS  = {
        'C'           : 1.0,
        'max_iter'    : 1000,
        'class_weight': 'balanced',
        'solver'      : 'lbfgs',
    }
    SVM_PARAMS = {
        'C'           : 1.0,
        'max_iter'    : 2000,
        'class_weight': 'balanced',
        'dual'        : 'auto',
    }

    # Dedicated conditional distortion-type classifier.
    TYPE_TFIDF_PARAMS = {
        'max_features': 50000,
        'ngram_range' : (1, 2),
        'min_df'      : 2,
        'max_df'      : 0.95,
        'sublinear_tf': True,
        'lowercase'   : True,
    }
    TYPE_LR_PARAMS = {
        'max_iter'    : 2000,
        'class_weight': 'balanced',
        'solver'      : 'lbfgs',
    }
    TYPE_LR_C_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0]

    #  Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def get_amp_dtype():
        if not torch.cuda.is_available():
            return None
        return (torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16)

    @staticmethod
    def batch_size_choices(max_len):
        """Return the Optuna batch-size search space for active models."""
        configured = os.getenv('CBT_BATCH_SIZE_CHOICES')
        if configured:
            choices = sorted({
                int(value.strip())
                for value in configured.split(',')
                if value.strip()
            })
            if choices and all(value > 0 for value in choices):
                return choices
            raise ValueError(
                'CBT_BATCH_SIZE_CHOICES must contain positive integers')
        return [8, 16]
