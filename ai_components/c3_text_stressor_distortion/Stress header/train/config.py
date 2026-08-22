
import os
from pathlib import Path
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[4]


class Config:

    # ── Paths (Modal overrides these via env vars) ────────────────────────────
    DATA_DIR = os.getenv(
        'DATA_DIR',
        str(PROJECT_ROOT / 'data/Stress header/processed')
    )
    OUTPUT_DIR = os.getenv(
        'OUTPUT_DIR',
        str(PROJECT_ROOT / 'models/c3_text_stressor_distortion/Stress header')
    )

    # ── Reproducibility ───────────────────────────────────────────────────────
    SEED = 42

    # ── Models — matched exactly to preprocessing MODEL_REGISTRY ─────────────
    TRANSFORMERS = ['BERT', 'MentalBERT', 'DeBERTa-v3']
    ML_BASELINES = ['LR', 'SVM']

    # ── Hyperparameters ───────────────────────────────────────────────────────
    # Optuna will override per model — these are fallback defaults
    BASE_HYPERPARAMS = {
        'num_epochs'       : 10,
        'batch_size'       : 16,      # Modal auto-patches based on GPU
        'learning_rate'    : 1.5e-5,
        'dropout'          : 0.3,
        'head_dropout'     : 0.1,     # two-layer head inner dropout
        'intermediate'     : 256,     # two-layer head hidden size
        'label_smoothing'  : 0.1,
        'warmup_ratio'     : 0.1,
        'alpha'            : 0.6,     # Head 1A loss weight
        'accum_steps'      : 4,       # gradient accumulation
        'lr_decay'         : 0.9,     # layer-wise LR decay factor
        'patience'         : 2,
        'weight_decay'     : 0.01,
        'max_grad_norm'    : 1.0,
        'head_lr_mult'     : 10.0,
        'MAX_LEN'          : 192,     # used if metadata.json max_len not found
    }

    # ── Text augmentation ────────────────────────────────────────────────────
    # Applied only to training splits, never validation/test.
    AUGMENTATION = {
        'enabled'          : True,
        'synonym_prob'     : 0.15,
        'max_replacements' : 1,
    }

    # ── Optuna ────────────────────────────────────────────────────────────────
    N_OPTUNA_TRIALS = 10
    OPTUNA_EPOCHS   = 2     # fast proxy — 2 epochs per trial
    # NOTE: Optuna runs separately per model inside train.py — NOT here

    # ── K-Fold ────────────────────────────────────────────────────────────────
    N_FOLDS = 5

    # ── TF-IDF + ML Baseline settings ────────────────────────────────────────
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
    }

    # ── Device ────────────────────────────────────────────────────────────────
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def get_amp_dtype():
        if not torch.cuda.is_available():
            return None
        return (torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16)
