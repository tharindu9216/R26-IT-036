"""Configuration for the independent distortion-type SLM experiments."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[5]


class TypeSLMConfig:
    # Recommended primary SLM. Override for controlled comparison experiments.
    BASE_MODEL = os.getenv(
        'TYPE_SLM_MODEL',
        'microsoft/Phi-4-mini-instruct',
    )

    DATA_DIR = Path(os.getenv(
        'DATA_DIR',
        str(PROJECT_ROOT / 'data/CBT header/processed'),
    ))
    OUTPUT_DIR = Path(os.getenv(
        'TYPE_SLM_OUTPUT_DIR',
        str(
            PROJECT_ROOT
            / 'models/c3_text_stressor_distortion/CBT header/type_slm'
        ),
    ))

    # Stage 1: supervised task adaptation.
    SFT_METHOD = 'QLoRA'
    MAX_SEQUENCE_LENGTH = 2048
    LORA_RANK = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05

    # Stage 2: preferred post-training method for this small dataset.
    PREFERENCE_METHOD = 'DPO'

    # Optional research ablation when genuine policy-gradient RL is required.
    FULL_RL_METHOD = 'GRPO'

    SEED = 42

