"""Inference configuration for the C3 cognitive-distortion ensemble."""

from __future__ import annotations

import json
import os

from settings import C3_BASE_MODELS_DIR as BASE_MODELS_DIR
from settings import C3_CBT_DIR as CBT_DIR

DEVICE = os.getenv("C3_DEVICE", "auto")
CBT_BINARY_CONFIG_PATH = CBT_DIR / "binary_config.json"
CBT_CALIBRATION_PATH = CBT_DIR / "binary_ensemble_calibration.json"

CBT_MODEL_DIRS = {
    "BERT": BASE_MODELS_DIR / "bert-base-uncased",
    # MentalBERT was initialized from BERT and its full fine-tuned encoder is
    # already inside the checkpoint, so the BERT architecture/tokenizer is
    # sufficient to reconstruct it entirely offline.
    "MentalBERT": BASE_MODELS_DIR / "bert-base-uncased",
    "DeBERTa-v3": BASE_MODELS_DIR / "deberta-v3-base",
}


def load_binary_config() -> dict:
    if not CBT_BINARY_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing CBT config: {CBT_BINARY_CONFIG_PATH}")
    return json.loads(CBT_BINARY_CONFIG_PATH.read_text(encoding="utf-8"))


def validate_config() -> None:
    """Fail early when a required local artifact is absent or misconfigured."""

    config = load_binary_config()
    expected = {"BERT", "MentalBERT", "DeBERTa-v3"}
    models = config.get("models", {})
    weights = config.get("deployment", {}).get("weights", {})
    if set(models) != expected or set(weights) != expected:
        raise ValueError("CBT ensemble must contain BERT, MentalBERT, and DeBERTa-v3")
    if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-9:
        raise ValueError("CBT ensemble weights must sum to 1.0")
    threshold = float(config["deployment"]["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("CBT decision threshold must be between 0 and 1")

    required = [CBT_BINARY_CONFIG_PATH, CBT_CALIBRATION_PATH]
    for name, item in models.items():
        required.extend(
            (
                CBT_DIR / item["checkpoint_file"],
                CBT_MODEL_DIRS[name] / "config.json",
                CBT_MODEL_DIRS[name]
                / ("spm.model" if name == "DeBERTa-v3" else "vocab.txt"),
            )
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing C3 CBT artifacts: " + ", ".join(missing))
