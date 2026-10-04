"""Inference configuration for the C3 text-stress ensemble."""

from __future__ import annotations

import os

from settings import C3_BASE_MODELS_DIR as BASE_MODELS_DIR
from settings import C3_STRESS_DIR as STRESS_DIR

DEVICE = os.getenv("C3_DEVICE", "auto")
METADATA_PATH = STRESS_DIR / "metadata.json"
DEFAULT_NUM_SUBREDDITS = 10
STRESS_DECISION_THRESHOLD = 0.5
STRESS_CALIBRATION_PATH = STRESS_DIR / "stress_ensemble_calibration.json"

STRESS_ENSEMBLE_MEMBERS = {
    "BERT": {
        "checkpoint": STRESS_DIR / "BERT_final.pt",
        "model_dir": BASE_MODELS_DIR / "bert-base-uncased",
        "max_len": 192,
        "weight": 0.5,
    },
    "DeBERTa-v3": {
        "checkpoint": STRESS_DIR / "DeBERTa-v3_final.pt",
        "model_dir": BASE_MODELS_DIR / "deberta-v3-base",
        "max_len": 192,
        "weight": 0.5,
    },
}


def validate_config() -> None:
    """Fail early when a required local artifact is absent or misconfigured."""

    if abs(sum(item["weight"] for item in STRESS_ENSEMBLE_MEMBERS.values()) - 1.0) > 1e-9:
        raise ValueError("Stress ensemble weights must sum to 1.0")
    if not 0.0 <= STRESS_DECISION_THRESHOLD <= 1.0:
        raise ValueError("Stress decision threshold must be between 0 and 1")
    required = [METADATA_PATH, STRESS_CALIBRATION_PATH]
    for name, item in STRESS_ENSEMBLE_MEMBERS.items():
        required.extend((item["checkpoint"], item["model_dir"] / "config.json"))
        required.append(
            item["model_dir"]
            / ("spm.model" if name == "DeBERTa-v3" else "vocab.txt")
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing C3 stress artifacts: " + ", ".join(missing))
