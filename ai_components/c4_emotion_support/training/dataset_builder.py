"""
SUPERSEDED -- BUILDS THE OLD 7-LABEL FEATURE VECTOR. See training/README.md.

Data preparation for the forecaster now lives in ../../../../forcasting/preprocess.py,
which builds dialogue-context strings over an 8-label space with
conversation-level splits. Kept for provenance only.

Dataset builder for the C4 next-emotion forecaster.

Converts DailyDialog (multi-turn, emotion-labeled dialogue) into
(feature_vector, target_label) training pairs that are wire-compatible
with EmotionForecaster._build_features().

Feature vector — 15 floats, must stay in sync with _build_features():
  indices 0–6   : one-hot of current_emotion  (EMOTION_LABELS order)
  indices 7–13  : one-hot of previous_emotion (EMOTION_LABELS order)
  index  14     : deviation_level as float  {None→0.0, Low→0.25, Moderate→0.6, High→0.9}

Target — integer class index 0–6, ordered by EMOTION_LABELS:
  ["neutral", "joy", "sadness", "anger", "fear", "surprise", "disgust"]

Why DailyDialog?
  The system uses 7 emotion labels.  DailyDialog is the only widely-used
  multi-turn dialogue corpus whose 7-class emotion schema maps 1-to-1 to
  that label set (0=no_emotion→neutral, 1=anger, 2=disgust, 3=fear,
  4=happiness→joy, 5=sadness, 6=surprise).  Every training sample is a
  consecutive triplet of turns (t-1, t, t+1) so the model learns to
  predict the NEXT emotion given the current and prior emotional state.

Usage:
  python dataset_builder.py
  # writes processed_data/X_train.npy, y_train.npy, X_validation.npy, ...
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Allow running as a script from inside training/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    EMOTION_LABELS,
    HIGH_DEVIATION,
    LOW_DEVIATION,
    MODERATE_DEVIATION,
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    SAME_EMOTION,
)

# ---------------------------------------------------------------------------
# DailyDialog → system label mapping
# ---------------------------------------------------------------------------
DAILY_DIALOG_MAP: Dict[int, str] = {
    0: "neutral",   # no_emotion
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "joy",       # happiness
    5: "sadness",
    6: "surprise",
}

EMOTION_TO_IDX: Dict[str, int] = {lbl: i for i, lbl in enumerate(EMOTION_LABELS)}

DEVIATION_TO_FLOAT: Dict[str, float] = {
    "None": SAME_EMOTION,
    "Low": LOW_DEVIATION,
    "Moderate": MODERATE_DEVIATION,
    "High": HIGH_DEVIATION,
}


# ---------------------------------------------------------------------------
# Deviation logic — mirrors deviation_tracker.py exactly so training
# features are computed identically to inference features
# ---------------------------------------------------------------------------
def _emotion_group(emotion: str) -> str:
    if emotion in POSITIVE:
        return "positive"
    if emotion in NEGATIVE:
        return "negative"
    if emotion in NEUTRAL:
        return "neutral"
    return "other"


def compute_deviation(prev: Optional[str], curr: str) -> Tuple[str, float]:
    """Return (deviation_level, deviation_score) matching deviation_tracker."""
    if prev is None or prev == curr:
        return "None", SAME_EMOTION
    pg, cg = _emotion_group(prev), _emotion_group(curr)
    if pg == "neutral" or cg == "neutral":
        return "Low", LOW_DEVIATION
    if pg == "negative" and cg == "negative":
        return "Moderate", MODERATE_DEVIATION
    if (pg == "positive") != (cg == "positive"):   # one positive, one negative
        return "High", HIGH_DEVIATION
    return "Moderate", MODERATE_DEVIATION


# ---------------------------------------------------------------------------
# Feature builder — exact replica of EmotionForecaster._build_features()
# ---------------------------------------------------------------------------
def build_features(
    current_emotion: str,
    previous_emotion: Optional[str],
    deviation_level: str,
) -> List[float]:
    features: List[float] = []
    for lbl in EMOTION_LABELS:
        features.append(1.0 if lbl == current_emotion else 0.0)
    for lbl in EMOTION_LABELS:
        features.append(1.0 if lbl == previous_emotion else 0.0)
    features.append(DEVIATION_TO_FLOAT.get(deviation_level, 0.0))
    return features  # 15 floats


# ---------------------------------------------------------------------------
# Dialogue → sample conversion
# ---------------------------------------------------------------------------
def dialogues_to_samples(
    dialogues, split_name: str = "split"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each dialogue with N turns create N-2 samples:
      (emotion[t-1], emotion[t]) → emotion[t+1],  for t in [1, N-2]

    A minimum of 3 turns is required to produce even one sample.
    """
    X: List[List[float]] = []
    y: List[int] = []
    skipped = 0
    label_counts: Dict[str, int] = {lbl: 0 for lbl in EMOTION_LABELS}

    for dlg in dialogues:
        raw = dlg["emotion"]          # list[int]
        if len(raw) < 3:
            skipped += 1
            continue

        emotions = [DAILY_DIALOG_MAP.get(int(e), "neutral") for e in raw]

        for t in range(1, len(emotions) - 1):
            prev_emo = emotions[t - 1]
            curr_emo = emotions[t]
            next_emo = emotions[t + 1]

            deviation_level, _ = compute_deviation(prev_emo, curr_emo)
            features = build_features(curr_emo, prev_emo, deviation_level)
            target = EMOTION_TO_IDX[next_emo]

            X.append(features)
            y.append(target)
            label_counts[next_emo] += 1

    print(f"[{split_name}] samples={len(X):,}  skipped_short={skipped}")
    print(f"[{split_name}] label distribution: {label_counts}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_daily_dialog():
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Install HuggingFace datasets first:  pip install datasets"
        ) from exc
    return load_dataset("daily_dialog")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def build_and_save(output_dir: Optional[Path] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    if output_dir is None:
        output_dir = Path(__file__).parent / "processed_data"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading DailyDialog from HuggingFace …")
    dataset = load_daily_dialog()

    splits: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for split_name in ("train", "validation", "test"):
        if split_name not in dataset:
            continue
        X, y = dialogues_to_samples(dataset[split_name], split_name)
        splits[split_name] = (X, y)
        np.save(output_dir / f"X_{split_name}.npy", X)
        np.save(output_dir / f"y_{split_name}.npy", y)
        print(f"  saved  X_{split_name}.npy {X.shape}   y_{split_name}.npy {y.shape}")

    metadata = {
        "feature_dim": 15,
        "num_classes": len(EMOTION_LABELS),
        "emotion_labels": EMOTION_LABELS,
        "label_to_index": EMOTION_TO_IDX,
        "daily_dialog_to_system": {str(k): v for k, v in DAILY_DIALOG_MAP.items()},
        "feature_description": (
            "indices 0-6: one-hot current_emotion (EMOTION_LABELS order); "
            "indices 7-13: one-hot previous_emotion (EMOTION_LABELS order); "
            "index 14: deviation_level as float "
            "{None=0.0, Low=0.25, Moderate=0.6, High=0.9}"
        ),
        "source": "DailyDialog (Li et al., 2017) via HuggingFace datasets",
        "sample_construction": "triplet (turn t-1, turn t) -> turn t+1 for each turn t in [1, N-2]",
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll data saved to: {output_dir}")
    return splits


if __name__ == "__main__":
    build_and_save()
