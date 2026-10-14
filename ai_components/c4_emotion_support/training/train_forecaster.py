"""
SUPERSEDED -- DOES NOT PRODUCE A MODEL THE DEMO CAN LOAD. See training/README.md.

The forecaster is now trained by ../../../../forcasting/train_deep.py and loaded
from a .pt checkpoint. This script targets the old 7-label one-hot contract and
the `EmotionForecaster._build_features` method, neither of which still exists.
Kept for provenance only.

Train the C4 next-emotion forecaster and save it as
  models/next_emotion_forecaster/forecaster.joblib

The saved model must satisfy the EmotionForecaster.predict() contract:
  - sklearn-compatible (has predict_proba or predict)
  - predict_proba(X)[0] returns 7 floats in EMOTION_LABELS order
    (guaranteed when training targets are integer indices 0-6 in
     EMOTION_LABELS order and all 7 classes appear in training data)

Run order:
  1. python dataset_builder.py      # builds training/processed_data/
  2. python train_forecaster.py     # trains and saves model

Usage:
  python train_forecaster.py [--data-dir PATH] [--model gradient_boosting|random_forest|logistic_regression]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import EMOTION_LABELS, NEXT_EMOTION_MODEL_PATH

OUTPUT_PATH = Path(__file__).resolve().parent.parent / NEXT_EMOTION_MODEL_PATH


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_splits(data_dir: Optional[Path] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    if data_dir is None:
        data_dir = Path(__file__).parent / "processed_data"
    data_dir = Path(data_dir)

    splits: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name in ("train", "validation", "test"):
        xp = data_dir / f"X_{name}.npy"
        yp = data_dir / f"y_{name}.npy"
        if xp.exists() and yp.exists():
            splits[name] = (np.load(xp), np.load(yp))

    if "train" not in splits:
        raise FileNotFoundError(
            f"No training data found in {data_dir}.\n"
            "Run  python dataset_builder.py  first."
        )
    return splits


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------
def make_models() -> Dict:
    return {
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            multi_class="multinomial",
            solver="lbfgs",
            C=1.0,
            random_state=42,
        ),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def print_class_balance(y: np.ndarray, label: str) -> None:
    counts = {EMOTION_LABELS[i]: int((y == i).sum()) for i in range(len(EMOTION_LABELS))}
    print(f"  [{label}] class counts: {counts}")


def train_all(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    target_model: Optional[str] = None,
) -> Tuple[object, str, Dict[str, float]]:
    X_train, y_train = splits["train"]
    X_val, y_val = splits.get("validation", splits.get("test", splits["train"]))
    X_test, y_test = splits.get("test", (X_val, y_val))

    print(f"\nTraining samples : {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Test samples      : {len(X_test):,}")
    print_class_balance(y_train, "train")

    # Validate that all 7 classes appear — required for correct predict_proba ordering
    missing = set(range(len(EMOTION_LABELS))) - set(np.unique(y_train))
    if missing:
        missing_names = [EMOTION_LABELS[i] for i in missing]
        print(f"WARNING: these emotion classes are absent from training data: {missing_names}")
        print("  predict_proba column order will be wrong for those classes.")
        print("  Consider augmenting or balancing the dataset.\n")

    candidates = make_models()
    if target_model:
        if target_model not in candidates:
            raise ValueError(f"Unknown model '{target_model}'. Choose from: {list(candidates)}")
        candidates = {target_model: candidates[target_model]}

    best_model, best_name, best_acc = None, "", -1.0
    val_accs: Dict[str, float] = {}

    for name, model in candidates.items():
        print(f"\nTraining  {name} …")
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        acc = float(accuracy_score(y_val, val_pred))
        val_accs[name] = acc
        print(f"  val accuracy = {acc:.4f}")
        if acc > best_acc:
            best_acc, best_model, best_name = acc, model, name

    print(f"\nBest: {best_name}  (val_acc={best_acc:.4f})")

    # Final test evaluation
    test_pred = best_model.predict(X_test)
    test_acc = float(accuracy_score(y_test, test_pred))
    print(f"Test accuracy = {test_acc:.4f}\n")
    print("Classification report:")
    print(
        classification_report(
            y_test, test_pred,
            labels=list(range(len(EMOTION_LABELS))),
            target_names=EMOTION_LABELS,
            zero_division=0,
        )
    )

    return best_model, best_name, val_accs


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_model(model: object, model_name: str, val_accs: Dict[str, float]) -> None:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    model_file = OUTPUT_PATH / "forecaster.joblib"
    joblib.dump(model, model_file)
    print(f"Model saved → {model_file}")

    info = {
        "model_type": model_name,
        "emotion_labels": EMOTION_LABELS,
        "label_to_index": {lbl: i for i, lbl in enumerate(EMOTION_LABELS)},
        "feature_dim": 15,
        "num_classes": len(EMOTION_LABELS),
        "val_accuracies": val_accs,
        "feature_description": (
            "indices 0-6: one-hot current_emotion; "
            "indices 7-13: one-hot previous_emotion; "
            "index 14: deviation_level float"
        ),
        "note": (
            "predict_proba columns are ordered by model.classes_ which must equal "
            "[0,1,2,3,4,5,6] for correct mapping to EMOTION_LABELS. "
            "EmotionForecaster.predict() resolves this via model.classes_."
        ),
    }
    with open(OUTPUT_PATH / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"Metadata saved → {OUTPUT_PATH / 'training_info.json'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train C4 emotion forecaster")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--model",
        choices=["gradient_boosting", "random_forest", "logistic_regression"],
        default=None,
        help="Train only this model (default: try all three, pick best by val accuracy)",
    )
    args = parser.parse_args()

    print("=== C4 Emotion Forecaster — Training ===")
    splits = load_splits(args.data_dir)
    best_model, best_name, val_accs = train_all(splits, target_model=args.model)
    save_model(best_model, best_name, val_accs)
    print("\nDone. Load the model by placing forecaster.joblib in")
    print(f"  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
