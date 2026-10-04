"""Fit and evaluate C3 ensemble temperature scaling.

The original held-out test split is deterministically repartitioned into a
calibration partition and a final evaluation partition.  Base checkpoints,
ensemble weights, and thresholds are locked before this script runs.  The
temperature is fitted only on the calibration partition; ECE, Brier score,
NLL, and the reliability diagram are reported only on the disjoint evaluation
partition.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.c3_text_stressor_distortion.app.calibration import (  # noqa: E402
    apply_temperature,
    calibration_metrics,
    fit_affine_temperature,
)
from backend.c3_text_stressor_distortion.app.cbt_model import (  # noqa: E402
    CBTPredictor,
)
from backend.c3_text_stressor_distortion.app.config import (  # noqa: E402
    CBT_CALIBRATION_PATH,
    STRESS_CALIBRATION_PATH,
)
from backend.c3_text_stressor_distortion.app.stress_model import (  # noqa: E402
    StressPredictor,
)


REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "c3_text_stressor_distortion"
    / "Calibration"
)


def _classification_metrics(labels, probabilities, threshold: float) -> dict:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def _plot_reliability(before: dict, after: dict, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot([0, 1], [0, 1], "--", color="#666666", label="Perfect calibration")
    for label, metrics, color in (
        ("Before", before, "#C44E52"),
        ("After temperature scaling", after, "#4C72B0"),
    ):
        rows = [row for row in metrics["reliability_bins"] if row["count"]]
        ax.plot(
            [row["mean_confidence"] for row in rows],
            [row["fraction_positive"] for row in rows],
            marker="o",
            linewidth=2,
            color=color,
            label=(
                f"{label} (ECE={metrics['ece']:.3f}, "
                f"Brier={metrics['brier_score']:.3f})"
            ),
        )
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_xlabel("Mean predicted positive probability")
    ax.set_ylabel("Observed positive frequency")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _task_spec(task: str):
    if task == "stress":
        frame = pd.read_csv(
            PROJECT_ROOT / "data/Stress header/processed/dreaddit_test.csv"
        )
        return {
            "predictor": StressPredictor(use_calibration=False),
            "texts": frame["text"].astype(str).tolist(),
            "labels": frame["label"].astype(int).to_numpy(),
            "threshold": 0.50,
            "artifact": STRESS_CALIBRATION_PATH,
            "report_dir": REPORT_ROOT / "stress",
            "title": "C3 Stress Ensemble Reliability",
        }
    if task == "cbt":
        frame = pd.read_csv(
            PROJECT_ROOT / "data/CBT header/processed/cbt_test.csv"
        )
        return {
            "predictor": CBTPredictor(use_calibration=False),
            "texts": frame["Patient Question"].astype(str).tolist(),
            "labels": (frame["label"].astype(int).to_numpy() != 0).astype(int),
            "threshold": 0.37,
            "artifact": CBT_CALIBRATION_PATH,
            "report_dir": REPORT_ROOT / "cbt",
            "title": "C3 Cognitive-Distortion Ensemble Reliability",
        }
    raise ValueError(f"Unknown task: {task}")


def run_task(
    task: str,
    *,
    calibration_fraction: float,
    seed: int,
    batch_size: int,
    n_bins: int,
    reuse_probabilities: bool,
) -> dict:
    spec = _task_spec(task)
    labels = spec["labels"]
    indices = np.arange(len(labels))
    evaluation_indices, calibration_indices = train_test_split(
        indices,
        test_size=calibration_fraction,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    cached_predictions = spec["report_dir"] / "calibration_predictions.csv"
    if reuse_probabilities and cached_predictions.exists():
        cached = pd.read_csv(cached_predictions)
        if not np.array_equal(cached["true_label"].to_numpy(dtype=int), labels):
            raise ValueError(f"Cached labels do not match {task} test data")
        raw_positive = cached["raw_positive_probability"].to_numpy(dtype=float)
        print(f"[{task}] reusing locked final-checkpoint probabilities", flush=True)
    else:
        print(
            f"[{task}] running final-checkpoint ensemble inference for "
            f"{len(labels)} held-out texts...",
            flush=True,
        )
        raw_matrix = spec["predictor"].predict_proba_many_raw(
            spec["texts"],
            batch_size=batch_size,
        ).numpy()
        raw_positive = raw_matrix[:, 1]
    temperature, bias = fit_affine_temperature(
        labels[calibration_indices],
        raw_positive[calibration_indices],
    )
    calibrated_positive = apply_temperature(raw_positive, temperature, bias)
    calibrated_threshold = float(
        apply_temperature([spec["threshold"]], temperature, bias)[0]
    )

    raw_decisions = raw_positive >= spec["threshold"]
    calibrated_decisions = calibrated_positive >= calibrated_threshold
    if not np.array_equal(raw_decisions, calibrated_decisions):
        raise AssertionError("Monotonic calibration unexpectedly changed decisions")

    evaluation_labels = labels[evaluation_indices]
    before = calibration_metrics(
        evaluation_labels,
        raw_positive[evaluation_indices],
        n_bins=n_bins,
    )
    after = calibration_metrics(
        evaluation_labels,
        calibrated_positive[evaluation_indices],
        n_bins=n_bins,
    )
    before.update(_classification_metrics(
        evaluation_labels,
        raw_positive[evaluation_indices],
        spec["threshold"],
    ))
    after.update(_classification_metrics(
        evaluation_labels,
        calibrated_positive[evaluation_indices],
        calibrated_threshold,
    ))

    report_dir = spec["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)
    reliability_path = report_dir / "reliability_diagram.png"
    _plot_reliability(before, after, reliability_path, spec["title"])

    partition = np.full(len(labels), "evaluation", dtype=object)
    partition[calibration_indices] = "calibration"
    prediction_frame = pd.DataFrame({
        "original_test_index": indices,
        "partition": partition,
        "true_label": labels,
        "raw_positive_probability": raw_positive,
        "calibrated_positive_probability": calibrated_positive,
        "raw_prediction": raw_decisions.astype(int),
        "calibrated_prediction": calibrated_decisions.astype(int),
    })
    prediction_frame.to_csv(report_dir / "calibration_predictions.csv", index=False)

    artifact = {
        "version": 1,
        "task": task,
        "method": "affine_temperature_scaling",
        "temperature": temperature,
        "bias": bias,
        "input_threshold": float(spec["threshold"]),
        "calibrated_threshold": calibrated_threshold,
        "fit_protocol": {
            "source": "deterministic calibration partition of original held-out test split",
            "random_seed": seed,
            "calibration_fraction": calibration_fraction,
            "calibration_samples": int(len(calibration_indices)),
            "evaluation_samples": int(len(evaluation_indices)),
            "stratified": True,
            "base_models_or_weights_refit": False,
            "evaluation_partition_used_for_fitting": False,
            "note": (
                "The old full-test result is retained as historical. For the "
                "calibration study, only the disjoint evaluation partition is "
                "used to report post-calibration metrics."
            ),
        },
        "evaluation": {
            "before": before,
            "after": after,
            "decisions_preserved": True,
        },
        "reliability_diagram": str(reliability_path.relative_to(PROJECT_ROOT)),
    }
    spec["artifact"].parent.mkdir(parents=True, exist_ok=True)
    spec["artifact"].write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    (report_dir / "calibration_report.json").write_text(
        json.dumps(artifact, indent=2),
        encoding="utf-8",
    )
    print(
        f"[{task}] T={temperature:.6f}, bias={bias:.6f}; "
        f"ECE {before['ece']:.4f}->{after['ece']:.4f}; "
        f"Brier {before['brier_score']:.4f}->{after['brier_score']:.4f}",
        flush=True,
    )
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=("stress", "cbt", "both"),
        default="both",
    )
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument(
        "--reuse-probabilities",
        action="store_true",
        help="Reuse the existing locked-checkpoint probability CSV.",
    )
    args = parser.parse_args()
    if not 0.05 <= args.calibration_fraction <= 0.50:
        parser.error("--calibration-fraction must be between 0.05 and 0.50")
    return args


def main() -> None:
    args = parse_args()
    tasks = ("stress", "cbt") if args.task == "both" else (args.task,)
    for task in tasks:
        run_task(
            task,
            calibration_fraction=args.calibration_fraction,
            seed=args.seed,
            batch_size=args.batch_size,
            n_bins=args.n_bins,
            reuse_probabilities=args.reuse_probabilities,
        )


if __name__ == "__main__":
    main()
