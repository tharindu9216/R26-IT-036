"""Compare Stress/CBT late-fusion policies on a labelled validation CSV.

Required CSV columns: ``stress_probability``, ``cbt_probability``, ``label``.
The binary label must be 0/1 and should represent a human-annotated text-only
support need (or a dedicated C3 fusion target), not a final route label already
affected by sensor/emotion inputs. This script is evaluation-only: never tune
on the held-out test set.

Example:
    python evaluate_fusion.py validation_router_labels.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from emotion_chain.fusion import fuse_stress_and_cbt


@dataclass(frozen=True)
class Example:
    stress_probability: float
    cbt_probability: float
    label: bool


def load_examples(path: Path) -> list[Example]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"stress_probability", "cbt_probability", "label"}
    if not rows:
        raise ValueError("Validation CSV is empty")
    missing = required - set(rows[0])
    if missing:
        raise ValueError("Missing CSV columns: " + ", ".join(sorted(missing)))

    examples = []
    for row_number, row in enumerate(rows, start=2):
        try:
            label_value = int(row["label"])
            if label_value not in {0, 1}:
                raise ValueError("label must be 0 or 1")
            examples.append(
                Example(
                    stress_probability=float(row["stress_probability"]),
                    cbt_probability=float(row["cbt_probability"]),
                    label=bool(label_value),
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid row {row_number}: {exc}") from exc
    return examples


def binary_metrics(labels: list[bool], predictions: list[bool]) -> dict[str, float]:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("labels and predictions must have equal non-zero length")
    tp = sum(label and prediction for label, prediction in zip(labels, predictions))
    tn = sum(not label and not prediction for label, prediction in zip(labels, predictions))
    fp = sum(not label and prediction for label, prediction in zip(labels, predictions))
    fn = sum(label and not prediction for label, prediction in zip(labels, predictions))

    def class_f1(true_positive: int, false_positive: int, false_negative: int) -> float:
        denominator = 2 * true_positive + false_positive + false_negative
        return 0.0 if denominator == 0 else 2 * true_positive / denominator

    positive_f1 = class_f1(tp, fp, fn)
    negative_f1 = class_f1(tn, fn, fp)
    return {
        "accuracy": (tp + tn) / len(labels),
        "precision": 0.0 if tp + fp == 0 else tp / (tp + fp),
        "recall": 0.0 if tp + fn == 0 else tp / (tp + fn),
        "f1_positive": positive_f1,
        "f1_macro": (positive_f1 + negative_f1) / 2.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate(
    examples: list[Example],
    *,
    method: str,
    fusion_threshold: float,
    stress_threshold: float,
    cbt_threshold: float,
    stress_weight: float = 0.5,
    cbt_weight: float = 0.5,
) -> dict[str, float]:
    predictions = [
        fuse_stress_and_cbt(
            item.stress_probability,
            item.cbt_probability,
            method=method,
            threshold=fusion_threshold,
            stress_weight=stress_weight,
            cbt_weight=cbt_weight,
            stress_threshold=stress_threshold,
            cbt_threshold=cbt_threshold,
        ).is_flagged
        for item in examples
    ]
    return binary_metrics([item.label for item in examples], predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--stress-threshold", type=float, default=0.5)
    parser.add_argument("--cbt-threshold", type=float, default=0.37)
    args = parser.parse_args()
    examples = load_examples(args.csv_path)

    results: dict[str, dict] = {}
    for method in ("or", "stress_only", "cbt_only"):
        results[method] = evaluate(
            examples,
            method=method,
            fusion_threshold=0.5,
            stress_threshold=args.stress_threshold,
            cbt_threshold=args.cbt_threshold,
        )

    weighted_candidates = []
    for weight_index in range(1, 10):
        stress_weight = weight_index / 10.0
        cbt_weight = 1.0 - stress_weight
        for threshold_index in range(5, 96, 5):
            threshold = threshold_index / 100.0
            metrics = evaluate(
                examples,
                method="weighted_average",
                fusion_threshold=threshold,
                stress_threshold=args.stress_threshold,
                cbt_threshold=args.cbt_threshold,
                stress_weight=stress_weight,
                cbt_weight=cbt_weight,
            )
            weighted_candidates.append(
                {
                    "stress_weight": stress_weight,
                    "cbt_weight": cbt_weight,
                    "threshold": threshold,
                    **metrics,
                }
            )
    results["weighted_average_best"] = max(
        weighted_candidates,
        key=lambda item: (item["f1_macro"], item["f1_positive"]),
    )
    results["n_examples"] = {"count": len(examples)}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
