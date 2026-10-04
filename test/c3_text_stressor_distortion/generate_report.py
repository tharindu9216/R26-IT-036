"""Run real C3 sample-text tests and generate evaluation charts."""

from __future__ import annotations

import csv
import json
import sys
import time
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "report"


class PredictionResult(unittest.TextTestResult):
    """Record test status together with the model prediction fields."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def startTest(self, test):
        self._started_at = time.perf_counter()
        self._status = "passed"
        self._message = ""
        super().startTest(test)

    def addFailure(self, test, err):
        self._status = "failed"
        self._message = self._exc_info_to_string(err, test).splitlines()[-1]
        super().addFailure(test, err)

    def addError(self, test, err):
        self._status = "error"
        self._message = self._exc_info_to_string(err, test).splitlines()[-1]
        super().addError(test, err)

    def addSkip(self, test, reason):
        self._status = "skipped"
        self._message = reason
        super().addSkip(test, reason)

    def stopTest(self, test):
        test_id = test.id()
        self.records.append(
            {
                "header": "CBT" if "cbt_header" in test_id.lower() else "Stress",
                "case_id": getattr(test, "case_id", test_id.rsplit(".", 1)[-1]),
                "text": getattr(test, "sample_text", ""),
                "expected_label": getattr(test, "expected_label", ""),
                "predicted_label": getattr(test, "predicted_label", ""),
                "positive_probability": getattr(test, "positive_probability", ""),
                "confidence": getattr(test, "confidence", ""),
                "correct": self._status == "passed",
                "status": self._status,
                "duration_seconds": round(time.perf_counter() - self._started_at, 3),
                "message": self._message,
            }
        )
        super().stopTest(test)


def _metrics(rows: list[dict]) -> dict:
    labelled = [row for row in rows if row["expected_label"] != ""]
    tn = sum(row["expected_label"] == 0 and row["predicted_label"] == 0 for row in labelled)
    fp = sum(row["expected_label"] == 0 and row["predicted_label"] == 1 for row in labelled)
    fn = sum(row["expected_label"] == 1 and row["predicted_label"] == 0 for row in labelled)
    tp = sum(row["expected_label"] == 1 and row["predicted_label"] == 1 for row in labelled)
    total = len(labelled)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "total": total,
        "passed": tp + tn,
        "failed": fp + fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }


def _write_accuracy_chart(metrics: dict[str, dict]) -> None:
    headers = ("Stress", "CBT")
    values = [metrics[header]["accuracy"] for header in headers]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    bars = axis.bar(headers, values, color=("#1565c0", "#6a1b9a"), width=0.55)
    axis.set_ylim(0, 1.1)
    axis.set_ylabel("Accuracy")
    axis.set_title("Sample-Text Classification Accuracy")
    axis.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=4)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "accuracy_chart.png", dpi=200)
    plt.close(figure)


def _write_confusion_matrices(metrics: dict[str, dict]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    for axis, header in zip(axes, ("Stress", "CBT")):
        metric = metrics[header]
        matrix = np.asarray(
            [
                [metric["true_negative"], metric["false_positive"]],
                [metric["false_negative"], metric["true_positive"]],
            ]
        )
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
        for row_index in range(2):
            for column_index in range(2):
                axis.text(column_index, row_index, matrix[row_index, column_index],
                          ha="center", va="center", fontsize=18, fontweight="bold")
        axis.set_xticks((0, 1), ("Predicted 0", "Predicted 1"))
        axis.set_yticks((0, 1), ("Actual 0", "Actual 1"))
        axis.set_title(f"{header} header")
    figure.suptitle("Confusion Matrices")
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "confusion_matrix_chart.png", dpi=200)
    plt.close(figure)


def _write_probability_graph(rows: list[dict]) -> None:
    labelled = [row for row in rows if row["positive_probability"] != ""]
    labels = [row["case_id"] for row in labelled]
    probabilities = [float(row["positive_probability"]) for row in labelled]
    colors = ["#2e7d32" if row["correct"] else "#c62828" for row in labelled]
    figure, axis = plt.subplots(figsize=(8, 7))
    positions = np.arange(len(labelled))
    axis.barh(positions, probabilities, color=colors)
    axis.set_yticks(positions, labels)
    axis.set_xlim(0, 1)
    axis.set_xlabel("Positive-class probability")
    axis.set_title("Positive-Class Probability per Sample")
    axis.axvline(0.5, color="#455a64", linestyle="--", linewidth=1)
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "positive_probability_graph.png", dpi=200)
    plt.close(figure)


def _write_reports(rows: list[dict], successful: bool, elapsed: float) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fields = (
        "header", "case_id", "text", "expected_label", "predicted_label",
        "positive_probability", "confidence", "correct", "status",
        "duration_seconds", "message",
    )
    with (REPORT_DIR / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        header: _metrics([row for row in rows if row["header"] == header])
        for header in ("Stress", "CBT")
    }
    summary = {
        "successful": successful,
        "elapsed_seconds": round(elapsed, 3),
        "metrics": metrics,
    }
    (REPORT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_accuracy_chart(metrics)
    _write_confusion_matrices(metrics)
    _write_probability_graph(rows)


def main() -> int:
    suite = unittest.defaultTestLoader.discover(str(ROOT), pattern="test_*_header.py")
    runner = unittest.TextTestRunner(verbosity=2, resultclass=PredictionResult)
    started_at = time.perf_counter()
    result = runner.run(suite)
    elapsed = time.perf_counter() - started_at
    _write_reports(result.records, result.wasSuccessful(), elapsed)
    print(f"\nPrediction reports written to: {REPORT_DIR}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
