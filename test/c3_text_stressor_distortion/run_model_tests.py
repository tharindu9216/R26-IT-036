"""Evaluate every C3 classifier and generate per-header metric reports."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SUITES = {
    "cbt": ROOT / "CBT header",
    "stress": ROOT / "stress_header",
}
MODEL_ARGUMENTS = {
    "lr": "Logistic Regression",
    "svm": "SVM",
    "bert": "BERT",
    "mentalbert": "MentalBERT",
    "deberta": "DeBERTa-v3",
    "ensemble": "Ensemble",
}
ENSEMBLE_DETAILS = {
    "stress": {
        "name": "BERT + DeBERTa-v3 Probability Ensemble",
        "members": ["BERT", "DeBERTa-v3"],
        "weights": {"BERT": 0.5, "DeBERTa-v3": 0.5},
        "decision_threshold": 0.5,
        "selection": "Stress training pipeline unweighted probability mean",
    },
    "cbt": {
        "name": "OOF-Weighted Three-Transformer Ensemble",
        "members": ["BERT", "MentalBERT", "DeBERTa-v3"],
        "weights": {"BERT": 0.1, "MentalBERT": 0.4, "DeBERTa-v3": 0.5},
        "decision_threshold": 0.37,
        "selection": "Five-fold out-of-fold macro-F1 selection",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate 50 labelled cases against five individual models and "
            "the trained ensemble for each C3 header."
        )
    )
    parser.add_argument("--header", choices=("cbt", "stress", "all"), default="all")
    parser.add_argument(
        "--model",
        choices=(*MODEL_ARGUMENTS, "all"),
        default="all",
        help="evaluate one model or all five models plus the ensemble",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports",
        help="directory for CSV, JSON, and PNG reports",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return a failing exit code when any semantic case is misclassified",
    )
    return parser.parse_args()


def compute_metrics(
    expected: list[int], predicted: list[int], probabilities: list[float]
) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    labels = np.asarray(expected, dtype=int)
    predictions = np.asarray(predicted, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "passed": int((labels == predictions).sum()),
        "failed": int((labels != predictions).sum()),
        "total": int(len(labels)),
    }


def print_metrics_table(header: str, results: list[dict[str, Any]]) -> None:
    title = "CBT HEADER" if header == "cbt" else "STRESS HEADER"
    print(f"\n{'=' * 112}\n{title} — 50 CASES PER MODEL\n{'=' * 112}")
    print(
        f"{'Model':<22} {'Pass':>7} {'Acc':>7} {'BalAcc':>7} {'Prec':>7} "
        f"{'Recall':>7} {'F1':>7} {'F1-M':>7} {'MCC':>7} {'ROC':>7} {'PR':>7}"
    )
    print("-" * 112)
    for result in results:
        metrics = result["metrics"]
        print(
            f"{result['model']:<22} "
            f"{metrics['passed']:>2}/{metrics['total']:<4} "
            f"{metrics['accuracy']:>7.3f} "
            f"{metrics['balanced_accuracy']:>7.3f} "
            f"{metrics['precision']:>7.3f} "
            f"{metrics['recall']:>7.3f} "
            f"{metrics['f1']:>7.3f} "
            f"{metrics['f1_macro']:>7.3f} "
            f"{metrics['mcc']:>7.3f} "
            f"{metrics['roc_auc']:>7.3f} "
            f"{metrics['pr_auc']:>7.3f}"
        )
    if any(result["model"] == "Ensemble" for result in results):
        details = ENSEMBLE_DETAILS[header]
        weights = ", ".join(
            f"{member}={details['weights'][member]:.2f}"
            for member in details["members"]
        )
        print(
            f"\nEnsemble: {details['name']}\n"
            f"Members/weights: {weights}\n"
            f"Decision threshold: {details['decision_threshold']:.2f}\n"
            f"Method: {details['selection']}"
        )


def write_reports(
    output_dir: Path,
    header: str,
    results: list[dict[str, Any]],
) -> None:
    header_dir = output_dir / header
    header_dir.mkdir(parents=True, exist_ok=True)
    metric_fields = [
        "model",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "f1_macro",
        "mcc",
        "roc_auc",
        "pr_auc",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "passed",
        "failed",
        "total",
        "elapsed_seconds",
        "ensemble_members",
        "ensemble_weights",
        "decision_threshold",
    ]
    with (header_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
        for result in results:
            ensemble = ENSEMBLE_DETAILS[header] if result["model"] == "Ensemble" else None
            writer.writerow(
                {
                    "model": result["model"],
                    **result["metrics"],
                    "elapsed_seconds": round(result["elapsed_seconds"], 3),
                    "ensemble_members": (
                        "; ".join(ensemble["members"]) if ensemble else ""
                    ),
                    "ensemble_weights": (
                        "; ".join(
                            f"{member}={ensemble['weights'][member]:.2f}"
                            for member in ensemble["members"]
                        )
                        if ensemble
                        else ""
                    ),
                    "decision_threshold": (
                        ensemble["decision_threshold"] if ensemble else ""
                    ),
                }
            )

    prediction_fields = [
        "model",
        "case_id",
        "category",
        "expected_label",
        "predicted_label",
        "positive_probability",
        "correct",
        "text",
        "rationale",
    ]
    with (header_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields)
        writer.writeheader()
        for result in results:
            writer.writerows(result["predictions"])

    serializable = {
        "header": header,
        "cases_per_model": 50,
        "models_evaluated": [result["model"] for result in results],
        "ensemble": ENSEMBLE_DETAILS[header],
        "results": [
            {
                "model": result["model"],
                "metrics": result["metrics"],
                "elapsed_seconds": result["elapsed_seconds"],
            }
            for result in results
        ],
    }
    (header_dir / "metrics.json").write_text(
        json.dumps(serializable, indent=2) + "\n", encoding="utf-8"
    )


def save_graphs(
    output_dir: Path,
    header: str,
    results: list[dict[str, Any]],
    *,
    file_prefix: str = "",
    comparison_name: str = "Model",
    expand_primary_ensemble: bool = True,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import precision_recall_curve, roc_curve

    header_dir = output_dir / header
    header_dir.mkdir(parents=True, exist_ok=True)
    title = "CBT Header" if header == "cbt" else "Stress Header"
    ensemble_members = " + ".join(ENSEMBLE_DETAILS[header]["members"])
    model_names = []
    for result in results:
        if result.get("plot_label"):
            model_names.append(result["plot_label"])
        elif result["model"] == "Ensemble" and expand_primary_ensemble:
            model_names.append(f"Ensemble\n({ensemble_members})")
        else:
            model_names.append(result["model"])

    plotted_metrics = (
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("specificity", "Specificity"),
        ("roc_auc", "ROC-AUC"),
    )
    x = np.arange(len(model_names))
    width = 0.8 / len(plotted_metrics)
    fig, axis = plt.subplots(figsize=(15, 7))
    for index, (key, label) in enumerate(plotted_metrics):
        offset = (index - (len(plotted_metrics) - 1) / 2) * width
        axis.bar(
            x + offset,
            [result["metrics"][key] for result in results],
            width,
            label=label,
        )
    axis.set_title(f"{title} — {comparison_name} Metrics on 50 Semantic Cases")
    axis.set_ylabel("Score")
    axis.set_ylim(0.0, 1.05)
    axis.set_xticks(x, model_names, rotation=18, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(header_dir / f"{file_prefix}metrics_comparison.png", dpi=180)
    plt.close(fig)

    columns = 3
    rows = (len(results) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(14, 4.5 * rows), squeeze=False)
    for axis, result in zip(axes.flat, results):
        metrics = result["metrics"]
        matrix = np.asarray(
            [
                [metrics["true_negative"], metrics["false_positive"]],
                [metrics["false_negative"], metrics["true_positive"]],
            ]
        )
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=25)
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    fontsize=15,
                    color="white" if matrix[row, column] > 12 else "black",
                )
        model_title = result.get("plot_label", result["model"]).replace("\n", " ")
        if result["model"] == "Ensemble" and expand_primary_ensemble:
            model_title = f"Ensemble: {ensemble_members}"
        axis.set_title(f"{model_title}\nAcc {metrics['accuracy']:.2f}")
        axis.set_xticks((0, 1), ("Negative", "Positive"))
        axis.set_yticks((0, 1), ("Negative", "Positive"))
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("Expected label")
    for axis in axes.flat[len(results) :]:
        axis.axis("off")
    fig.suptitle(f"{title} — {comparison_name} Confusion Matrices", fontsize=16)
    fig.tight_layout()
    fig.savefig(header_dir / f"{file_prefix}confusion_matrices.png", dpi=180)
    plt.close(fig)

    fig, (roc_axis, pr_axis) = plt.subplots(1, 2, figsize=(14, 6))
    prevalence = 0.5
    for result in results:
        expected = [row["expected_label"] for row in result["predictions"]]
        scores = [row["positive_probability"] for row in result["predictions"]]
        false_positive_rate, true_positive_rate, _ = roc_curve(expected, scores)
        precision, recall, _ = precision_recall_curve(expected, scores)
        curve_name = result.get("plot_label", result["model"]).replace("\n", " ")
        if result["model"] == "Ensemble" and expand_primary_ensemble:
            curve_name = f"Ensemble: {ensemble_members}"
        roc_axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{curve_name} ({result['metrics']['roc_auc']:.2f})",
        )
        pr_axis.plot(
            recall,
            precision,
            label=f"{curve_name} ({result['metrics']['pr_auc']:.2f})",
        )
    roc_axis.plot((0, 1), (0, 1), "--", color="gray", label="Chance")
    roc_axis.set(title="ROC curves", xlabel="False positive rate", ylabel="True positive rate")
    pr_axis.axhline(prevalence, linestyle="--", color="gray", label="Chance")
    pr_axis.set(title="Precision–recall curves", xlabel="Recall", ylabel="Precision")
    for axis in (roc_axis, pr_axis):
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.05)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle(
        f"{title} — {comparison_name} Probability Ranking Performance",
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(header_dir / f"{file_prefix}roc_pr_curves.png", dpi=180)
    plt.close(fig)


def build_cbt_ensemble_comparisons(
    individual_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Combine cached individual predictions using five fixed strategies."""
    required = ("Logistic Regression", "SVM", "BERT", "MentalBERT", "DeBERTa-v3")
    by_model = {result["model"]: result for result in individual_results}
    missing = [name for name in required if name not in by_model]
    if missing:
        raise ValueError(
            "CBT ensemble comparison requires all five individual models; "
            f"missing: {', '.join(missing)}"
        )
    rows_by_model = {
        name: {row["case_id"]: row for row in by_model[name]["predictions"]}
        for name in required
    }
    case_ids = [row["case_id"] for row in by_model["BERT"]["predictions"]]
    transformer_members = ("BERT", "MentalBERT", "DeBERTa-v3")
    all_members = required
    strategies = (
        {
            "model": "OOF Weighted Soft Vote",
            "plot_label": "OOF Weighted Soft\n3 transformers",
            "members": transformer_members,
            "weights": {"BERT": 0.1, "MentalBERT": 0.4, "DeBERTa-v3": 0.5},
            "threshold": 0.37,
            "voting": "soft",
            "selection": "Saved five-fold OOF macro-F1 configuration",
        },
        {
            "model": "Equal Soft Vote (3 Transformers)",
            "plot_label": "Equal Soft Vote\n3 transformers",
            "members": transformer_members,
            "weights": {name: 1 / 3 for name in transformer_members},
            "threshold": 0.5,
            "voting": "soft",
            "selection": "Fixed equal weights; no selection on these cases",
        },
        {
            "model": "Hard Majority Vote (3 Transformers)",
            "plot_label": "Hard Majority\n3 transformers",
            "members": transformer_members,
            "weights": {name: 1 / 3 for name in transformer_members},
            "threshold": 0.5,
            "voting": "hard",
            "selection": "Majority of thresholded member predictions",
        },
        {
            "model": "Equal Soft Vote (All 5 Models)",
            "plot_label": "Equal Soft Vote\nall 5 models",
            "members": all_members,
            "weights": {name: 0.2 for name in all_members},
            "threshold": 0.5,
            "voting": "soft",
            "selection": "Fixed equal weights; no selection on these cases",
        },
        {
            "model": "Hard Majority Vote (All 5 Models)",
            "plot_label": "Hard Majority\nall 5 models",
            "members": all_members,
            "weights": {name: 0.2 for name in all_members},
            "threshold": 0.5,
            "voting": "hard",
            "selection": "Majority of thresholded member predictions",
        },
    )
    comparisons: list[dict[str, Any]] = []
    for strategy in strategies:
        prediction_rows = []
        for case_id in case_ids:
            source = rows_by_model["BERT"][case_id]
            if strategy["voting"] == "soft":
                probability = sum(
                    strategy["weights"][member]
                    * rows_by_model[member][case_id]["positive_probability"]
                    for member in strategy["members"]
                )
            else:
                probability = sum(
                    rows_by_model[member][case_id]["predicted_label"]
                    for member in strategy["members"]
                ) / len(strategy["members"])
            predicted_label = int(probability >= strategy["threshold"])
            prediction_rows.append(
                {
                    **source,
                    "model": strategy["model"],
                    "predicted_label": predicted_label,
                    "positive_probability": probability,
                    "correct": predicted_label == source["expected_label"],
                }
            )
        metrics = compute_metrics(
            [row["expected_label"] for row in prediction_rows],
            [row["predicted_label"] for row in prediction_rows],
            [row["positive_probability"] for row in prediction_rows],
        )
        comparisons.append(
            {
                **strategy,
                "metrics": metrics,
                "predictions": prediction_rows,
                "elapsed_seconds": 0.0,
            }
        )
    return comparisons


def write_cbt_ensemble_reports(
    output_dir: Path,
    comparisons: list[dict[str, Any]],
) -> None:
    header_dir = output_dir / "cbt"
    header_dir.mkdir(parents=True, exist_ok=True)
    metric_fields = [
        "method",
        "voting",
        "members",
        "weights",
        "decision_threshold",
        "selection",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "f1_macro",
        "mcc",
        "roc_auc",
        "pr_auc",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "passed",
        "failed",
        "total",
    ]
    with (header_dir / "ensemble_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
        for result in comparisons:
            writer.writerow(
                {
                    "method": result["model"],
                    "voting": result["voting"],
                    "members": "; ".join(result["members"]),
                    "weights": "; ".join(
                        f"{name}={weight:.4f}"
                        for name, weight in result["weights"].items()
                    ),
                    "decision_threshold": result["threshold"],
                    "selection": result["selection"],
                    **result["metrics"],
                }
            )
    prediction_fields = [
        "model",
        "case_id",
        "category",
        "expected_label",
        "predicted_label",
        "positive_probability",
        "correct",
        "text",
        "rationale",
    ]
    with (header_dir / "ensemble_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields)
        writer.writeheader()
        for result in comparisons:
            writer.writerows(
                {field: row[field] for field in prediction_fields}
                for row in result["predictions"]
            )
    payload = {
        "header": "cbt",
        "evaluation_note": (
            "These fixed strategies are compared on the 50 semantic cases. "
            "Only the OOF-weighted method was selected without using these cases."
        ),
        "results": [
            {
                "method": result["model"],
                "voting": result["voting"],
                "members": list(result["members"]),
                "weights": result["weights"],
                "decision_threshold": result["threshold"],
                "selection": result["selection"],
                "metrics": result["metrics"],
            }
            for result in comparisons
        ],
    }
    (header_dir / "ensemble_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def print_cbt_ensemble_table(comparisons: list[dict[str, Any]]) -> None:
    print(f"\n{'=' * 106}\nCBT ENSEMBLE METHOD COMPARISON\n{'=' * 106}")
    print(
        f"{'Method':<40} {'Pass':>7} {'Acc':>7} {'Prec':>7} {'Recall':>7} "
        f"{'F1':>7} {'F1-M':>7} {'MCC':>7} {'ROC':>7}"
    )
    print("-" * 106)
    for result in comparisons:
        metrics = result["metrics"]
        print(
            f"{result['model']:<40} {metrics['passed']:>2}/{metrics['total']:<4} "
            f"{metrics['accuracy']:>7.3f} {metrics['precision']:>7.3f} "
            f"{metrics['recall']:>7.3f} {metrics['f1']:>7.3f} "
            f"{metrics['f1_macro']:>7.3f} {metrics['mcc']:>7.3f} "
            f"{metrics['roc_auc']:>7.3f}"
        )


def evaluate_header(
    header: str,
    suite_dir: Path,
    model_names: list[str],
) -> list[dict[str, Any]]:
    from model_case_runner import load_cases, load_predictor

    cases = load_cases(suite_dir / "cases.csv")
    results: list[dict[str, Any]] = []
    for model_name in model_names:
        print(f"[{header}] Evaluating {model_name} on {len(cases)} cases...", flush=True)
        started = time.perf_counter()
        predictor = load_predictor(header, model_name)
        prediction_rows = []
        try:
            for case in cases:
                output = predictor.predict(case.text)
                prediction_rows.append(
                    {
                        "model": model_name,
                        "case_id": case.case_id,
                        "category": case.category,
                        "expected_label": case.expected_label,
                        "predicted_label": output.predicted_label,
                        "positive_probability": output.positive_probability,
                        "correct": output.predicted_label == case.expected_label,
                        "text": case.text,
                        "rationale": case.rationale,
                    }
                )
        finally:
            predictor.close()
        metrics = compute_metrics(
            [row["expected_label"] for row in prediction_rows],
            [row["predicted_label"] for row in prediction_rows],
            [row["positive_probability"] for row in prediction_rows],
        )
        results.append(
            {
                "model": model_name,
                "metrics": metrics,
                "predictions": prediction_rows,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    return results


def main() -> int:
    args = parse_args()
    os.environ["RUN_C3_MODEL_TESTS"] = "1"
    # These tests evaluate local checkpoints and should not perform metadata
    # requests on every model load. A missing Hugging Face scaffold now fails
    # immediately with a useful cache error instead of retrying the network.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    selected_headers = SUITES if args.header == "all" else {args.header: SUITES[args.header]}
    model_names = (
        list(MODEL_ARGUMENTS.values())
        if args.model == "all"
        else [MODEL_ARGUMENTS[args.model]]
    )
    any_failures = False
    for header, suite_dir in selected_headers.items():
        results = evaluate_header(header, suite_dir, model_names)
        print_metrics_table(header, results)
        write_reports(args.output_dir, header, results)
        save_graphs(args.output_dir, header, results)
        if header == "cbt" and args.model == "all":
            comparisons = build_cbt_ensemble_comparisons(results)
            print_cbt_ensemble_table(comparisons)
            write_cbt_ensemble_reports(args.output_dir, comparisons)
            save_graphs(
                args.output_dir,
                "cbt",
                comparisons,
                file_prefix="ensemble_",
                comparison_name="Ensemble Method",
                expand_primary_ensemble=False,
            )
            any_failures = any_failures or any(
                result["metrics"]["failed"] for result in comparisons
            )
        header_output = (args.output_dir / header).resolve()
        print(f"\nReports and graphs: {header_output}")
        any_failures = any_failures or any(
            result["metrics"]["failed"] for result in results
        )
    return 1 if args.strict and any_failures else 0


if __name__ == "__main__":
    sys.exit(main())
