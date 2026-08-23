"""Evaluate 50 unsupervised BERTopic cases per C3 header and plot results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SUITES = {
    "cbt": ROOT / "CBT header",
    "stress": ROOT / "stress_header",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run 50 semantic assignment checks against each saved BERTopic model. "
            "Because BERTopic is unsupervised this reports topic coverage and "
            "assignment quality rather than classification accuracy."
        )
    )
    parser.add_argument("--header", choices=("cbt", "stress", "all"), default="all")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports",
        help="parent directory for topic CSV JSON and PNG reports",
    )
    return parser.parse_args()


def evaluate_header(header: str, suite_dir: Path) -> dict[str, Any]:
    from topic_case_runner import (
        LoadedTopicPredictor,
        load_topic_cases,
        validate_topic_output,
    )

    cases = load_topic_cases(suite_dir / "bertopic_cases.csv")
    print(f"[{header}] Evaluating BERTopic on {len(cases)} cases...", flush=True)
    predictor = LoadedTopicPredictor(header)
    rows: list[dict[str, Any]] = []
    try:
        for index, case in enumerate(cases, start=1):
            output = predictor.predict(case.text)
            validate_topic_output(output, predictor.available_topic_ids)
            rows.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "topic_id": output.topic_id,
                    "theme": output.theme,
                    "similarity": output.similarity,
                    "status": output.status,
                    "is_clear": output.is_clear,
                    "is_weak": output.is_weak,
                    "is_ambiguous": output.is_ambiguous,
                    "runner_up_similarity": output.runner_up_similarity,
                    "top_two_margin": output.top_two_margin,
                    "terms": "; ".join(output.terms),
                    "elapsed_ms": output.elapsed_ms,
                    "text": case.text,
                    "rationale": case.rationale,
                }
            )
            if index % 10 == 0:
                print(f"[{header}] {index}/{len(cases)} cases complete", flush=True)

        # A small repeatability sample catches accidental stochastic inference
        # without doubling the cost of the full 50-case evaluation.
        repeatability_rows = []
        for case, first in zip(cases[:5], rows[:5]):
            repeated = predictor.predict(case.text)
            repeatability_rows.append(
                {
                    "case_id": case.case_id,
                    "same_topic": repeated.topic_id == first["topic_id"],
                    "similarity_difference": abs(
                        repeated.similarity - float(first["similarity"])
                    ),
                }
            )
        available_topic_count = len(predictor.available_topic_ids)
        minimum_similarity = predictor.minimum_similarity
    finally:
        predictor.close()

    similarities = [float(row["similarity"]) for row in rows]
    latencies = [float(row["elapsed_ms"]) for row in rows]
    margins = [
        float(row["top_two_margin"])
        for row in rows
        if row["top_two_margin"] is not None
    ]
    topic_counts = Counter(int(row["topic_id"]) for row in rows)
    status_counts = Counter(str(row["status"]) for row in rows)
    repeatable = sum(
        row["same_topic"] and row["similarity_difference"] <= 1e-6
        for row in repeatability_rows
    )
    metrics = {
        "model_type": "unsupervised_nearest_topic_centroid",
        "evaluation_note": (
            "Case categories are coverage descriptors and not ground-truth topic "
            "labels. Accuracy precision recall and F1 are therefore not applicable."
        ),
        "total_cases": len(rows),
        "valid_assignments": len(rows),
        "valid_assignment_rate": len(rows) / len(cases),
        "available_topics_excluding_outlier": available_topic_count,
        "unique_topics_assigned": len(topic_counts),
        "topic_coverage_rate": (
            len(topic_counts) / available_topic_count if available_topic_count else 0.0
        ),
        "minimum_clear_similarity": minimum_similarity,
        "clear_matches": status_counts["Clear match"],
        "clear_match_rate": status_counts["Clear match"] / len(rows),
        "weak_matches": status_counts["Weak match"],
        "weak_match_rate": status_counts["Weak match"] / len(rows),
        "ambiguous_matches": status_counts["Ambiguous match"],
        "ambiguous_match_rate": status_counts["Ambiguous match"] / len(rows),
        "mean_similarity": statistics.fmean(similarities),
        "median_similarity": statistics.median(similarities),
        "minimum_similarity": min(similarities),
        "maximum_similarity": max(similarities),
        "mean_top_two_margin": statistics.fmean(margins) if margins else None,
        "mean_latency_ms": statistics.fmean(latencies),
        "median_latency_ms": statistics.median(latencies),
        "maximum_latency_ms": max(latencies),
        "repeatability_cases": len(repeatability_rows),
        "repeatable_cases": repeatable,
        "repeatability_rate": repeatable / len(repeatability_rows),
    }
    return {
        "header": header,
        "metrics": metrics,
        "predictions": rows,
        "repeatability": repeatability_rows,
        "topic_counts": dict(sorted(topic_counts.items())),
    }


def print_metrics(result: dict[str, Any]) -> None:
    header = result["header"].upper()
    metrics = result["metrics"]
    print(f"\n{'=' * 76}\n{header} BERTOPIC — 50 UNSUPERVISED CASES\n{'=' * 76}")
    print(f"Valid assignments : {metrics['valid_assignments']}/{metrics['total_cases']}")
    print(
        "Topic coverage    : "
        f"{metrics['unique_topics_assigned']}/"
        f"{metrics['available_topics_excluding_outlier']} "
        f"({metrics['topic_coverage_rate']:.1%})"
    )
    print(
        "Match outcomes    : "
        f"clear={metrics['clear_matches']} ({metrics['clear_match_rate']:.1%}), "
        f"weak={metrics['weak_matches']} ({metrics['weak_match_rate']:.1%}), "
        f"ambiguous={metrics['ambiguous_matches']} "
        f"({metrics['ambiguous_match_rate']:.1%})"
    )
    print(
        "Similarity        : "
        f"mean={metrics['mean_similarity']:.3f}, "
        f"median={metrics['median_similarity']:.3f}, "
        f"range={metrics['minimum_similarity']:.3f}–"
        f"{metrics['maximum_similarity']:.3f}"
    )
    print(
        "Inference latency : "
        f"mean={metrics['mean_latency_ms']:.1f} ms, "
        f"median={metrics['median_latency_ms']:.1f} ms, "
        f"max={metrics['maximum_latency_ms']:.1f} ms"
    )
    print(
        "Repeatability     : "
        f"{metrics['repeatable_cases']}/{metrics['repeatability_cases']}"
    )
    print("Accuracy/F1       : N/A (BERTopic is unsupervised; no manual topic labels)")


def write_reports(output_dir: Path, result: dict[str, Any]) -> Path:
    report_dir = output_dir / result["header"] / "bertopic"
    report_dir.mkdir(parents=True, exist_ok=True)
    prediction_fields = [
        "case_id",
        "category",
        "topic_id",
        "theme",
        "similarity",
        "status",
        "is_clear",
        "is_weak",
        "is_ambiguous",
        "runner_up_similarity",
        "top_two_margin",
        "terms",
        "elapsed_ms",
        "text",
        "rationale",
    ]
    with (report_dir / "topic_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields)
        writer.writeheader()
        writer.writerows(result["predictions"])

    metric_fields = ("metric", "value")
    with (report_dir / "topic_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
        for key, value in result["metrics"].items():
            writer.writerow({"metric": key, "value": value})

    (report_dir / "topic_metrics.json").write_text(
        json.dumps(
            {
                "header": result["header"],
                "metrics": result["metrics"],
                "topic_counts": result["topic_counts"],
                "repeatability": result["repeatability"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return report_dir


def save_graphs(report_dir: Path, result: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = result["predictions"]
    metrics = result["metrics"]
    title = "CBT" if result["header"] == "cbt" else "Stress"

    topic_rows: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        topic_rows[(int(row["topic_id"]), str(row["theme"]))].append(row)
    ordered = sorted(topic_rows.items(), key=lambda item: len(item[1]), reverse=True)
    labels = [f"{topic_id}: {theme[:32]}" for (topic_id, theme), _ in ordered]
    counts = [len(values) for _, values in ordered]
    figure_height = max(6.0, len(labels) * 0.34)
    fig, axis = plt.subplots(figsize=(12, figure_height))
    positions = np.arange(len(labels))
    axis.barh(positions, counts, color="#4776b4")
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Number of 50 semantic cases")
    axis.set_title(f"{title} BERTopic — Assigned Topic Distribution")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(report_dir / "topic_distribution.png", dpi=180)
    plt.close(fig)

    similarities = [float(row["similarity"]) for row in rows]
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.hist(similarities, bins=12, color="#4c9f70", edgecolor="white")
    axis.axvline(
        metrics["minimum_clear_similarity"],
        color="#c43d3d",
        linestyle="--",
        label=f"Clear threshold ({metrics['minimum_clear_similarity']:.2f})",
    )
    axis.axvline(
        metrics["mean_similarity"],
        color="#2e4057",
        linestyle=":",
        label=f"Mean ({metrics['mean_similarity']:.3f})",
    )
    axis.set_xlabel("Cosine similarity to nearest saved topic centroid")
    axis.set_ylabel("Cases")
    axis.set_title(f"{title} BERTopic — Similarity Distribution")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "similarity_distribution.png", dpi=180)
    plt.close(fig)

    status_order = ("Clear match", "Weak match", "Ambiguous match")
    status_counts = Counter(str(row["status"]) for row in rows)
    fig, (status_axis, latency_axis) = plt.subplots(1, 2, figsize=(13, 5.5))
    status_axis.bar(
        status_order,
        [status_counts[status] for status in status_order],
        color=("#3b8d61", "#d0923e", "#8264a8"),
    )
    status_axis.set_ylim(0, 50)
    status_axis.set_ylabel("Cases")
    status_axis.set_title("Assignment clarity")
    status_axis.grid(axis="y", alpha=0.25)
    latency_axis.hist(
        [float(row["elapsed_ms"]) for row in rows],
        bins=12,
        color="#5676a5",
        edgecolor="white",
    )
    latency_axis.set_xlabel("Inference time (ms)")
    latency_axis.set_ylabel("Cases")
    latency_axis.set_title("Per-case CPU latency")
    latency_axis.grid(axis="y", alpha=0.25)
    fig.suptitle(f"{title} BERTopic — Clarity and Runtime")
    fig.tight_layout()
    fig.savefig(report_dir / "clarity_and_latency.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    os.environ["RUN_C3_BERTOPIC_TESTS"] = "1"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/c3-matplotlib")
    selected = SUITES if args.header == "all" else {args.header: SUITES[args.header]}
    for header, suite_dir in selected.items():
        result = evaluate_header(header, suite_dir)
        print_metrics(result)
        report_dir = write_reports(args.output_dir, result)
        save_graphs(report_dir, result)
        print(f"Reports and graphs: {report_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
