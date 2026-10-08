"""Lightweight metadata helpers for the Stress BERTopic Streamlit panel."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence


MIN_CLEAR_TOPIC_SIMILARITY = 0.45
MIN_CLEAR_TOPIC_MARGIN = 0.05


def display_topic_name(raw_name: str, topic_id: int) -> str:
    """Turn BERTopic's machine-generated name into a readable label."""
    if topic_id == -1:
        return "Outlier / no specific learned theme"
    name = re.sub(rf"^{re.escape(str(topic_id))}_", "", raw_name)
    return name.replace("_", " · ").strip().title()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing BERTopic report: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_topic_catalog(report_dir: str | Path) -> dict[str, Any]:
    """Load report CSVs into a compact topic lookup used by the UI."""
    report_dir = Path(report_dir)
    metrics_path = report_dir / "topic_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing BERTopic metrics: {metrics_path}")

    topics: dict[int, dict[str, Any]] = {}
    for row in _read_csv(report_dir / "topic_info.csv"):
        topic_id = int(row["Topic"])
        topics[topic_id] = {
            "topic_id": topic_id,
            "raw_name": row["Name"],
            "display_name": display_topic_name(row["Name"], topic_id),
            "training_documents": int(row["Count"]),
            "assigned_documents": int(row["Assigned_Count_All_Splits"]),
            "terms": [],
            "stress_distribution": {},
            "subreddit_counts": {},
        }

    for row in _read_csv(report_dir / "topic_terms.csv"):
        topic_id = int(row["topic"])
        if topic_id in topics:
            topics[topic_id]["terms"].append(row["term"])

    for row in _read_csv(report_dir / "topic_by_stress_label.csv"):
        topic_id = int(row["topic"])
        if topic_id in topics:
            topics[topic_id]["stress_distribution"][
                row["stress_label"]
            ] = {
                "documents": int(row["document_count"]),
                "percentage": float(row["percentage_within_topic"]),
            }

    for row in _read_csv(report_dir / "topic_by_subreddit.csv"):
        topic_id = int(row["topic"])
        if topic_id in topics:
            topics[topic_id]["subreddit_counts"][row["subreddit"]] = int(
                row["document_count"])

    return {
        "topics": topics,
        "metrics": json.loads(metrics_path.read_text()),
    }


def load_cbt_topic_catalog(report_dir: str | Path) -> dict[str, Any]:
    """Load CBT topic metadata and post-training label distributions."""
    report_dir = Path(report_dir)
    metrics_path = report_dir / "topic_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing BERTopic metrics: {metrics_path}")

    topics: dict[int, dict[str, Any]] = {}
    for row in _read_csv(report_dir / "topic_info.csv"):
        topic_id = int(row["Topic"])
        topics[topic_id] = {
            "topic_id": topic_id,
            "raw_name": row["Name"],
            "display_name": display_topic_name(row["Name"], topic_id),
            "training_documents": int(row["Count"]),
            "assigned_documents": int(row["Assigned_Count_All_Splits"]),
            "terms": [],
            "distortion_status_distribution": {},
            "distortion_type_distribution": {},
        }

    for row in _read_csv(report_dir / "topic_terms.csv"):
        topic_id = int(row["topic"])
        if topic_id in topics:
            topics[topic_id]["terms"].append(row["term"])

    for row in _read_csv(report_dir / "topic_by_distortion_status.csv"):
        topic_id = int(row["topic"])
        if topic_id in topics:
            topics[topic_id]["distortion_status_distribution"][
                row["distortion_status"]
            ] = {
                "documents": int(row["document_count"]),
                "percentage": float(row["percentage_within_topic"]),
            }

    for row in _read_csv(report_dir / "topic_by_distortion_type.csv"):
        topic_id = int(row["topic"])
        if topic_id in topics:
            topics[topic_id]["distortion_type_distribution"][
                row["distortion_type"]
            ] = {
                "documents": int(row["document_count"]),
                "percentage": float(row["percentage_within_topic"]),
            }

    return {
        "topics": topics,
        "metrics": json.loads(metrics_path.read_text()),
    }


def resolve_embedding_topic_ids(
        available_topic_ids: Iterable[int], embedding_count: int) -> list[int]:
    """Resolve safetensors row order for full or non-outlier exports."""
    all_topic_ids = sorted(int(topic_id) for topic_id in available_topic_ids)
    non_outlier_ids = [topic_id for topic_id in all_topic_ids if topic_id != -1]
    if embedding_count == len(all_topic_ids):
        return all_topic_ids
    if embedding_count == len(non_outlier_ids):
        return non_outlier_ids
    raise ValueError(
        "Topic embedding rows do not match topic metadata: "
        f"{embedding_count} rows for {len(all_topic_ids)} topics")


def rank_topic_scores(
        topic_ids: Sequence[int], scores: Sequence[float], top_n: int = 5,
) -> list[dict[str, float | int]]:
    """Rank non-outlier topic similarities in descending order."""
    if len(topic_ids) != len(scores):
        raise ValueError("Topic IDs and similarity scores must have equal length")
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    ranked = sorted(
        (
            {"topic_id": int(topic_id), "similarity": float(score)}
            for topic_id, score in zip(topic_ids, scores)
            if int(topic_id) != -1
        ),
        key=lambda item: item["similarity"],
        reverse=True,
    )
    return ranked[:top_n]


def assess_topic_match(
    ranked_topics: Sequence[dict[str, float | int]],
    minimum_similarity: float = MIN_CLEAR_TOPIC_SIMILARITY,
    minimum_margin: float = MIN_CLEAR_TOPIC_MARGIN,
) -> dict[str, Any]:
    """Give an interpretable, non-probabilistic status to a topic ranking."""
    if not ranked_topics:
        raise ValueError("At least one ranked topic is required")

    best_similarity = float(ranked_topics[0]["similarity"])
    runner_up_similarity = (
        float(ranked_topics[1]["similarity"])
        if len(ranked_topics) > 1
        else None
    )
    top_two_margin = (
        best_similarity - runner_up_similarity
        if runner_up_similarity is not None
        else None
    )
    weak = best_similarity < minimum_similarity
    ambiguous = (
        top_two_margin is not None
        and top_two_margin < minimum_margin
    )

    if weak:
        status = "Weak match"
        explanation = (
            f"No clear semantic topic: the closest candidate's similarity "
            f"({best_similarity:.3f}) is below the {minimum_similarity:.2f} "
            "interpretation threshold."
        )
        if ambiguous:
            explanation += (
                f" The top two candidates are also only "
                f"{top_two_margin:.3f} apart."
            )
    elif ambiguous:
        status = "Ambiguous match"
        explanation = (
            "No clear single topic: the closest candidates are only "
            f"{top_two_margin:.3f} apart, below the "
            f"{minimum_margin:.2f} interpretation margin."
        )
    else:
        status = "Clear match"
        explanation = (
            "The first-ranked theme is a clear primary semantic match under "
            "the interface's similarity and separation rules."
        )

    return {
        "status": status,
        "is_clear": not weak and not ambiguous,
        "is_weak": weak,
        "is_ambiguous": ambiguous,
        "best_similarity": best_similarity,
        "runner_up_similarity": runner_up_similarity,
        "top_two_margin": top_two_margin,
        "minimum_similarity": float(minimum_similarity),
        "minimum_margin": float(minimum_margin),
        "explanation": explanation,
    }


def largest_topics(
        catalog: dict[str, Any], limit: int = 15,
) -> list[dict[str, Any]]:
    """Return a table-ready view of the largest non-outlier topics."""
    topics = [
        topic
        for topic_id, topic in catalog["topics"].items()
        if topic_id != -1
    ]
    topics.sort(key=lambda topic: topic["training_documents"], reverse=True)
    return [
        {
            "Topic": topic["topic_id"],
            "Theme": topic["display_name"],
            "Training documents": topic["training_documents"],
            "All assigned documents": topic["assigned_documents"],
            "Top terms": ", ".join(topic["terms"][:5]),
        }
        for topic in topics[:limit]
    ]
