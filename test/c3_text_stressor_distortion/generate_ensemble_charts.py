"""Evaluate each C3 ensemble member and save one chart per header.

This is an explicit model test runner rather than an automatically discovered
unit test because it loads the real transformer checkpoints.  It reuses the
sample cases from ``test_stress_header.py`` and ``test_cbt_header.py``.
"""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
REPORT_DIR = ROOT / "report" / "ensemble"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.c3_text_stressor_distortion.app import cbt_model, stress_model
from backend.c3_text_stressor_distortion.app.config import (
    CBT_BINARY_CONFIG_PATH,
    CBT_RUNTIME_HF_ID_OVERRIDES,
    STRESS_DECISION_THRESHOLD,
    STRESS_ENSEMBLE_MEMBERS,
)
from test_cbt_header import CBT_CASES
from test_stress_header import STRESS_CASES


METRIC_NAMES = ("Accuracy", "Precision", "Recall", "F1")


def _cached_model_path(model_id: str) -> str:
    """Prefer a complete local Hugging Face snapshot when available."""
    cache_root = Path(
        os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")
    ) / "hub"
    model_root = cache_root / f"models--{model_id.replace('/', '--')}"
    reference = model_root / "refs" / "main"
    if reference.exists():
        snapshot = model_root / "snapshots" / reference.read_text(
            encoding="utf-8"
        ).strip()
        if snapshot.exists():
            return str(snapshot)
    return model_id


def _positive_probability(resource, text: str, logits_fn: Callable) -> float:
    encoded = resource.tokenizer(
        text,
        max_length=resource.max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(resource.device)
    attention_mask = encoded["attention_mask"].to(resource.device)
    with torch.inference_mode():
        logits = logits_fn(resource.model, input_ids, attention_mask)
        return float(torch.softmax(logits, dim=1)[0, 1].item())


def _metrics(expected: list[int], predicted: list[int]) -> dict[str, float]:
    tp = sum(actual == 1 and guess == 1 for actual, guess in zip(expected, predicted))
    tn = sum(actual == 0 and guess == 0 for actual, guess in zip(expected, predicted))
    fp = sum(actual == 0 and guess == 1 for actual, guess in zip(expected, predicted))
    fn = sum(actual == 1 and guess == 0 for actual, guess in zip(expected, predicted))
    accuracy = (tp + tn) / len(expected) if expected else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


def _evaluate(
    predictor,
    cases: tuple,
    member_names: tuple[str, ...],
    member_thresholds: dict[str, float],
    ensemble_threshold: float,
    logits_fn: Callable,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    resources = {name: predictor.xai_resources(name) for name in member_names}
    weights = {name: float(resources[name].weight) for name in member_names}
    expected = [int(case[2]) for case in cases]
    probabilities = {name: [] for name in member_names}
    probabilities["Ensemble"] = []

    for _case_id, text, _expected in cases:
        member_probabilities = {
            name: _positive_probability(resources[name], text, logits_fn)
            for name in member_names
        }
        for name, probability in member_probabilities.items():
            probabilities[name].append(probability)
        probabilities["Ensemble"].append(
            sum(weights[name] * member_probabilities[name] for name in member_names)
        )

    thresholds = {**member_thresholds, "Ensemble": ensemble_threshold}
    results = {
        name: _metrics(
            expected,
            [int(value >= thresholds[name]) for value in probabilities[name]],
        )
        for name in (*member_names, "Ensemble")
    }
    return results, weights


def _save_chart(
    header: str,
    results: dict[str, dict[str, float]],
    weights: dict[str, float],
    ensemble_threshold: float,
) -> Path:
    names = tuple(results)
    x = np.arange(len(METRIC_NAMES))
    width = 0.8 / len(names)
    figure, (metric_axis, weight_axis) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": (2.3, 1)}
    )

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(names)))
    for index, (name, color) in enumerate(zip(names, colors)):
        offset = (index - (len(names) - 1) / 2) * width
        bars = metric_axis.bar(
            x + offset,
            [results[name][metric] for metric in METRIC_NAMES],
            width,
            label=name,
            color=color,
        )
        metric_axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)

    metric_axis.set_xticks(x, METRIC_NAMES)
    metric_axis.set_ylim(0, 1.12)
    metric_axis.set_ylabel("Score")
    metric_axis.set_title("Member vs weighted-ensemble performance")
    metric_axis.grid(axis="y", alpha=0.25)
    metric_axis.legend(loc="lower left")

    member_names = tuple(weights)
    weight_bars = weight_axis.bar(
        member_names,
        [weights[name] for name in member_names],
        color=plt.cm.Purples(np.linspace(0.45, 0.85, len(member_names))),
    )
    weight_axis.bar_label(weight_bars, fmt="%.2f", padding=3)
    weight_axis.set_ylim(0, 1.0)
    weight_axis.set_ylabel("Weight")
    weight_axis.set_title(f"Deployed weights\nthreshold = {ensemble_threshold:.2f}")
    weight_axis.tick_params(axis="x", rotation=18)
    weight_axis.grid(axis="y", alpha=0.25)

    figure.suptitle(f"C3 {header} Header Ensemble — 10 Sample Tests", fontsize=16)
    figure.tight_layout()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / f"{header.lower()}_ensemble_chart.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _print_results(header: str, results: dict[str, dict[str, float]]) -> None:
    print(f"\n{header} header")
    print(f"{'Model':<14} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 58)
    for name, metrics in results.items():
        print(
            f"{name:<14} "
            f"{metrics['Accuracy']:>10.3f} "
            f"{metrics['Precision']:>10.3f} "
            f"{metrics['Recall']:>10.3f} "
            f"{metrics['F1']:>10.3f}"
        )


def _run_stress() -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    original_ids = {
        name: spec["runtime_hf_id"] for name, spec in STRESS_ENSEMBLE_MEMBERS.items()
    }
    predictor = None
    try:
        for spec in STRESS_ENSEMBLE_MEMBERS.values():
            spec["runtime_hf_id"] = _cached_model_path(spec["runtime_hf_id"])
        predictor = stress_model.StressPredictor()
        member_names = tuple(STRESS_ENSEMBLE_MEMBERS)
        return _evaluate(
            predictor,
            STRESS_CASES,
            member_names,
            {name: STRESS_DECISION_THRESHOLD for name in member_names},
            STRESS_DECISION_THRESHOLD,
            lambda model, ids, mask: model(
                input_ids=ids, attention_mask=mask
            )[0],
        )
    finally:
        del predictor
        for name, runtime_id in original_ids.items():
            STRESS_ENSEMBLE_MEMBERS[name]["runtime_hf_id"] = runtime_id
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_cbt() -> tuple[
    dict[str, dict[str, float]], dict[str, float], float
]:
    config = json.loads(CBT_BINARY_CONFIG_PATH.read_text(encoding="utf-8"))
    original_overrides = dict(CBT_RUNTIME_HF_ID_OVERRIDES)
    predictor = None
    try:
        CBT_RUNTIME_HF_ID_OVERRIDES.update(
            {
                name: _cached_model_path(
                    CBT_RUNTIME_HF_ID_OVERRIDES.get(
                        name, spec.get("runtime_hf_id", spec["hf_id"])
                    )
                )
                for name, spec in config["models"].items()
            }
        )
        predictor = cbt_model.CBTPredictor()
        member_names = tuple(config["deployment"]["members"])
        ensemble_threshold = float(config["deployment"]["threshold"])
        results, weights = _evaluate(
            predictor,
            CBT_CASES,
            member_names,
            {
                name: float(config["models"][name]["threshold"])
                for name in member_names
            },
            ensemble_threshold,
            lambda model, ids, mask: model(input_ids=ids, attention_mask=mask),
        )
        return results, weights, ensemble_threshold
    finally:
        del predictor
        CBT_RUNTIME_HF_ID_OVERRIDES.clear()
        CBT_RUNTIME_HF_ID_OVERRIDES.update(original_overrides)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    stress_results, stress_weights = _run_stress()
    _print_results("Stress", stress_results)
    stress_chart = _save_chart(
        "Stress", stress_results, stress_weights, STRESS_DECISION_THRESHOLD
    )
    print(f"Chart: {stress_chart}")

    cbt_results, cbt_weights, cbt_threshold = _run_cbt()
    _print_results("CBT", cbt_results)
    cbt_chart = _save_chart("CBT", cbt_results, cbt_weights, cbt_threshold)
    print(f"Chart: {cbt_chart}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
