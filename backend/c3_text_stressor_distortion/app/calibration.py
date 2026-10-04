"""Probability calibration and calibration-quality metrics for C3.

Temperature scaling is applied to the binary ensemble log-odds.  The scalar
temperature is fitted on a calibration partition that is disjoint from the
reported evaluation partition.  Transforming the original decision threshold
through the same monotonic function preserves the classifier's locked binary
decisions while making the returned probability better calibrated.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


EPSILON = 1e-7


def _clip_probabilities(probabilities: Iterable[float]) -> np.ndarray:
    return np.clip(
        np.asarray(list(probabilities), dtype=float),
        EPSILON,
        1.0 - EPSILON,
    )


def apply_temperature(
    probabilities: Iterable[float],
    temperature: float,
    bias: float = 0.0,
) -> np.ndarray:
    """Apply affine binary temperature scaling to probability log-odds."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be a finite positive number")
    values = _clip_probabilities(probabilities)
    logits = np.log(values) - np.log1p(-values)
    if not math.isfinite(bias):
        raise ValueError("Calibration bias must be finite")
    scaled = logits / float(temperature) + float(bias)
    return 1.0 / (1.0 + np.exp(-scaled))


def reliability_bins(
    labels: Iterable[int],
    probabilities: Iterable[float],
    *,
    n_bins: int = 15,
) -> list[dict[str, float | int]]:
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    y_true = np.asarray(list(labels), dtype=int)
    y_prob = _clip_probabilities(probabilities)
    if y_true.shape != y_prob.shape:
        raise ValueError("labels and probabilities must have equal length")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    assignments = np.minimum(np.digitize(y_prob, edges[1:-1]), n_bins - 1)
    rows = []
    for index in range(n_bins):
        mask = assignments == index
        count = int(mask.sum())
        rows.append({
            "bin": index,
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "count": count,
            "mean_confidence": float(y_prob[mask].mean()) if count else 0.0,
            "fraction_positive": float(y_true[mask].mean()) if count else 0.0,
        })
    return rows


def calibration_metrics(
    labels: Iterable[int],
    probabilities: Iterable[float],
    *,
    n_bins: int = 15,
) -> dict[str, float | int | list[dict[str, float | int]]]:
    y_true = np.asarray(list(labels), dtype=int)
    y_prob = _clip_probabilities(probabilities)
    if y_true.shape != y_prob.shape or not y_true.size:
        raise ValueError("Non-empty labels and probabilities must have equal length")
    bins = reliability_bins(y_true, y_prob, n_bins=n_bins)
    ece = sum(
        (row["count"] / len(y_true))
        * abs(row["fraction_positive"] - row["mean_confidence"])
        for row in bins
    )
    brier = float(np.mean((y_prob - y_true) ** 2))
    nll = float(-np.mean(
        y_true * np.log(y_prob) + (1 - y_true) * np.log1p(-y_prob)
    ))
    return {
        "samples": int(len(y_true)),
        "n_bins": int(n_bins),
        "ece": float(ece),
        "brier_score": brier,
        "negative_log_likelihood": nll,
        "reliability_bins": bins,
    }


def fit_temperature(
    labels: Iterable[int],
    probabilities: Iterable[float],
) -> float:
    """Fit one positive temperature by minimizing binary NLL."""
    from scipy.optimize import minimize_scalar

    y_true = np.asarray(list(labels), dtype=int)
    y_prob = _clip_probabilities(probabilities)
    if y_true.shape != y_prob.shape or not y_true.size:
        raise ValueError("Non-empty labels and probabilities must have equal length")

    def objective(log_temperature: float) -> float:
        calibrated = apply_temperature(y_prob, math.exp(log_temperature))
        return float(-np.mean(
            y_true * np.log(calibrated)
            + (1 - y_true) * np.log1p(-calibrated)
        ))

    result = minimize_scalar(
        objective,
        bounds=(-4.0, 4.0),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"Temperature fitting failed: {result.message}")
    return float(math.exp(float(result.x)))


def fit_affine_temperature(
    labels: Iterable[int],
    probabilities: Iterable[float],
) -> tuple[float, float]:
    """Fit temperature plus bias (binary Platt/logistic calibration).

    A learned intercept is important when class prevalence and the ensemble's
    implicit prior differ.  The positive slope keeps the mapping monotonic.
    """
    from sklearn.linear_model import LogisticRegression

    y_true = np.asarray(list(labels), dtype=int)
    y_prob = _clip_probabilities(probabilities)
    if y_true.shape != y_prob.shape or not y_true.size:
        raise ValueError("Non-empty labels and probabilities must have equal length")
    logits = (np.log(y_prob) - np.log1p(-y_prob)).reshape(-1, 1)
    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
    )
    model.fit(logits, y_true)
    slope = float(model.coef_[0, 0])
    bias = float(model.intercept_[0])
    if not math.isfinite(slope) or slope <= 0:
        raise RuntimeError("Calibration fit did not produce a positive slope")
    return 1.0 / slope, bias


@dataclass(frozen=True)
class TemperatureCalibrator:
    temperature: float
    bias: float
    input_threshold: float
    calibrated_threshold: float
    artifact_path: Path | None = None
    metadata: dict | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "TemperatureCalibrator":
        artifact_path = Path(path)
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Missing probability-calibration artifact: {artifact_path}"
            )
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        if payload.get("method") not in {
            "temperature_scaling",
            "affine_temperature_scaling",
        }:
            raise ValueError(
                f"Unsupported calibration method in {artifact_path}: "
                f"{payload.get('method')!r}"
            )
        temperature = float(payload["temperature"])
        bias = float(payload.get("bias", 0.0))
        input_threshold = float(payload["input_threshold"])
        calibrated_threshold = float(payload["calibrated_threshold"])
        expected_threshold = float(
            apply_temperature([input_threshold], temperature, bias)[0]
        )
        if abs(expected_threshold - calibrated_threshold) > 1e-8:
            raise ValueError(
                f"Calibration threshold in {artifact_path} is inconsistent "
                "with its temperature"
            )
        return cls(
            temperature=temperature,
            bias=bias,
            input_threshold=input_threshold,
            calibrated_threshold=calibrated_threshold,
            artifact_path=artifact_path,
            metadata=payload,
        )

    def calibrate_positive(self, probability: float) -> float:
        return float(apply_temperature(
            [probability],
            self.temperature,
            self.bias,
        )[0])

    def calibrate_distribution(self, probabilities) -> np.ndarray:
        array = np.asarray(probabilities, dtype=float)
        if array.shape[-1] != 2:
            raise ValueError("Binary probabilities must have a final dimension of 2")
        positive = apply_temperature(
            array[..., 1].reshape(-1),
            self.temperature,
            self.bias,
        )
        positive = positive.reshape(array.shape[:-1])
        return np.stack([1.0 - positive, positive], axis=-1)

    def summary(self) -> dict:
        return {
            "method": "affine_temperature_scaling",
            "temperature": self.temperature,
            "bias": self.bias,
            "input_threshold": self.input_threshold,
            "calibrated_threshold": self.calibrated_threshold,
            "artifact_path": str(self.artifact_path) if self.artifact_path else None,
        }
