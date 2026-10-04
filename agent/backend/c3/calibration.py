"""Runtime affine-temperature calibration for C3 binary ensembles."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def apply_temperature(probabilities, temperature: float, bias: float) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    if not math.isfinite(bias):
        raise ValueError("Calibration bias must be finite")
    values = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    logits = np.log(values) - np.log1p(-values)
    scaled = logits / temperature + bias
    return 1.0 / (1.0 + np.exp(-scaled))


@dataclass(frozen=True)
class TemperatureCalibrator:
    temperature: float
    bias: float
    input_threshold: float
    calibrated_threshold: float
    artifact_path: Path

    @classmethod
    def from_json(cls, path: str | Path) -> "TemperatureCalibrator":
        artifact_path = Path(path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Missing C3 calibration artifact: {artifact_path}")
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        if payload.get("method") != "affine_temperature_scaling":
            raise ValueError(f"Unsupported calibration method in {artifact_path}")
        instance = cls(
            temperature=float(payload["temperature"]),
            bias=float(payload["bias"]),
            input_threshold=float(payload["input_threshold"]),
            calibrated_threshold=float(payload["calibrated_threshold"]),
            artifact_path=artifact_path,
        )
        expected = float(apply_temperature(
            [instance.input_threshold],
            instance.temperature,
            instance.bias,
        )[0])
        if abs(expected - instance.calibrated_threshold) > 1e-8:
            raise ValueError(f"Inconsistent threshold in {artifact_path}")
        return instance

    def calibrate_distribution(self, probabilities) -> np.ndarray:
        array = np.asarray(probabilities, dtype=float)
        positive = apply_temperature(
            array[..., 1],
            self.temperature,
            self.bias,
        )
        return np.stack([1.0 - positive, positive], axis=-1)

    def summary(self) -> dict:
        return {
            "method": "affine_temperature_scaling",
            "temperature": self.temperature,
            "bias": self.bias,
            "input_threshold": self.input_threshold,
            "calibrated_threshold": self.calibrated_threshold,
            "artifact_path": str(self.artifact_path),
        }
