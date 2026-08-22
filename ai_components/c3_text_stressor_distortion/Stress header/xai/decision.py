"""Shared binary-decision helpers for local XAI methods."""

from __future__ import annotations

from collections.abc import Sequence


def select_predicted_class(
    probabilities: Sequence[float],
    decision_threshold: float | None = None,
) -> int:
    """Select a class using argmax or an explicit positive-class threshold."""
    values = [float(value) for value in probabilities]
    if not values:
        raise ValueError("At least one class probability is required.")
    if decision_threshold is None:
        return max(range(len(values)), key=values.__getitem__)
    if len(values) != 2:
        raise ValueError(
            "A decision threshold can only be applied to binary probabilities."
        )
    threshold = float(decision_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("decision_threshold must be between 0 and 1.")
    return int(values[1] >= threshold)
