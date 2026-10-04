import numpy as np


def sigmoid(x):
    x = np.clip(
        np.asarray(x, dtype=float),
        -40.0,
        40.0,
    )
    return 1.0 / (1.0 + np.exp(-x))


def increasing_membership(x, center, scale):
    return sigmoid(
        (np.asarray(x) - center) / scale
    )


def decreasing_membership(x, center, scale):
    return sigmoid(
        (center - np.asarray(x)) / scale
    )


def medium_membership(
    low_membership,
    high_membership,
):
    m = 1.0 - np.maximum(
        low_membership,
        high_membership,
    )
    return np.clip(m, 0.0, 1.0)


def safe_scale(q33, q67, floor=0.20):
    return max(
        abs(q67 - q33) / 2.0,
        floor,
    )
