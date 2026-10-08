"""Feature-MLP inference and signal processing for the C1 stress component.

The implementation mirrors the exported training/live-test pipeline:

raw BVP + EDA -> 60-second feature window -> personal z-score ->
SimpleImputer -> StandardScaler -> Feature-MLP -> isotonic calibration.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from scipy.signal import butter, filtfilt, find_peaks
from torch import nn

from settings import C1_ARTIFACT_DIR as C1_DIR
from xai import FeatureAttribution, explain_features

FEATURE_MLP_NAME = "Feature-MLP"

ALL_FEATURE_COLS = (
    "eda_mean",
    "eda_std",
    "eda_min",
    "eda_max",
    "eda_range",
    "eda_slope",
    "eda_num_peaks",
    "eda_peak_amplitude",
    "eda_phasic_mean",
    "eda_phasic_std",
    "eda_phasic_max",
    "eda_tonic_mean",
    "bvp_mean_ibi",
    "bvp_std_ibi",
    "bvp_min_ibi",
    "bvp_max_ibi",
    "bvp_heart_rate",
    "bvp_prv_sdnn",
    "bvp_prv_rmssd",
    "bvp_prv_pnn50",
)


class SignalQualityError(ValueError):
    """Raised when sensor data cannot produce a trustworthy feature window."""


# Which hardware produced the BVP channel. This matters for exactly one check:
# the MAX30102 reports raw unsigned IR counts, so its DC level tells you whether
# a finger is still on the sensor. The Empatica E4 exports an already band-passed
# BVP centred on zero, where the same test would condemn every window -- see
# ``e4.py``. Everything downstream of that check is scale-free and identical.
BVP_SOURCE_MAX30102 = "max30102"
BVP_SOURCE_E4 = "e4"
BVP_SOURCES = frozenset({BVP_SOURCE_MAX30102, BVP_SOURCE_E4})
MAX30102_MIN_MEDIAN_IR = 20000.0


def validate_bvp_source(bvp_source: str) -> str:
    normalized = str(bvp_source).strip().lower()
    if normalized not in BVP_SOURCES:
        choices = ", ".join(sorted(BVP_SOURCES))
        raise ValueError(f"Unknown BVP source {bvp_source!r}; expected one of: {choices}")
    return normalized


@dataclass(frozen=True)
class SensorSample:
    t_sec: float
    bvp: float
    eda: float


@dataclass(frozen=True)
class WindowDiagnostics:
    pulse_peaks: int
    valid_ibi: int
    eda_samples: int
    bvp_samples: int
    median_ir: float
    quality_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class CalibrationStats:
    feature_mu: np.ndarray
    feature_sd: np.ndarray
    feature_cols: tuple[str, ...]
    start_t: float
    end_t: float
    n_windows: int

    def __post_init__(self) -> None:
        count = len(self.feature_cols)
        if self.feature_mu.shape != (count,) or self.feature_sd.shape != (count,):
            raise ValueError("Calibration feature statistics have an invalid shape")
        if not np.all(np.isfinite(self.feature_mu)):
            raise ValueError("Calibration feature means must be finite")
        if not np.all(np.isfinite(self.feature_sd)) or np.any(self.feature_sd <= 0):
            raise ValueError("Calibration feature standard deviations must be positive")


@dataclass(frozen=True)
class C1Prediction:
    label: str
    is_stressed: bool
    raw_probability: float
    calibrated_probability: float
    threshold: float
    window_start_t: float
    window_end_t: float
    features: dict[str, float]
    diagnostics: WindowDiagnostics


class FeatureMLP(nn.Module):
    """Architecture reconstructed from ``feature_mlp.pt`` and ``dl_params``."""

    def __init__(
        self,
        n_features: int,
        hidden: Sequence[int] = (64, 32),
        p_drop: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = n_features
        for size in hidden:
            layers.extend(
                (
                    nn.Linear(previous, int(size)),
                    nn.BatchNorm1d(int(size)),
                    nn.ReLU(),
                    nn.Dropout(float(p_drop)),
                )
            )
            previous = int(size)
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values).squeeze(1)


def _torch_load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions before ``weights_only``.
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported Feature-MLP checkpoint: {type(state).__name__}")
    return {key.removeprefix("module."): value for key, value in state.items()}


def _as_arrays(
    samples: Iterable[SensorSample] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(samples, np.ndarray):
        values = np.asarray(samples, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("Sample array must have columns: t_sec, bvp, eda")
    else:
        values = np.asarray(
            [(item.t_sec, item.bvp, item.eda) for item in samples],
            dtype=np.float64,
        )
    if values.size == 0:
        raise ValueError("No sensor samples were provided")
    finite = np.isfinite(values).all(axis=1)
    values = values[finite]
    if len(values) < 5:
        raise SignalQualityError("At least five finite sensor samples are required")
    values = values[np.argsort(values[:, 0], kind="stable")]
    keep = np.r_[True, np.diff(values[:, 0]) > 1e-9]
    values = values[keep]
    return values[:, 0], values[:, 1], values[:, 2]


def _butter_filter(
    values: np.ndarray,
    fs: float,
    cutoff: float | Sequence[float],
    btype: str = "low",
    order: int = 4,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    if len(values) < 20:
        raise SignalQualityError("Not enough samples for signal filtering")
    nyquist = 0.5 * fs
    if btype == "band":
        normalized = [float(cutoff[0]) / nyquist, float(cutoff[1]) / nyquist]
    else:
        normalized = float(cutoff) / nyquist
    normalized = np.clip(normalized, 1e-6, 0.999)
    b, a = butter(order, normalized, btype=btype)
    return filtfilt(b, a, values)


def _resample_by_time(
    timestamps: np.ndarray,
    values: np.ndarray,
    fs: float,
    start_t: float,
    end_t: float,
) -> np.ndarray:
    # Keep one sample on either side of the requested interval.  A live
    # stream rarely lands exactly on every five-second boundary; discarding
    # the sample just before the boundary could make a valid, continuous
    # stream look incomplete.  The bracketing samples also let np.interp do
    # proper interpolation at both edges instead of holding an edge value.
    left = int(np.searchsorted(timestamps, start_t, side="left"))
    right = int(np.searchsorted(timestamps, end_t, side="right"))
    nearest_start = min(
        (
            abs(float(timestamps[index]) - start_t)
            for index in (left - 1, left)
            if 0 <= index < len(timestamps)
        ),
        default=float("inf"),
    )
    nearest_end = min(
        (
            abs(float(timestamps[index]) - end_t)
            for index in (right - 1, right)
            if 0 <= index < len(timestamps)
        ),
        default=float("inf"),
    )
    selected_t = timestamps[max(0, left - 1) : min(len(timestamps), right + 1)]
    selected_values = values[max(0, left - 1) : min(len(values), right + 1)]
    if len(selected_t) < 5:
        raise SignalQualityError("Not enough samples in the requested time window")
    if nearest_start > 0.25 or nearest_end > 0.25:
        raise SignalQualityError(
            "Raw stream does not fully cover the requested window "
            f"(nearest boundary samples: {nearest_start:.3f}s, {nearest_end:.3f}s)"
        )
    grid = np.arange(start_t, end_t, 1.0 / fs, dtype=np.float64)
    return np.interp(grid, selected_t, selected_values)


def _detect_bvp_peaks(bvp: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    filtered = _butter_filter(bvp, fs, (0.5, 4.0), btype="band", order=3)
    median = np.median(filtered)
    mad = np.median(np.abs(filtered - median))
    robust_sd = 1.4826 * mad
    if not np.isfinite(robust_sd) or robust_sd < 1e-8:
        robust_sd = np.std(filtered) + 1e-8
    normalized = (filtered - median) / robust_sd
    peaks, _ = find_peaks(
        normalized,
        distance=max(1, int(0.30 * fs)),
        prominence=0.5,
    )
    return peaks, filtered


def _decompose_eda(eda: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cleaned = _butter_filter(eda, fs, 1.0, btype="low", order=4)
    tonic = _butter_filter(cleaned, fs, 0.05, btype="low", order=2)
    return cleaned, tonic, cleaned - tonic


def _eda_features(
    cleaned: np.ndarray, tonic: np.ndarray, phasic: np.ndarray, fs: float
) -> list[float]:
    if len(cleaned) < 3:
        return [np.nan] * 12
    seconds = np.arange(len(cleaned)) / fs
    slope = np.polyfit(seconds, cleaned, 1)[0]
    median = np.median(phasic)
    mad = np.median(np.abs(phasic - median))
    robust_sd = 1.4826 * mad
    if not np.isfinite(robust_sd) or robust_sd < 1e-8:
        robust_sd = np.std(phasic) + 1e-8
    phasic_z = (phasic - median) / robust_sd
    peaks, properties = find_peaks(
        phasic_z,
        prominence=0.5,
        distance=max(1, int(1.0 * fs)),
    )
    peak_amplitude = (
        float(np.mean(properties["prominences"])) if len(peaks) else 0.0
    )
    return [
        float(np.mean(cleaned)),
        float(np.std(cleaned)),
        float(np.min(cleaned)),
        float(np.max(cleaned)),
        float(np.max(cleaned) - np.min(cleaned)),
        float(slope),
        float(len(peaks)),
        peak_amplitude,
        float(np.mean(phasic)),
        float(np.std(phasic)),
        float(np.max(phasic)),
        float(np.mean(tonic)),
    ]


def _pulse_interval_features(ibi_ms: np.ndarray) -> list[float]:
    ibi_ms = np.asarray(ibi_ms, dtype=np.float64)
    ibi_ms = ibi_ms[np.isfinite(ibi_ms)]
    if len(ibi_ms) < 2:
        return [np.nan] * 8
    mean_ibi = float(np.mean(ibi_ms))
    differences = np.diff(ibi_ms)
    return [
        mean_ibi,
        float(np.std(ibi_ms)),
        float(np.min(ibi_ms)),
        float(np.max(ibi_ms)),
        60000.0 / mean_ibi,
        float(np.std(ibi_ms, ddof=1)),
        float(np.sqrt(np.mean(differences**2))) if len(differences) else np.nan,
        float(np.mean(np.abs(differences) > 50) * 100.0)
        if len(differences)
        else np.nan,
    ]


def extract_feature_window(
    samples: Iterable[SensorSample] | np.ndarray,
    start_t: float,
    end_t: float,
    feature_cols: Sequence[str],
    *,
    bvp_fs: float = 64.0,
    eda_fs: float = 4.0,
    bvp_source: str = BVP_SOURCE_MAX30102,
) -> tuple[np.ndarray, WindowDiagnostics]:
    """Extract the ordered Feature-MLP vector from one raw 60-second window."""

    bvp_source = validate_bvp_source(bvp_source)
    if end_t <= start_t:
        raise ValueError("end_t must be greater than start_t")
    unknown = sorted(set(feature_cols) - set(ALL_FEATURE_COLS))
    if unknown:
        raise ValueError("Unsupported C1 features: " + ", ".join(unknown))

    timestamps, bvp, eda = _as_arrays(samples)
    bvp_resampled = _resample_by_time(timestamps, bvp, bvp_fs, start_t, end_t)
    eda_resampled = _resample_by_time(timestamps, eda, eda_fs, start_t, end_t)

    peaks, _ = _detect_bvp_peaks(bvp_resampled, bvp_fs)
    pulse_times = start_t + peaks / bvp_fs
    ibi = np.diff(pulse_times) * 1000.0
    ibi_midpoints = (
        (pulse_times[:-1] + pulse_times[1:]) / 2.0
        if len(pulse_times) >= 2
        else np.asarray([])
    )
    plausible = (ibi > 300.0) & (ibi < 2000.0)
    ibi = ibi[plausible]
    ibi_midpoints = ibi_midpoints[plausible]
    inside = (ibi_midpoints >= start_t) & (ibi_midpoints < end_t)
    window_ibi = ibi[inside]

    eda_cleaned, tonic, phasic = _decompose_eda(eda_resampled, eda_fs)
    feature_values = _eda_features(eda_cleaned, tonic, phasic, eda_fs)
    feature_values.extend(_pulse_interval_features(window_ibi))
    feature_map = dict(zip(ALL_FEATURE_COLS, feature_values, strict=True))
    ordered = np.asarray([feature_map[name] for name in feature_cols], dtype=np.float64)

    median_ir = float(np.median(bvp_resampled))
    flags: list[str] = []
    # Only meaningful for the MAX30102; the E4's zero-centred BVP has a median
    # near 1, not near 50,000, so applying this to it would flag every window.
    if bvp_source == BVP_SOURCE_MAX30102 and median_ir < MAX30102_MIN_MEDIAN_IR:
        flags.append("weak_bvp_or_finger_missing")
    if len(window_ibi) < 2:
        flags.append("insufficient_valid_ibi")
    if float(np.ptp(eda_resampled)) < 1e-8:
        flags.append("flat_eda_signal")

    diagnostics = WindowDiagnostics(
        pulse_peaks=int(len(peaks)),
        valid_ibi=int(len(window_ibi)),
        eda_samples=int(len(eda_resampled)),
        bvp_samples=int(len(bvp_resampled)),
        median_ir=median_ir,
        quality_flags=tuple(flags),
    )
    return ordered, diagnostics


def compute_calibration(
    samples: Iterable[SensorSample] | np.ndarray,
    manifest: dict,
    *,
    start_t: float | None = None,
    bvp_source: str = BVP_SOURCE_MAX30102,
) -> CalibrationStats:
    """Compute personal feature mean/std from the first 300 seconds (49 windows)."""

    bvp_source = validate_bvp_source(bvp_source)
    timestamps, bvp, eda = _as_arrays(samples)
    values = np.column_stack((timestamps, bvp, eda))
    calibration_start = float(timestamps[0] if start_t is None else start_t)
    calibration_sec = float(manifest.get("calibration_sec", 300.0))
    window_sec = float(manifest.get("window_sec", 60.0))
    step_sec = float(manifest.get("step_sec", 5.0))
    calibration_end = calibration_start + calibration_sec
    # A channel sampled for exactly N seconds ends one sample interval before
    # N (e.g. 359.984375 for 64 Hz). The window extractor already permits a
    # quarter-second edge tolerance, so apply the same rule here.
    if timestamps[-1] < calibration_end - 0.25:
        remaining = calibration_end - float(timestamps[-1])
        raise ValueError(f"Calibration needs {remaining:.1f} more seconds")

    windows: list[np.ndarray] = []
    diagnostics: list[WindowDiagnostics] = []
    window_start = calibration_start
    while window_start + window_sec <= calibration_end + 1e-9:
        features, window_diagnostics = extract_feature_window(
            values,
            window_start,
            window_start + window_sec,
            manifest["feature_cols"],
            bvp_fs=float(manifest.get("bvp_fs", 64.0)),
            eda_fs=float(manifest.get("eda_fs", 4.0)),
            bvp_source=bvp_source,
        )
        windows.append(features)
        diagnostics.append(window_diagnostics)
        window_start += step_sec

    expected_windows = int(manifest.get("calibration_windows", len(windows)))
    if len(windows) != expected_windows:
        raise ValueError(
            f"Expected {expected_windows} calibration windows, extracted {len(windows)}"
        )
    weak_window_count = sum(
        "weak_bvp_or_finger_missing" in item.quality_flags for item in diagnostics
    )
    if weak_window_count > len(diagnostics) // 2:
        raise SignalQualityError(
            "BVP signal was weak for most calibration windows; check finger placement"
        )

    matrix = np.asarray(windows, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        feature_mu = np.nanmean(matrix, axis=0)
        feature_sd = np.nanstd(matrix, axis=0)
    feature_mu[~np.isfinite(feature_mu)] = 0.0
    feature_sd[~np.isfinite(feature_sd) | (feature_sd < 1e-8)] = 1.0
    return CalibrationStats(
        feature_mu=feature_mu,
        feature_sd=feature_sd,
        feature_cols=tuple(manifest["feature_cols"]),
        start_t=calibration_start,
        end_t=calibration_end,
        n_windows=len(windows),
    )


class C1FeatureMLPPredictor:
    """Loads all local C1 artifacts and produces calibrated stress predictions."""

    def __init__(self, artifact_dir: str | Path = C1_DIR) -> None:
        self.artifact_dir = Path(artifact_dir).expanduser().resolve()
        self.manifest = self._read_json("run_manifest.json")
        self.dl_params = self._read_json("dl_params.json")
        self._validate_artifacts()
        self.feature_cols = tuple(self.manifest["feature_cols"])
        self.threshold = float(self.manifest["thresholds"][FEATURE_MLP_NAME])

        mlp_config = self.dl_params["mlp"]
        self.model = FeatureMLP(
            len(self.feature_cols),
            hidden=tuple(mlp_config["hidden"]),
            p_drop=float(mlp_config["p_drop"]),
        )
        self.model.load_state_dict(
            _torch_load_state(self.artifact_dir / "feature_mlp.pt"), strict=True
        )
        self.model.eval()

        with (self.artifact_dir / "feature_mlp_prep.pkl").open("rb") as file:
            preprocessing = pickle.load(file)
        if not isinstance(preprocessing, tuple) or len(preprocessing) != 2:
            raise TypeError("feature_mlp_prep.pkl must contain (imputer, scaler)")
        self.imputer, self.scaler = preprocessing

        with (self.artifact_dir / "calibrators.pkl").open("rb") as file:
            calibrators = pickle.load(file)
        try:
            self.calibrator = calibrators[FEATURE_MLP_NAME]
        except (KeyError, TypeError) as exc:
            raise KeyError("calibrators.pkl is missing the Feature-MLP calibrator") from exc

    def _read_json(self, filename: str) -> dict:
        path = self.artifact_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing C1 artifact: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _validate_artifacts(self) -> None:
        required = (
            "feature_mlp.pt",
            "feature_mlp_prep.pkl",
            "calibrators.pkl",
        )
        missing = [str(self.artifact_dir / name) for name in required if not (self.artifact_dir / name).exists()]
        if missing:
            raise FileNotFoundError("Missing C1 artifacts: " + ", ".join(missing))
        if self.manifest.get("calib_mode") != "first_k":
            raise ValueError("C1 live inference requires calib_mode='first_k'")
        if not self.manifest.get("per_subject_norm", False):
            raise ValueError("C1 Feature-MLP requires personal feature normalization")
        if len(self.manifest.get("feature_cols", ())) != 18:
            raise ValueError("C1 Feature-MLP must receive exactly 18 features")

    def calibrate(
        self,
        samples: Iterable[SensorSample] | np.ndarray,
        *,
        start_t: float | None = None,
        bvp_source: str = BVP_SOURCE_MAX30102,
    ) -> CalibrationStats:
        return compute_calibration(
            samples, self.manifest, start_t=start_t, bvp_source=bvp_source
        )

    def _prepare_features(
        self,
        raw_features: Sequence[float] | np.ndarray,
        calibration: CalibrationStats,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(raw, model-ready) feature arrays: personal z-score -> imputer -> scaler."""

        if tuple(calibration.feature_cols) != self.feature_cols:
            raise ValueError("Calibration feature order does not match the model manifest")
        raw = np.asarray(raw_features, dtype=np.float64)
        if raw.shape != (len(self.feature_cols),):
            raise ValueError(f"Expected {len(self.feature_cols)} features, got {raw.shape}")
        personal = ((raw - calibration.feature_mu) / calibration.feature_sd).reshape(1, -1)
        prepared = self.imputer.transform(personal)
        prepared = self.scaler.transform(prepared).astype(np.float32)
        return raw, prepared

    def predict_features(
        self,
        raw_features: Sequence[float] | np.ndarray,
        calibration: CalibrationStats,
        *,
        window_start_t: float,
        window_end_t: float,
        diagnostics: WindowDiagnostics,
    ) -> C1Prediction:
        raw, prepared = self._prepare_features(raw_features, calibration)
        with torch.inference_mode():
            logit = self.model(torch.from_numpy(prepared))
            raw_probability = float(torch.sigmoid(logit)[0].item())
        calibrated_probability = float(
            np.asarray(self.calibrator.predict([raw_probability])).reshape(-1)[0]
        )
        calibrated_probability = float(np.clip(calibrated_probability, 0.0, 1.0))
        stressed = calibrated_probability >= self.threshold
        return C1Prediction(
            label="STRESS" if stressed else "BASELINE",
            is_stressed=stressed,
            raw_probability=raw_probability,
            calibrated_probability=calibrated_probability,
            threshold=self.threshold,
            window_start_t=float(window_start_t),
            window_end_t=float(window_end_t),
            features={
                name: float(value) for name, value in zip(self.feature_cols, raw, strict=True)
            },
            diagnostics=diagnostics,
        )

    def predict(
        self,
        samples: Iterable[SensorSample] | np.ndarray,
        calibration: CalibrationStats,
        *,
        start_t: float,
        end_t: float,
        bvp_source: str = BVP_SOURCE_MAX30102,
    ) -> C1Prediction:
        features, diagnostics = extract_feature_window(
            samples,
            start_t,
            end_t,
            self.feature_cols,
            bvp_fs=float(self.manifest.get("bvp_fs", 64.0)),
            eda_fs=float(self.manifest.get("eda_fs", 4.0)),
            bvp_source=bvp_source,
        )
        return self.predict_features(
            features,
            calibration,
            window_start_t=start_t,
            window_end_t=end_t,
            diagnostics=diagnostics,
        )

    def explain_features(
        self,
        raw_features: Sequence[float] | np.ndarray,
        calibration: CalibrationStats,
        *,
        n_steps: int = 64,
    ) -> list[FeatureAttribution]:
        """Integrated Gradients attribution over the 18 model-ready features
        (personal z-score -> imputer -> scaler, the same space the model
        actually sees). Baseline is all-zeros, i.e. each subject's own
        calibrated center, since StandardScaler leaves that near zero."""

        _raw, prepared = self._prepare_features(raw_features, calibration)
        inputs = torch.from_numpy(prepared).clone().requires_grad_(True)
        baseline = torch.zeros_like(inputs)

        def forward_func(x: torch.Tensor) -> torch.Tensor:
            # self.model(x) already returns shape [batch] (FeatureMLP.forward
            # squeezes its own [batch, 1] head output) -- no further squeeze.
            return torch.sigmoid(self.model(x))

        return explain_features(
            forward_func=forward_func,
            inputs=inputs,
            baseline=baseline,
            feature_names=self.feature_cols,
            n_steps=n_steps,
        )

    def explain(
        self,
        prediction: C1Prediction,
        calibration: CalibrationStats,
        *,
        n_steps: int = 64,
    ) -> list[FeatureAttribution]:
        """Explain an already-made prediction using its own stored raw features."""

        raw_features = [prediction.features[name] for name in self.feature_cols]
        return self.explain_features(raw_features, calibration, n_steps=n_steps)
