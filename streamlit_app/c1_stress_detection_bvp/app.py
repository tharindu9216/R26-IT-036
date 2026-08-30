import json
import pickle
import shlex
import sys
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt, find_peaks, resample_poly

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:
    from Xai.c1_stress_detection_bvp.BVP_EDA import shap_explain, gradcam
    XAI_AVAILABLE = True
except ImportError:
    XAI_AVAILABLE = False
MODELS_ROOT = PROJECT_ROOT / "models" / "c1_stress_detection_bvp"
LEGACY_MODEL_RUN = APP_DIR / "model_run"
SERIAL_BAUD = 230400
CALIBRATION_SEC = 90.0
WINDOW_SEC = 60.0
STEP_SEC = 5.0
BVP_FS = 64
EDA_FS = 4
TARGET_FS = 32

ALL_FEATURE_COLS = [
    "eda_mean", "eda_std", "eda_min", "eda_max", "eda_range", "eda_slope",
    "eda_num_peaks", "eda_peak_amplitude",
    "eda_phasic_mean", "eda_phasic_std", "eda_phasic_max", "eda_tonic_mean",
    "bvp_mean_ibi", "bvp_std_ibi", "bvp_min_ibi", "bvp_max_ibi",
    "bvp_heart_rate", "bvp_prv_sdnn", "bvp_prv_rmssd", "bvp_prv_pnn50",
]

CLASSICAL_MODELS = [
    "Logistic Regression",
    "Random Forest",
    "SVM (RBF)",
    "HistGradientBoosting",
]

MODEL_SLUG = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
    "SVM (RBF)": "svm_rbf",
    "HistGradientBoosting": "histgradientboosting",
    "Feature-MLP": "feature_mlp",
    "CNN-LSTM": "cnn_lstm",
}


def _model_count(run_path):
    """Return the number of complete, loadable model artifacts in a run."""
    model_dir = run_path / "models"
    count = sum((model_dir / f"{MODEL_SLUG[name]}.pkl").is_file() for name in CLASSICAL_MODELS)
    count += int(
        (model_dir / "feature_mlp.pt").is_file()
        and (model_dir / "feature_mlp_prep.pkl").is_file()
    )
    count += int(
        (model_dir / "cnn_lstm.pt").is_file()
        and (model_dir / "cnn_lstm_prep.pkl").is_file()
    )
    return count


def discover_model_runs(models_root=MODELS_ROOT):
    """Find exported notebook runs anywhere below the app's models folder."""
    if not models_root.is_dir():
        return []
    runs = {
        manifest.parent.resolve()
        for manifest in models_root.rglob("run_manifest.json")
        if (manifest.parent / "models").is_dir()
    }
    return sorted(runs, key=lambda path: (-_model_count(path), str(path)))


def default_model_run():
    discovered = discover_model_runs()
    if discovered:
        return discovered[0]
    if (LEGACY_MODEL_RUN / "run_manifest.json").is_file():
        return LEGACY_MODEL_RUN
    return MODELS_ROOT


DEFAULT_MODEL_RUN = default_model_run()


class FeatureMLP(nn.Module):
    def __init__(self, n_features=18, hidden=(128, 64), p_drop=0.3):
        super().__init__()
        layers, prev = [], n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p_drop)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


class CNNLSTM(nn.Module):
    def __init__(self, n_channels=2, conv_ch=(32, 64), lstm_hidden=64,
                 lstm_layers=1, p_drop=0.3, bidirectional=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, conv_ch[0], kernel_size=7, padding=3),
            nn.BatchNorm1d(conv_ch[0]), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(conv_ch[0], conv_ch[1], kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_ch[1]), nn.ReLU(), nn.MaxPool1d(2),
            nn.Dropout(p_drop),
        )
        self.lstm = nn.LSTM(
            conv_ch[1], lstm_hidden, num_layers=lstm_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=p_drop if lstm_layers > 1 else 0.0,
        )
        out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(p_drop), nn.Linear(out_dim, 1))

    def forward(self, x):
        z = self.conv(x.permute(0, 2, 1))
        z, _ = self.lstm(z.permute(0, 2, 1))
        # This must match the notebook architecture used to create cnn_lstm.pt.
        return self.head(z.mean(dim=1)).squeeze(1)


def butter_filter(x, fs, cutoff, btype="low", order=4):
    x = np.asarray(x, dtype=np.float64).ravel()
    if len(x) < 20:
        raise ValueError("Not enough samples for filtering.")
    nyq = 0.5 * fs
    wn = [cutoff[0] / nyq, cutoff[1] / nyq] if btype == "band" else cutoff / nyq
    wn = np.clip(wn, 1e-6, 0.999)
    b, a = butter(order, wn, btype=btype)
    return filtfilt(b, a, x)


def detect_bvp_peaks(bvp, fs):
    filt = butter_filter(bvp, fs, [0.5, 4.0], btype="band", order=3)
    med = np.median(filt)
    mad = np.median(np.abs(filt - med))
    robust_sd = 1.4826 * mad
    if not np.isfinite(robust_sd) or robust_sd < 1e-8:
        robust_sd = np.std(filt) + 1e-8
    z = (filt - med) / robust_sd
    peaks, _ = find_peaks(z, distance=max(1, int(0.30 * fs)), prominence=0.5)
    return peaks


def decompose_eda(eda, fs):
    eda = np.asarray(eda, dtype=np.float64).ravel()
    eda_clean = butter_filter(eda, fs, 1.0, btype="low", order=4)
    tonic = butter_filter(eda_clean, fs, 0.05, btype="low", order=2)
    phasic = eda_clean - tonic
    return eda_clean, tonic, phasic


def eda_window_features(eda_w, tonic_w, phasic_w, fs):
    eda_w = np.asarray(eda_w, dtype=np.float64)
    tonic_w = np.asarray(tonic_w, dtype=np.float64)
    phasic_w = np.asarray(phasic_w, dtype=np.float64)
    if len(eda_w) < 3:
        return [np.nan] * 12
    t = np.arange(len(eda_w)) / fs
    slope = np.polyfit(t, eda_w, 1)[0]

    # A fixed prominence of 0.01 assumes WESAD/Empatica microsiemens units.
    # Scale the phasic signal by its own robust spread first (same trick
    # detect_bvp_peaks() uses) so peak detection works on any device's raw
    # ADC scale, not just the one the model was trained on.
    med = np.median(phasic_w)
    mad = np.median(np.abs(phasic_w - med))
    robust_sd = 1.4826 * mad
    if not np.isfinite(robust_sd) or robust_sd < 1e-8:
        robust_sd = np.std(phasic_w) + 1e-8
    phasic_z = (phasic_w - med) / robust_sd

    pk, props = find_peaks(phasic_z, prominence=0.5, distance=max(1, int(1.0 * fs)))
    peak_amp = float(np.mean(props["prominences"])) if len(pk) else 0.0
    return [
        float(np.mean(eda_w)), float(np.std(eda_w)),
        float(np.min(eda_w)), float(np.max(eda_w)),
        float(np.max(eda_w) - np.min(eda_w)), float(slope),
        float(len(pk)), peak_amp,
        float(np.mean(phasic_w)), float(np.std(phasic_w)), float(np.max(phasic_w)),
        float(np.mean(tonic_w)),
    ]


def pulse_interval_features(ibi_ms):
    ibi_ms = np.asarray(ibi_ms, dtype=np.float64)
    ibi_ms = ibi_ms[np.isfinite(ibi_ms)]
    if len(ibi_ms) < 2:
        return [np.nan] * 8
    mean_ibi = float(np.mean(ibi_ms))
    diff_ibi = np.diff(ibi_ms)
    return [
        mean_ibi,
        float(np.std(ibi_ms)),
        float(np.min(ibi_ms)),
        float(np.max(ibi_ms)),
        60000.0 / mean_ibi,
        float(np.std(ibi_ms, ddof=1)),
        float(np.sqrt(np.mean(diff_ibi ** 2))) if len(diff_ibi) else np.nan,
        float(np.mean(np.abs(diff_ibi) > 50) * 100.0) if len(diff_ibi) else np.nan,
    ]


def resample_by_time(t, x, fs, start_t, end_t):
    # Tolerance for how close real samples must be to the window's edges.
    # streamlit_autorefresh polls once per second, so serial data naturally
    # arrives in ~1s bursts with some jitter between them. 0.25s was tighter
    # than that cadence and rejected perfectly good calibration windows on
    # ordinary jitter. 1.5s comfortably absorbs normal polling jitter while
    # still catching a genuine multi-second sensor/connection dropout.
    EDGE_TOLERANCE_SEC = 1.5
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    mask = (t >= start_t) & (t <= end_t)
    t, x = t[mask], x[mask]
    if len(t) < 5:
        raise ValueError("Not enough samples in requested interval.")
    keep = np.r_[True, np.diff(t) > 1e-9]
    t, x = t[keep], x[keep]
    grid = np.arange(start_t, end_t, 1.0 / fs, dtype=np.float64)
    if t[0] > start_t + EDGE_TOLERANCE_SEC:
        raise ValueError(
            f"Raw stream does not fully cover the requested window "
            f"[{start_t:.2f}, {end_t:.2f}]s: first in-range sample is at "
            f"t={t[0]:.2f}s ({t[0] - start_t:.2f}s gap at the window's START)."
        )
    if t[-1] < end_t - EDGE_TOLERANCE_SEC:
        raise ValueError(
            f"Raw stream does not fully cover the requested window "
            f"[{start_t:.2f}, {end_t:.2f}]s: last in-range sample is at "
            f"t={t[-1]:.2f}s ({end_t - t[-1]:.2f}s gap at the window's END)."
        )
    return np.interp(grid, t, x)


def extract_one_window(raw_df, start_t, end_t, feature_cols):
    t = raw_df["t_sec"].to_numpy(dtype=float)
    bvp = raw_df["bvp"].to_numpy(dtype=float)
    eda = raw_df["eda"].to_numpy(dtype=float)

    bvp64 = resample_by_time(t, bvp, BVP_FS, start_t, end_t)
    eda4 = resample_by_time(t, eda, EDA_FS, start_t, end_t)

    peaks = detect_bvp_peaks(bvp64, BVP_FS)
    pulse_t = start_t + peaks / BVP_FS
    ibi = np.diff(pulse_t) * 1000.0
    ibi_mid = (pulse_t[:-1] + pulse_t[1:]) / 2.0 if len(pulse_t) >= 2 else np.array([])
    ok = (ibi > 300.0) & (ibi < 2000.0)
    ibi, ibi_mid = ibi[ok], ibi_mid[ok]
    inside = (ibi_mid >= start_t) & (ibi_mid < end_t)
    ibi_w = ibi[inside]

    eda_clean, tonic, phasic = decompose_eda(eda4, EDA_FS)
    f_eda = eda_window_features(eda_clean, tonic, phasic, EDA_FS)
    f_bvp = pulse_interval_features(ibi_w)
    feature_dict = dict(zip(ALL_FEATURE_COLS, f_eda + f_bvp))
    ordered = np.asarray([feature_dict[c] for c in feature_cols], dtype=np.float64)

    bvp_clean = butter_filter(bvp64, BVP_FS, [0.5, 4.0], btype="band", order=3)
    bvp32 = resample_poly(bvp_clean, TARGET_FS, BVP_FS)
    eda32 = resample_poly(eda_clean, TARGET_FS, EDA_FS)
    expected = int(round((end_t - start_t) * TARGET_FS))
    n = min(len(bvp32), len(eda32), expected)
    raw32 = np.stack([eda32[:n], bvp32[:n]], axis=1)
    if n < expected:
        raw32 = np.vstack([raw32, np.repeat(raw32[-1:, :], expected - n, axis=0)])
    elif n > expected:
        raw32 = raw32[:expected]

    return ordered, raw32.astype(np.float32), {
        "pulse_peaks": int(len(peaks)),
        "valid_ibi": int(len(ibi_w)),
        "eda_samples": int(len(eda4)),
        "bvp_samples": int(len(bvp64)),
        "median_ir": float(np.median(bvp64)),
    }


def compute_calibration(raw_df, manifest):
    feature_cols = manifest["feature_cols"]
    window_sec = float(manifest.get("window_sec", WINDOW_SEC))
    step_sec = float(manifest.get("step_sec", STEP_SEC))
    calibration_sec = float(manifest.get("calibration_sec", CALIBRATION_SEC))
    t0 = float(raw_df["t_sec"].min())
    t_end = t0 + calibration_sec
    if raw_df["t_sec"].max() < t_end:
        remaining = t_end - float(raw_df["t_sec"].max())
        raise ValueError(f"Calibration needs {remaining:.1f} more seconds.")

    feature_windows, raw_windows = [], []
    s = t0
    while s + window_sec <= t_end + 1e-9:
        feat, raw32, _ = extract_one_window(raw_df, s, s + window_sec, feature_cols)
        feature_windows.append(feat)
        raw_windows.append(raw32)
        s += step_sec

    Xf = np.asarray(feature_windows, dtype=np.float64)
    Xr = np.asarray(raw_windows, dtype=np.float64)
    feature_mu = np.nanmean(Xf, axis=0)
    feature_sd = np.nanstd(Xf, axis=0)
    feature_mu[~np.isfinite(feature_mu)] = 0.0
    feature_sd[~np.isfinite(feature_sd) | (feature_sd < 1e-8)] = 1.0
    raw_mu = Xr.mean(axis=(0, 1))
    raw_sd = Xr.std(axis=(0, 1))
    raw_sd[raw_sd < 1e-8] = 1.0
    return {
        "feature_mu": feature_mu,
        "feature_sd": feature_sd,
        "raw_mu": raw_mu,
        "raw_sd": raw_sd,
        "n_windows": int(len(Xf)),
    }


def _torch_load_state(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _validate_manifest(manifest, manifest_path):
    feature_cols = manifest.get("feature_cols")
    if not isinstance(feature_cols, list) or not feature_cols:
        raise ValueError(f"Invalid or missing feature_cols in {manifest_path}")
    unknown = sorted(set(feature_cols) - set(ALL_FEATURE_COLS))
    if unknown:
        raise ValueError(f"Unsupported feature(s) in {manifest_path}: {', '.join(unknown)}")

    if manifest.get("per_subject_norm", False) and manifest.get("calib_mode") != "first_k":
        raise ValueError(
            "This live app supports per-subject normalization only for calib_mode='first_k'."
        )

    expected_rates = {"bvp_fs": BVP_FS, "eda_fs": EDA_FS, "target_fs": TARGET_FS}
    for key, app_value in expected_rates.items():
        model_value = manifest.get(key, app_value)
        if float(model_value) != float(app_value):
            raise ValueError(
                f"Incompatible {key}: model expects {model_value}, app uses {app_value}."
            )


def _positive_class_index(model, model_name):
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"{model_name} does not provide predict_proba().")
    classes = np.asarray(getattr(model, "classes_", [])).ravel()
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1:
        raise ValueError(f"{model_name} must have exactly one positive class labelled 1.")
    return int(matches[0])


@st.cache_resource(show_spinner=False)
def load_model_bundle(run_path_str):
    run_path = Path(run_path_str).expanduser().resolve()
    manifest_path = run_path / "run_manifest.json"
    model_dir = run_path / "models"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing: {manifest_path}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing model directory: {model_dir}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    _validate_manifest(manifest, manifest_path)

    dl_params_path = run_path / "dl_params.json"
    if dl_params_path.exists():
        with open(dl_params_path, "r") as f:
            dl_params = json.load(f)
    else:
        dl_params = {
            "mlp": {"hidden": [64, 32], "p_drop": 0.2, "lr": 0.003},
            "cnn": {"conv_ch": [64, 128], "lstm_hidden": 128, "p_drop": 0.4, "lr": 0.001},
        }

    models = {}
    for name in CLASSICAL_MODELS:
        p = model_dir / f"{MODEL_SLUG[name]}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                model = pickle.load(f)
            models[name] = {
                "kind": "classical",
                "model": model,
                "positive_class_index": _positive_class_index(model, name),
            }

    mlp_pt = model_dir / "feature_mlp.pt"
    mlp_prep = model_dir / "feature_mlp_prep.pkl"
    if mlp_pt.exists() and mlp_prep.exists():
        cfg = dl_params["mlp"]
        mlp = FeatureMLP(len(manifest["feature_cols"]), tuple(cfg["hidden"]), float(cfg["p_drop"]))
        mlp.load_state_dict(_torch_load_state(mlp_pt))
        mlp.eval()
        with open(mlp_prep, "rb") as f:
            imp, sc = pickle.load(f)
        models["Feature-MLP"] = {"kind": "mlp", "model": mlp, "imputer": imp, "scaler": sc}

    cnn_pt = model_dir / "cnn_lstm.pt"
    cnn_prep = model_dir / "cnn_lstm_prep.pkl"
    if cnn_pt.exists() and cnn_prep.exists():
        cfg = dl_params["cnn"]
        cnn = CNNLSTM(2, tuple(cfg["conv_ch"]), int(cfg["lstm_hidden"]), p_drop=float(cfg["p_drop"]))
        cnn.load_state_dict(_torch_load_state(cnn_pt))
        cnn.eval()
        with open(cnn_prep, "rb") as f:
            global_mu, global_sd = pickle.load(f)
        global_mu = np.asarray(global_mu, dtype=np.float64).reshape(-1)
        global_sd = np.asarray(global_sd, dtype=np.float64).reshape(-1)
        if global_mu.shape != (2,) or global_sd.shape != (2,):
            raise ValueError(
                f"CNN-LSTM preprocessing must contain two channels; got "
                f"mu={global_mu.shape}, sd={global_sd.shape}."
            )
        if not np.all(np.isfinite(global_mu)) or not np.all(np.isfinite(global_sd)):
            raise ValueError("CNN-LSTM preprocessing contains non-finite values.")
        if np.any(global_sd <= 0):
            raise ValueError("CNN-LSTM preprocessing standard deviations must be positive.")
        models["CNN-LSTM"] = {
            "kind": "cnn", "model": cnn,
            "global_mu": global_mu,
            "global_sd": global_sd,
        }

    if not models:
        raise FileNotFoundError(f"No complete model artifacts found in {model_dir}")

    return {"run_path": run_path, "manifest": manifest, "dl_params": dl_params, "models": models}


def predict_all(bundle, raw_feature_vector, raw32, calibration, ensemble_models=None):
    manifest = bundle["manifest"]
    x_feat = ((raw_feature_vector - calibration["feature_mu"]) / calibration["feature_sd"]).reshape(1, -1)
    x_raw = (raw32.astype(np.float64) - calibration["raw_mu"]) / calibration["raw_sd"]
    rows = []

    for name, item in bundle["models"].items():
        if item["kind"] == "classical":
            probabilities = item["model"].predict_proba(x_feat)
            p = float(probabilities[0, item["positive_class_index"]])
        elif item["kind"] == "mlp":
            X = item["imputer"].transform(x_feat)
            X = item["scaler"].transform(X).astype(np.float32)
            with torch.no_grad():
                p = float(torch.sigmoid(item["model"](torch.from_numpy(X)))[0].item())
        elif item["kind"] == "cnn":
            X = (x_raw - item["global_mu"]) / item["global_sd"]
            X = X.astype(np.float32)[None, :, :]
            with torch.no_grad():
                p = float(torch.sigmoid(item["model"](torch.from_numpy(X)))[0].item())
        else:
            continue

        threshold = float(manifest.get("thresholds", {}).get(name, 0.5))
        rows.append({
            "model": name,
            "stress_probability": p,
            "threshold": threshold,
            "prediction": "STRESS" if p >= threshold else "BASELINE",
        })

    if ensemble_models:
        chosen = [r for r in rows if r["model"] in ensemble_models]
        if len(chosen) >= 2:
            avg_p = float(np.mean([r["stress_probability"] for r in chosen]))
            avg_threshold = float(np.mean([r["threshold"] for r in chosen]))
            rows.append({
                "model": "Ensemble (" + " + ".join(ensemble_models) + ")",
                "stress_probability": avg_p,
                "threshold": avg_threshold,
                "prediction": "STRESS" if avg_p >= avg_threshold else "BASELINE",
            })

    return pd.DataFrame(rows)


def available_ports():
    if serial is None:
        return []
    return [p.device for p in serial.tools.list_ports.comports()]


def init_state():
    defaults = {
        "ser": None,
        "serial_text_buffer": "",
        "samples": [],
        "stream_start_ms": None,
        "calibration_started": False,
        "calibration": None,
        "last_prediction_at": None,
        "prediction_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def disconnect_serial():
    ser = st.session_state.get("ser")
    if ser is not None:
        try:
            ser.close()
        except Exception:
            pass
    st.session_state.ser = None


def clear_capture():
    st.session_state.samples = []
    st.session_state.serial_text_buffer = ""
    st.session_state.stream_start_ms = None
    st.session_state.calibration_started = False
    st.session_state.calibration = None
    st.session_state.last_prediction_at = None
    st.session_state.prediction_history = []


def start_calibration():
    """Start a fresh 5-minute calibration from the next serial sample."""
    st.session_state.samples = []
    st.session_state.serial_text_buffer = ""
    st.session_state.stream_start_ms = None
    st.session_state.calibration = None
    st.session_state.last_prediction_at = None
    st.session_state.prediction_history = []
    st.session_state.calibration_started = True

    # Discard any sensor samples that were queued before the user pressed Start.
    ser = st.session_state.get("ser")
    if ser is not None:
        try:
            ser.reset_input_buffer()
        except Exception:
            pass


def poll_serial(max_bytes=250000):
    ser = st.session_state.get("ser")
    if ser is None:
        return 0
    waiting = int(getattr(ser, "in_waiting", 0))
    if waiting <= 0:
        return 0

    data = ser.read(min(waiting, max_bytes)).decode("utf-8", errors="ignore")
    text = st.session_state.serial_text_buffer + data
    lines = text.split("\n")
    st.session_state.serial_text_buffer = lines[-1]
    count = 0

    for line in lines[:-1]:
        line = line.strip()
        if not line.startswith("DATA,"):
            continue
        parts = line.split(",")
        if len(parts) != 4:
            continue
        try:
            ms = int(parts[1]); bvp = float(parts[2]); eda = float(parts[3])
        except ValueError:
            continue
        if st.session_state.stream_start_ms is None:
            st.session_state.stream_start_ms = ms
        t_sec = (ms - st.session_state.stream_start_ms) / 1000.0
        if t_sec < 0:
            clear_capture(); st.session_state.stream_start_ms = ms; t_sec = 0.0
        st.session_state.samples.append((t_sec, bvp, eda))
        count += 1

    if st.session_state.samples and st.session_state.calibration is not None:
        latest = st.session_state.samples[-1][0]
        keep_after = latest - 420.0
        if keep_after > 0:
            st.session_state.samples = [r for r in st.session_state.samples if r[0] >= keep_after]
    return count


def samples_df():
    if not st.session_state.samples:
        return pd.DataFrame(columns=["t_sec", "bvp", "eda"])
    return pd.DataFrame(st.session_state.samples, columns=["t_sec", "bvp", "eda"])


def prepare_live_chart_data(df, max_rows=3000):
    """Return only finite, ordered sensor rows that Vega can safely plot."""
    columns = ["t_sec", "bvp", "eda"]
    if df.empty or not set(columns).issubset(df.columns):
        return pd.DataFrame(columns=columns)

    chart_df = df.loc[:, columns].tail(max_rows).copy()
    for column in columns:
        chart_df[column] = pd.to_numeric(chart_df[column], errors="coerce")
    finite = np.isfinite(chart_df[columns].to_numpy(dtype=np.float64)).all(axis=1)
    return chart_df.loc[finite].sort_values("t_sec").reset_index(drop=True)


def _padded_chart_domain(values):
    """Build a non-degenerate, finite scale domain for a numeric series."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot build a chart domain without finite values.")

    low = float(values.min())
    high = float(values.max())
    if high > low:
        padding = 0.03 * (high - low)
    else:
        padding = max(1.0, abs(low) * 0.01)
    return [low - padding, high + padding]


def build_live_signal_chart(chart_df, field, y_title):
    """Build a live signal chart, or return None until it has a real extent."""
    if (
        field not in chart_df.columns
        or len(chart_df) < 2
        or chart_df["t_sec"].nunique() < 2
    ):
        return None

    x_domain = _padded_chart_domain(chart_df["t_sec"])
    y_domain = _padded_chart_domain(chart_df[field])
    return (
        alt.Chart(chart_df)
        .mark_line(clip=True)
        .encode(
            x=alt.X(
                "t_sec:Q",
                title="Time (s)",
                scale=alt.Scale(domain=x_domain, nice=False),
            ),
            y=alt.Y(
                f"{field}:Q",
                title=y_title,
                scale=alt.Scale(domain=y_domain, nice=False),
            ),
        )
        .properties(height=230)
    )


st.set_page_config(page_title="BVP + EDA Stress Model Tester", layout="wide")
init_state()
st.title("BVP + EDA Stress Model Tester")
st.caption("MAX30102 + CJMCU-6701 • 60 s window • 5 s prediction step")

with st.sidebar:
    st.header("Model files")
    discovered_runs = discover_model_runs()
    if discovered_runs:
        selected_run = st.selectbox(
            "Model run",
            options=discovered_runs,
            format_func=lambda path: str(path.relative_to(MODELS_ROOT)),
            help="Runs are discovered automatically below the models folder.",
        )
        run_path = str(selected_run)
    else:
        run_path = st.text_input("Model run folder", value=str(DEFAULT_MODEL_RUN))
    try:
        bundle = load_model_bundle(run_path)
        manifest = bundle["manifest"]
        st.success(f"Loaded {len(bundle['models'])} model(s)")
        st.write("Notebook-selected model:", manifest.get("selected_model", "unknown"))
        st.write("Feature count:", len(manifest.get("feature_cols", [])))
        st.write("Window:", f"{manifest.get('window_sec', 60)} s")
        st.write("Calibration:", f"{manifest.get('calibration_sec', 300)} s")
        st.dataframe(pd.DataFrame({"model": list(bundle["models"].keys())}), hide_index=True, use_container_width=True)

        st.divider()
        st.header("Ensemble")
        available_model_names = list(bundle["models"].keys())
        default_ensemble = [m for m in ("Logistic Regression", "Random Forest") if m in available_model_names]
        ensemble_models = st.multiselect(
            "Average probabilities from (soft voting)",
            options=available_model_names,
            default=default_ensemble,
            help="Combined threshold is the mean of the selected models' own thresholds. Pick 2+ to enable.",
        )
    except ModuleNotFoundError as e:
        bundle = None
        ensemble_models = []
        install_command = (
            f"{shlex.quote(sys.executable)} -m pip install -r "
            f"{shlex.quote(str(APP_DIR / 'requirements.txt'))}"
        )
        st.error(f"Missing Python package: {e.name}")
        st.caption("Stop the app, run this command, then start it again:")
        st.code(install_command, language="bash")
    except Exception as e:
        bundle = None
        ensemble_models = []
        st.error(str(e))

    st.divider()
    st.header("Serial")
    ports = available_ports()
    port = st.selectbox("Port", options=ports if ports else ["/dev/ttyUSB0"])
    baud = st.number_input("Baud", 9600, 1000000, SERIAL_BAUD, 9600)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Connect", use_container_width=True):
            if serial is None:
                st.error("Install pyserial first.")
            else:
                disconnect_serial()
                try:
                    st.session_state.ser = serial.Serial(port=port, baudrate=int(baud), timeout=0)
                    time.sleep(0.3)
                    st.success("Connected")
                except Exception as e:
                    st.error(f"Serial error: {e}")
    with c2:
        if st.button("Disconnect", use_container_width=True):
            disconnect_serial()
    if st.button("Clear / restart calibration", use_container_width=True):
        clear_capture()

    auto_refresh = st.checkbox("Auto refresh", value=True)
    if auto_refresh and st_autorefresh is not None:
        st_autorefresh(interval=1000, key="stress_live_refresh")
    elif auto_refresh:
        st.warning("Live polling is paused because streamlit-autorefresh is not installed.")
        st.caption("Stop the app, install it in this Python environment, then restart:")
        if sys.platform == "win32":
            refresh_install_command = (
                f'& "{sys.executable}" -m pip install streamlit-autorefresh'
            )
            command_language = "powershell"
        else:
            refresh_install_command = (
                f"{shlex.quote(sys.executable)} -m pip install streamlit-autorefresh"
            )
            command_language = "bash"
        st.code(refresh_install_command, language=command_language)

live_tab, csv_tab, setup_tab = st.tabs(["Live sensor test", "CSV replay", "Setup"])

with live_tab:
    if st.session_state.ser is not None:
        poll_serial()
    df = samples_df()

    if len(df) == 0:
        st.info("Connect the ESP32 and wait for: DATA,t_ms,bvp,eda")
    else:
        latest_t = float(df["t_sec"].iloc[-1])
        duration = latest_t - float(df["t_sec"].iloc[0])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Captured", f"{duration:.1f} s")
        m2.metric("Samples", f"{len(df):,}")
        m3.metric("Latest IR", f"{df['bvp'].iloc[-1]:.0f}")
        m4.metric("Latest EDA", f"{df['eda'].iloc[-1]:.0f}")

        chart_df = prepare_live_chart_data(df)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("MAX30102 raw IR")
            bvp_chart = build_live_signal_chart(chart_df, "bvp", "Raw IR")
            if bvp_chart is None:
                st.caption("Waiting for at least two valid sensor samples...")
            else:
                st.altair_chart(bvp_chart, use_container_width=True)
        with c2:
            st.subheader("CJMCU-6701 raw EDA")
            eda_chart = build_live_signal_chart(chart_df, "eda", "Raw EDA")
            if eda_chart is None:
                st.caption("Waiting for at least two valid sensor samples...")
            else:
                st.altair_chart(eda_chart, use_container_width=True)

        recent_eda = chart_df["eda"].tail(100)
        if len(recent_eda) >= 20 and (recent_eda.abs() < 1.0).all():
            st.warning(
                "EDA has remained at 0. Check CJMCU-6701 power, ground, analog OUT, "
                "electrode contact, and the connection from analog OUT to ESP32 GPIO 34."
            )

        if bundle is None:
            st.warning("Load the model run folder before prediction.")
        else:
            calibration_sec = float(bundle["manifest"].get("calibration_sec", 300.0))
            window_sec = float(bundle["manifest"].get("window_sec", 60.0))

            # -------------------------------------------------------------
            # MANUAL PERSONAL CALIBRATION
            # -------------------------------------------------------------
            if not st.session_state.calibration_started:
                st.subheader("Personal calibration")
                st.info(
                    "Check that both sensors are giving stable readings. "
                    "When you are ready, start the 5-minute calibration."
                )
                if st.button(
                    "Start 5-Minute Calibration",
                    type="primary",
                    use_container_width=True,
                    key="start_calibration_btn",
                ):
                    start_calibration()
                    st.rerun()

            elif st.session_state.calibration is None:
                # start_calibration() resets stream_start_ms, so latest_t is
                # measured from the first fresh sample after the button press.
                df = samples_df()
                if len(df) == 0:
                    st.subheader("Personal calibration")
                    st.info("Calibration starting... waiting for sensor data.")
                else:
                    latest_t = float(df["t_sec"].iloc[-1])
                    elapsed = min(latest_t, calibration_sec)
                    remaining = max(0.0, calibration_sec - latest_t)

                    st.subheader("Personal calibration")
                    st.progress(min(1.0, max(0.0, elapsed / calibration_sec)))
                    c_elapsed, c_remaining = st.columns(2)
                    c_elapsed.metric("Elapsed", f"{elapsed:.0f} s")
                    c_remaining.metric("Remaining", f"{remaining:.0f} s")
                    st.caption(
                        "Stay relaxed and relatively still. Keep the MAX30102 finger "
                        "position and both EDA electrodes stable for the full 5 minutes."
                    )

                    if latest_t >= calibration_sec:
                        try:
                            with st.spinner("Computing personal calibration statistics..."):
                                st.session_state.calibration = compute_calibration(
                                    df, bundle["manifest"]
                                )
                            st.success(
                                f"Calibration complete: "
                                f"{st.session_state.calibration['n_windows']} windows"
                            )
                            st.rerun()
                        except Exception as e:
                            st.error(f"Calibration failed: {e}")

            # -------------------------------------------------------------
            # AFTER CALIBRATION: USER CAN START THE CONVERSATION
            # -------------------------------------------------------------
            if st.session_state.calibration is not None:
                st.success("Calibration complete ✅")
                st.info("You can now start speaking or typing.")

                # Re-read data because Streamlit may have rerun immediately
                # after calibration was computed.
                df = samples_df()
                if len(df):
                    latest_t = float(df["t_sec"].iloc[-1])

                    # The calibration uses only the first 300 s. The first
                    # stress prediction uses a NEW 60 s post-calibration window.
                    first_prediction_t = calibration_sec + window_sec

                    if latest_t >= first_prediction_t:
                        end_t = latest_t
                        start_t = end_t - window_sec
                        last_p = st.session_state.last_prediction_at
                        step_sec = float(bundle["manifest"].get("step_sec", STEP_SEC))

                        if last_p is None or end_t - last_p >= step_sec:
                            try:
                                feat, raw32, diag = extract_one_window(
                                    df,
                                    start_t,
                                    end_t,
                                    bundle["manifest"]["feature_cols"],
                                )
                                results = predict_all(
                                    bundle,
                                    feat,
                                    raw32,
                                    st.session_state.calibration,
                                    ensemble_models,
                                )
                                st.session_state.last_prediction_at = end_t
                                st.session_state.prediction_history.append(
                                    (end_t, results.copy(), diag)
                                )
                                st.session_state.prediction_history = (
                                    st.session_state.prediction_history[-100:]
                                )
                            except Exception as e:
                                st.error(f"Prediction failed: {e}")

                        if st.session_state.prediction_history:
                            _, results, diag = st.session_state.prediction_history[-1]
                            st.subheader("Current prediction")
                            show = results.copy()
                            show["stress_probability"] = show["stress_probability"].map(
                                lambda x: f"{x:.3f}"
                            )
                            show["threshold"] = show["threshold"].map(
                                lambda x: f"{x:.3f}"
                            )
                            st.dataframe(
                                show,
                                hide_index=True,
                                use_container_width=True,
                            )
                            st.bar_chart(
                                results.set_index("model")[["stress_probability"]]
                            )

                            d1, d2, d3 = st.columns(3)
                            d1.metric("Pulse peaks", diag["pulse_peaks"])
                            d2.metric("Valid IBI", diag["valid_ibi"])
                            d3.metric("Median IR", f"{diag['median_ir']:.0f}")
                            if diag["median_ir"] < 20000:
                                st.warning(
                                    "MAX30102 signal looks weak / finger may be misplaced."
                                )

                            selected = bundle["manifest"].get("selected_model")
                            selected_row = results[results["model"] == selected]
                            if len(selected_row):
                                r = selected_row.iloc[0]
                                msg = (
                                    f"Selected model ({selected}): {r['prediction']} "
                                    f"• p={r['stress_probability']:.3f}"
                                )
                                if r["prediction"] == "STRESS":
                                    st.error(msg)
                                else:
                                    st.success(msg)

                            st.download_button(
                                "Download captured raw CSV",
                                data=df.to_csv(index=False).encode("utf-8"),
                                file_name="sensor_capture.csv",
                                mime="text/csv",
                            )
                    else:
                        need = max(0.0, first_prediction_t - latest_t)
                        st.info(
                            "Sensor stress model is warming up. "
                            f"Collect {need:.1f} more seconds for the first "
                            "post-calibration 60-second prediction window."
                        )

with csv_tab:
    st.write("Upload a raw capture CSV with columns **t_sec, bvp, eda**. Use at least 6 minutes.")
    uploaded = st.file_uploader("Raw capture CSV", type=["csv"])
    if uploaded is not None and bundle is not None:
        try:
            replay = pd.read_csv(uploaded).sort_values("t_sec").dropna(subset=["t_sec", "bvp", "eda"])
            if not {"t_sec", "bvp", "eda"}.issubset(replay.columns):
                raise ValueError("CSV must contain t_sec, bvp, eda")
            st.dataframe(replay.head(), use_container_width=True)
            if st.button("Run replay prediction"):
                cal = compute_calibration(replay, bundle["manifest"])
                end_t = float(replay["t_sec"].max())
                start_t = end_t - float(bundle["manifest"].get("window_sec", 60.0))
                feat, raw32, diag = extract_one_window(replay, start_t, end_t, bundle["manifest"]["feature_cols"])
                results = predict_all(bundle, feat, raw32, cal, ensemble_models)
                st.dataframe(results, hide_index=True, use_container_width=True)
                st.bar_chart(results.set_index("model")[["stress_probability"]])
                st.json(diag)

                if XAI_AVAILABLE:
                    with st.expander("Explain this prediction (SHAP / Grad-CAM)"):
                        model_name = st.selectbox(
                            "Model to explain",
                            list(bundle["models"].keys()),
                            key="xai_model_pick",
                        )
                        item = bundle["models"][model_name]
                        feature_cols = bundle["manifest"]["feature_cols"]
                        x_feat_row = (feat - cal["feature_mu"]) / cal["feature_sd"]

                        if item["kind"] in ("classical", "mlp"):
                            background = shap_explain.build_feature_background(
                                extract_one_window, replay, bundle["manifest"], cal
                            )
                            if item["kind"] == "classical":
                                tree_based = model_name in (
                                    "Random Forest", "HistGradientBoosting",
                                )
                                exp = shap_explain.explain_classical(
                                    item["model"], x_feat_row, background,
                                    feature_cols, tree_based,
                                )
                            else:
                                exp = shap_explain.explain_feature_mlp(
                                    item["model"], item["imputer"], item["scaler"],
                                    x_feat_row, background, feature_cols,
                                )
                            shap_df = pd.DataFrame(
                                {"feature": exp["feature"], "shap_value": exp["shap_value"]}
                            )
                            shap_df = shap_df.reindex(
                                shap_df["shap_value"].abs().sort_values(ascending=False).index
                            )
                            st.bar_chart(shap_df.set_index("feature")["shap_value"])
                            st.caption(
                                f"Base value {exp['base_value']:.3f} -> "
                                f"predicted {exp['predicted_value']:.3f}. "
                                "Positive bars push toward STRESS."
                            )
                        elif item["kind"] == "cnn":
                            x_raw_full = (raw32.astype(np.float64) - cal["raw_mu"]) / cal["raw_sd"]
                            x_raw_norm = (
                                (x_raw_full - item["global_mu"]) / item["global_sd"]
                            ).astype(np.float32)
                            cam = gradcam.cnn_lstm_gradcam(item["model"], x_raw_norm)
                            t_axis = np.arange(len(cam["saliency_1920"])) / TARGET_FS

                            def _minmax(v):
                                v = v.astype(np.float64)
                                return (v - v.min()) / max(v.max() - v.min(), 1e-8)

                            overlay_df = pd.DataFrame({
                                "t_sec": t_axis,
                                "eda (norm)": _minmax(raw32[:, 0]),
                                "bvp (norm)": _minmax(raw32[:, 1]),
                                "saliency": cam["saliency_1920"],
                            }).set_index("t_sec")
                            st.line_chart(overlay_df)
                            st.caption(
                                f"Grad-CAM w.r.t. the STRESS logit "
                                f"({cam['stress_logit']:.2f} -> p={cam['stress_probability']:.3f}). "
                                "Higher = more influential seconds."
                            )
                else:
                    st.caption(
                        "Install `shap` "
                        "(Xai/c1_stress_detection_bvp/BVP_EDA/requirements.txt) "
                        "to enable the Explain section."
                    )
        except Exception as e:
            st.error(str(e))

with setup_tab:
    st.subheader("Model discovery")
    st.write(
        "The app finds every exported run below the project-level "
        "`models/c1_stress_detection_bvp/` folder. Each run must have this structure:"
    )
    st.code("""models/c1_stress_detection_bvp/
└── BVP_EDA/any-run/
    ├── run_manifest.json
    ├── dl_params.json
    └── models/
        ├── logistic_regression.pkl
        ├── random_forest.pkl
        ├── svm_rbf.pkl
        ├── histgradientboosting.pkl
        ├── feature_mlp.pt
        ├── feature_mlp_prep.pkl
        ├── cnn_lstm.pt
        └── cnn_lstm_prep.pkl
""", language="text")
    st.subheader("Run")
    st.code("""python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r streamlit_app/c1_stress_detection_bvp/requirements.txt
python -m streamlit run streamlit_app/c1_stress_detection_bvp/app.py
""", language="bash")
    st.warning(
        "Prototype limitation: the notebook's EDA peak detector uses a fixed 0.01 prominence in WESAD EDA units. "
        "CJMCU ADC units differ, so eda_num_peaks/eda_peak_amplitude require real-device validation."
    )