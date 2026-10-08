"""C1 live BVP/EDA stress inference package."""

from .inference import (
    BVP_SOURCE_E4,
    BVP_SOURCE_MAX30102,
    C1FeatureMLPPredictor,
    C1Prediction,
    CalibrationStats,
    SensorSample,
    SignalQualityError,
    WindowDiagnostics,
    compute_calibration,
    extract_feature_window,
)
from .e4 import convert_e4_to_firmware_lines
from .replay import ReplaySerialConnection, parse_recording, recording_duration_seconds
from .sensor_service import (
    C1ServiceConfig,
    SerialDataParser,
    SerialSensorService,
    ServiceSnapshot,
    ServiceState,
)

__all__ = [
    "C1FeatureMLPPredictor",
    "C1Prediction",
    "C1ServiceConfig",
    "CalibrationStats",
    "ReplaySerialConnection",
    "SensorSample",
    "convert_e4_to_firmware_lines",
    "SerialDataParser",
    "SerialSensorService",
    "ServiceSnapshot",
    "ServiceState",
    "SignalQualityError",
    "WindowDiagnostics",
    "BVP_SOURCE_E4",
    "BVP_SOURCE_MAX30102",
    "compute_calibration",
    "extract_feature_window",
    "parse_recording",
    "recording_duration_seconds",
]
