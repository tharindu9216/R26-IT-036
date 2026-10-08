"""Continuous serial sensor service for the C1 Feature-MLP predictor."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .inference import (
    BVP_SOURCE_MAX30102,
    C1FeatureMLPPredictor,
    C1Prediction,
    CalibrationStats,
    SensorSample,
    validate_bvp_source,
)
from xai import FeatureAttribution


UINT32_MODULUS = 2**32


class SensorProtocolError(ValueError):
    """The ESP32 sent a fatal protocol/error line."""


class DeviceClockReset(RuntimeError):
    """The ESP32 clock moved backwards without a uint32 rollover."""


class ServiceState(str, Enum):
    STOPPED = "stopped"
    CONNECTING = "connecting"
    CALIBRATING = "calibrating"
    WARMING_UP = "warming_up"
    RUNNING = "running"
    ERROR = "error"


@dataclass(frozen=True)
class C1ServiceConfig:
    port: str
    baud_rate: int = 230400
    serial_timeout: float = 0.1
    process_interval: float = 0.2
    expected_sample_rate: int = 100
    buffer_seconds: int = 420
    smoothing_window: int = 3
    smoothing_required: int = 2
    # Calibration is measured in sensor time, not wall time. If the serial
    # connection stays open but the device timestamp does not advance for this
    # long, fail explicitly instead of leaving the UI frozen forever.
    stream_stall_seconds: float = 5.0
    # Which hardware produced the stream. A replayed Empatica E4 session is
    # still valid data, but its BVP is on a different scale from the ESP32's
    # MAX30102, so the finger-presence check has to know which it is.
    bvp_source: str = BVP_SOURCE_MAX30102

    def __post_init__(self) -> None:
        validate_bvp_source(self.bvp_source)
        if not self.port.strip():
            raise ValueError("A serial port is required")
        if self.baud_rate <= 0 or self.expected_sample_rate <= 0:
            raise ValueError("Baud rate and expected sample rate must be positive")
        if self.buffer_seconds < 365:
            raise ValueError("C1 buffer_seconds must cover calibration plus warm-up")
        if self.smoothing_window <= 0:
            raise ValueError("smoothing_window must be positive")
        if not 1 <= self.smoothing_required <= self.smoothing_window:
            raise ValueError(
                "smoothing_required must be between 1 and smoothing_window"
            )
        if self.stream_stall_seconds <= 0:
            raise ValueError("stream_stall_seconds must be positive")


@dataclass(frozen=True)
class ServiceSnapshot:
    state: ServiceState
    connected: bool
    sample_count: int
    elapsed_seconds: float
    calibration_remaining_seconds: float
    warmup_remaining_seconds: float
    calibration: CalibrationStats | None
    last_prediction: C1Prediction | None
    recent_predictions: tuple[C1Prediction, ...]
    smoothed_is_stressed: bool | None
    smoothed_probability: float | None
    last_error: str | None
    serial_line_count: int
    invalid_line_count: int
    data_progress_age_seconds: float | None
    stream_stalled: bool


def smooth_prediction_history(
    predictions: Sequence[C1Prediction],
    required_votes: int,
) -> tuple[bool | None, float | None]:
    """Return majority-vote label and mean probability for recent windows."""

    if required_votes <= 0:
        raise ValueError("required_votes must be positive")
    if not predictions:
        return None, None
    active_required_votes = min(required_votes, len(predictions))
    positive_votes = sum(item.is_stressed for item in predictions)
    probability = sum(item.calibrated_probability for item in predictions) / len(
        predictions
    )
    return positive_votes >= active_required_votes, probability


class SerialDataParser:
    """Parse ``DATA,<millis>,<MAX30102_IR>,<GSR_ADC>`` firmware lines."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._first_unwrapped_ms: int | None = None
        self._last_raw_ms: int | None = None
        self._rollover_offset = 0

    def parse(self, raw_line: bytes | str) -> SensorSample | None:
        line = (
            raw_line.decode("utf-8", errors="ignore")
            if isinstance(raw_line, bytes)
            else raw_line
        ).strip()
        if not line or line == "READY":
            return None
        if line.startswith("ERROR"):
            raise SensorProtocolError(line)
        if not line.startswith("DATA,"):
            return None

        fields = line.split(",")
        if len(fields) != 4:
            return None
        try:
            raw_ms = int(fields[1])
            bvp = float(fields[2])
            eda = float(fields[3])
        except ValueError:
            return None
        if raw_ms < 0 or raw_ms >= UINT32_MODULUS:
            return None

        if self._last_raw_ms is not None and raw_ms < self._last_raw_ms:
            backwards = self._last_raw_ms - raw_ms
            if backwards > UINT32_MODULUS // 2:
                self._rollover_offset += UINT32_MODULUS
            else:
                self.reset()
                raise DeviceClockReset("ESP32 millis() restarted; calibration was reset")

        unwrapped_ms = raw_ms + self._rollover_offset
        if self._first_unwrapped_ms is None:
            self._first_unwrapped_ms = unwrapped_ms
        self._last_raw_ms = raw_ms
        return SensorSample(
            t_sec=(unwrapped_ms - self._first_unwrapped_ms) / 1000.0,
            bvp=bvp,
            eda=eda,
        )


class SerialSensorService:
    """Read the ESP32 continuously and publish a prediction every five seconds.

    Timeline:

    * 0-300 s: personal calibration
    * 300-360 s: collect a fresh, post-calibration 60-second window
    * 360 s onward: predict every 5 seconds from the latest 60-second window
    """

    def __init__(
        self,
        config: C1ServiceConfig,
        predictor: C1FeatureMLPPredictor | None = None,
        *,
        serial_factory: Callable[..., Any] | None = None,
        on_prediction: Callable[[C1Prediction], None] | None = None,
        on_state_change: Callable[[ServiceSnapshot], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self.config = config
        self.predictor = predictor or C1FeatureMLPPredictor()
        self._serial_factory = serial_factory
        self._on_prediction = on_prediction
        self._on_state_change = on_state_change
        self._on_error = on_error

        max_samples = config.expected_sample_rate * config.buffer_seconds
        self._samples: deque[SensorSample] = deque(maxlen=max_samples)
        self._parser = SerialDataParser()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._serial: Any | None = None
        self._reader_thread: threading.Thread | None = None
        self._processor_thread: threading.Thread | None = None

        self._state = ServiceState.STOPPED
        self._calibration: CalibrationStats | None = None
        self._calibration_start_t: float | None = None
        self._next_prediction_at: float | None = None
        self._last_prediction: C1Prediction | None = None
        self._prediction_history: deque[C1Prediction] = deque(
            maxlen=config.smoothing_window
        )
        self._last_error: str | None = None
        self._stream_started_at: float | None = None
        self._last_timestamp_advance_at: float | None = None
        self._last_sensor_t: float | None = None
        self._serial_line_count = 0
        self._invalid_line_count = 0

    @property
    def state(self) -> ServiceState:
        with self._lock:
            return self._state

    @property
    def is_running(self) -> bool:
        return self.state not in {ServiceState.STOPPED, ServiceState.ERROR}

    def snapshot(self) -> ServiceSnapshot:
        with self._lock:
            now = time.monotonic()
            latest_t = self._samples[-1].t_sec if self._samples else 0.0
            start_t = self._calibration_start_t
            elapsed = max(0.0, latest_t - start_t) if start_t is not None else 0.0
            calibration_sec = float(self.predictor.manifest.get("calibration_sec", 300.0))
            window_sec = float(self.predictor.manifest.get("window_sec", 60.0))
            calibration_remaining = (
                max(0.0, calibration_sec - elapsed)
                if self._calibration is None
                else 0.0
            )
            warmup_remaining = 0.0
            if self._calibration is not None and self._last_prediction is None:
                warmup_remaining = max(
                    0.0,
                    self._calibration.end_t + window_sec - latest_t,
                )
            recent_predictions = tuple(self._prediction_history)
            smoothed_is_stressed, smoothed_probability = smooth_prediction_history(
                recent_predictions,
                self.config.smoothing_required,
            )
            age_reference = self._last_timestamp_advance_at or self._stream_started_at
            valid_sample_age = (
                max(0.0, now - age_reference) if age_reference is not None else None
            )
            is_replay = self.config.port.startswith("replay:")
            monitored_states = {
                ServiceState.CONNECTING,
                ServiceState.CALIBRATING,
                ServiceState.WARMING_UP,
            }
            if not is_replay:
                monitored_states.add(ServiceState.RUNNING)
            stream_stalled = bool(
                self._serial is not None
                and self._state in monitored_states
                and valid_sample_age is not None
                and valid_sample_age >= self.config.stream_stall_seconds
            )
            return ServiceSnapshot(
                state=self._state,
                connected=self._serial is not None,
                sample_count=len(self._samples),
                elapsed_seconds=elapsed,
                calibration_remaining_seconds=calibration_remaining,
                warmup_remaining_seconds=warmup_remaining,
                calibration=self._calibration,
                last_prediction=self._last_prediction,
                recent_predictions=recent_predictions,
                smoothed_is_stressed=smoothed_is_stressed,
                smoothed_probability=smoothed_probability,
                last_error=self._last_error,
                serial_line_count=self._serial_line_count,
                invalid_line_count=self._invalid_line_count,
                data_progress_age_seconds=valid_sample_age,
                stream_stalled=stream_stalled,
            )

    def recent_samples(
        self,
        *,
        duration_seconds: float = 8.0,
        max_points: int = 420,
    ) -> tuple[SensorSample, ...]:
        """Return a bounded, display-only slice of the latest sensor signal.

        The inference buffer can contain several minutes at the full sensor
        rate.  Status polling only needs enough points to draw a smooth live
        trace, so this method crops by sensor time and downsamples while still
        retaining both endpoints.
        """

        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if max_points <= 0:
            raise ValueError("max_points must be positive")

        with self._lock:
            if not self._samples:
                return ()
            cutoff = self._samples[-1].t_sec - duration_seconds
            visible: list[SensorSample] = []
            for sample in reversed(self._samples):
                if sample.t_sec < cutoff:
                    break
                visible.append(sample)
            visible.reverse()

        if len(visible) <= max_points:
            return tuple(visible)
        if max_points == 1:
            return (visible[-1],)

        final_index = len(visible) - 1
        return tuple(
            visible[round(index * final_index / (max_points - 1))]
            for index in range(max_points)
        )

    def explain_last_prediction(self, *, n_steps: int = 64) -> list[FeatureAttribution]:
        """Integrated Gradients attribution for the most recent prediction.

        Raises ``ValueError`` if there is no prediction yet (still
        calibrating/warming up).
        """

        snapshot = self.snapshot()
        if snapshot.last_prediction is None or snapshot.calibration is None:
            raise ValueError("No prediction yet -- still calibrating/warming up")
        return self.predictor.explain(
            snapshot.last_prediction, snapshot.calibration, n_steps=n_steps
        )

    def start(self) -> None:
        with self._lock:
            if self._state not in {ServiceState.STOPPED, ServiceState.ERROR}:
                return
            self._stop_event.clear()
            self._state = ServiceState.CONNECTING
            self._last_error = None
        self._notify_state_change()

        try:
            factory = self._serial_factory
            if factory is None:
                try:
                    import serial
                except ImportError as exc:
                    raise RuntimeError(
                        "pyserial is required; install backend/c1/requirements.txt"
                    ) from exc
                factory = serial.Serial
            connection = factory(
                port=self.config.port,
                baudrate=self.config.baud_rate,
                timeout=self.config.serial_timeout,
            )
            try:
                connection.reset_input_buffer()
            except (AttributeError, OSError):
                pass
        except Exception as exc:
            self._set_terminal_error(exc)
            raise

        with self._lock:
            self._serial = connection
        self.restart_calibration(reset_input=False)
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="c1-serial-reader",
            daemon=True,
        )
        self._processor_thread = threading.Thread(
            target=self._processor_loop,
            name="c1-prediction-worker",
            daemon=True,
        )
        self._reader_thread.start()
        self._processor_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        for thread in (self._reader_thread, self._processor_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=2.0)
        with self._lock:
            connection = self._serial
            self._serial = None
            self._state = ServiceState.STOPPED
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
        self._notify_state_change()

    def restart_calibration(self, *, reset_input: bool = True) -> None:
        """Clear all personal state and begin a new five-minute calibration."""

        with self._lock:
            self._samples.clear()
            self._parser.reset()
            self._calibration = None
            self._calibration_start_t = None
            self._next_prediction_at = None
            self._last_prediction = None
            self._prediction_history.clear()
            self._last_error = None
            self._state = ServiceState.CALIBRATING
            self._stream_started_at = time.monotonic()
            self._last_timestamp_advance_at = None
            self._last_sensor_t = None
            self._serial_line_count = 0
            self._invalid_line_count = 0
            connection = self._serial
        if reset_input and connection is not None:
            try:
                connection.reset_input_buffer()
            except (AttributeError, OSError):
                pass
        self._notify_state_change()

    def ingest_line(self, raw_line: bytes | str) -> SensorSample | None:
        """Parse and store one firmware line; public for adapters and tests."""

        text = (
            raw_line.decode("utf-8", errors="ignore")
            if isinstance(raw_line, bytes)
            else raw_line
        ).strip()
        if text:
            with self._lock:
                self._serial_line_count += 1

        try:
            sample = self._parser.parse(raw_line)
        except DeviceClockReset:
            self.restart_calibration(reset_input=False)
            return None
        if sample is None:
            if text and text != "READY":
                with self._lock:
                    self._invalid_line_count += 1
            return None
        now = time.monotonic()
        with self._lock:
            if self._calibration_start_t is None:
                self._calibration_start_t = sample.t_sec
            self._samples.append(sample)
            if self._last_sensor_t is None or sample.t_sec > self._last_sensor_t:
                self._last_sensor_t = sample.t_sec
                self._last_timestamp_advance_at = now
        return sample

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                connection = self._serial
            if connection is None:
                return
            try:
                line = connection.readline()
                if line:
                    self.ingest_line(line)
            except Exception as exc:
                self._set_terminal_error(exc)
                self._stop_event.set()
                return

    def _processor_loop(self) -> None:
        while not self._stop_event.wait(self.config.process_interval):
            if self.state == ServiceState.ERROR:
                continue
            snapshot = self.snapshot()
            if snapshot.stream_stalled:
                if snapshot.sample_count == 0:
                    if snapshot.serial_line_count > 0:
                        message = (
                            "Serial is connected, but no valid DATA,<millis>,<ir>,<eda> "
                            "samples were received. Flash streamlit_serial.ino and use "
                            "230400 baud."
                        )
                    else:
                        message = (
                            "Serial is connected, but the sensor is not sending data. "
                            "Check the USB cable, port, firmware, and 230400 baud rate."
                        )
                else:
                    if self.config.port.startswith("replay:"):
                        message = (
                            f"Uploaded recording ended after "
                            f"{snapshot.elapsed_seconds:.1f} seconds. This C1 model needs "
                            "at least 360 seconds of overlapping BVP and EDA: 300 seconds "
                            "for personal calibration plus a fresh 60-second window."
                        )
                    else:
                        message = (
                            f"Sensor data stopped after {snapshot.elapsed_seconds:.1f} seconds. "
                            "Check the USB connection and make sure the device continuously "
                            "emits DATA,<millis>,<ir>,<eda> lines."
                        )
                self._set_terminal_error(RuntimeError(message))
                continue
            with self._lock:
                samples = tuple(self._samples)
                calibration = self._calibration
                calibration_start = self._calibration_start_t
                next_prediction_at = self._next_prediction_at
            if not samples or calibration_start is None:
                continue
            latest_t = samples[-1].t_sec

            if calibration is None:
                calibration_sec = float(
                    self.predictor.manifest.get("calibration_sec", 300.0)
                )
                # Wait until sensor time has actually crossed the boundary.
                # Starting up to 250 ms early raced the window extractor on
                # lower-rate real devices and produced a false coverage error.
                if latest_t < calibration_start + calibration_sec:
                    continue
                try:
                    calibration = self.predictor.calibrate(
                        samples,
                        start_t=calibration_start,
                        bvp_source=self.config.bvp_source,
                    )
                except Exception as exc:
                    self._set_terminal_error(exc)
                    continue
                window_sec = float(self.predictor.manifest.get("window_sec", 60.0))
                with self._lock:
                    self._calibration = calibration
                    self._next_prediction_at = calibration.end_t + window_sec
                    self._state = ServiceState.WARMING_UP
                self._notify_state_change()
                continue

            if next_prediction_at is None or latest_t < next_prediction_at:
                continue
            window_sec = float(self.predictor.manifest.get("window_sec", 60.0))
            step_sec = float(self.predictor.manifest.get("step_sec", 5.0))
            try:
                prediction = self.predictor.predict(
                    samples,
                    calibration,
                    start_t=next_prediction_at - window_sec,
                    end_t=next_prediction_at,
                    bvp_source=self.config.bvp_source,
                )
            except Exception as exc:
                self._report_error(exc)
                with self._lock:
                    self._next_prediction_at = next_prediction_at + step_sec
                continue

            with self._lock:
                self._last_prediction = prediction
                self._prediction_history.append(prediction)
                self._last_error = None
                self._next_prediction_at = next_prediction_at + step_sec
                self._state = ServiceState.RUNNING
            self._safe_callback(self._on_prediction, prediction)
            self._notify_state_change()

    def _set_terminal_error(self, error: Exception) -> None:
        with self._lock:
            self._state = ServiceState.ERROR
            self._last_error = str(error)
        self._safe_callback(self._on_error, error)
        self._notify_state_change()

    def _report_error(self, error: Exception) -> None:
        with self._lock:
            self._last_error = str(error)
        self._safe_callback(self._on_error, error)
        self._notify_state_change()

    def _notify_state_change(self) -> None:
        self._safe_callback(self._on_state_change, self.snapshot())

    @staticmethod
    def _safe_callback(callback: Callable[[Any], None] | None, value: Any) -> None:
        if callback is None:
            return
        try:
            callback(value)
        except Exception:
            # Application callbacks must never stop sensor acquisition.
            pass

    def __enter__(self) -> "SerialSensorService":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
