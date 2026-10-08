from __future__ import annotations

import unittest
import time
from types import SimpleNamespace

import numpy as np

from c1 import (
    BVP_SOURCE_E4,
    BVP_SOURCE_MAX30102,
    C1FeatureMLPPredictor,
    C1ServiceConfig,
    SerialDataParser,
    SerialSensorService,
    ServiceState,
    convert_e4_to_firmware_lines,
    recording_duration_seconds,
)
from c1.inference import SignalQualityError, _resample_by_time, extract_feature_window
from c1.sensor_service import DeviceClockReset, SensorProtocolError
from c1.sensor_service import smooth_prediction_history


def synthetic_sensor_capture(duration_seconds: float = 360.0) -> np.ndarray:
    timestamps = np.arange(0.0, duration_seconds + 0.1, 0.1)
    # Strong raw IR with a stable 72 BPM pulse plus a small harmonic.
    bvp = (
        50000.0
        + 3500.0 * np.sin(2.0 * np.pi * 1.2 * timestamps)
        + 350.0 * np.sin(2.0 * np.pi * 2.4 * timestamps)
    )
    # Slowly varying ADC-scale EDA with regular phasic responses.
    eda = (
        1900.0
        + 20.0 * np.sin(2.0 * np.pi * 0.01 * timestamps)
        + 5.0 * np.maximum(0.0, np.sin(2.0 * np.pi * 0.12 * timestamps))
    )
    return np.column_stack((timestamps, bvp, eda))


def synthetic_e4_capture(duration_seconds: float = 400.0) -> tuple[bytes, bytes]:
    """A band-passed, zero-centred BVP + microsiemens EDA pair, as the E4 exports.

    Deliberately on E4's scale rather than the MAX30102's: the point of these
    tests is that the finger-presence check must not be applied to it.
    """

    bvp_t = np.arange(0.0, duration_seconds, 1.0 / 64.0)
    bvp = 60.0 * np.sin(2.0 * np.pi * 1.2 * bvp_t) + 6.0 * np.sin(
        2.0 * np.pi * 2.4 * bvp_t
    )
    eda_t = np.arange(0.0, duration_seconds, 1.0 / 4.0)
    eda = (
        2.4
        + 0.05 * np.sin(2.0 * np.pi * 0.01 * eda_t)
        + 0.02 * np.maximum(0.0, np.sin(2.0 * np.pi * 0.12 * eda_t))
    )

    def as_csv(rate: float, values: np.ndarray) -> bytes:
        rows = ["1500000000.000000", f"{rate:.6f}"]
        rows.extend(f"{value:.6f}" for value in values)
        newline = chr(10)
        return (newline.join(rows) + newline).encode()

    return as_csv(64.0, bvp), as_csv(4.0, eda)


def synthetic_tabular_e4_capture(duration_seconds: float = 400.0) -> tuple[bytes, bytes]:
    bvp_t = np.arange(92.234375, 92.234375 + duration_seconds, 1.0 / 64.0)
    eda_t = np.arange(92.25, 92.25 + duration_seconds, 1.0 / 4.0)
    bvp_rows = ["time_seconds,BVP"] + [
        f"{time:.6f},{np.sin(time):.6f}" for time in bvp_t
    ]
    eda_rows = ["time_seconds,EDA"] + [
        f"{time:.6f},{2.0 + 0.1 * np.sin(time):.6f}" for time in eda_t
    ]
    return (
        ("\n".join(bvp_rows) + "\n").encode(),
        ("\n".join(eda_rows) + "\n").encode(),
    )


class SerialProtocolTests(unittest.TestCase):
    def test_parses_raw_firmware_line(self) -> None:
        parser = SerialDataParser()
        first = parser.parse("DATA,1000,50123,1987")
        second = parser.parse(b"DATA,1100,50200,1991\n")
        self.assertEqual(first.t_sec, 0.0)
        self.assertAlmostEqual(second.t_sec, 0.1)
        self.assertEqual(second.bvp, 50200.0)
        self.assertEqual(second.eda, 1991.0)

    def test_ignores_status_and_malformed_lines(self) -> None:
        parser = SerialDataParser()
        self.assertIsNone(parser.parse("READY"))
        self.assertIsNone(parser.parse("not data"))
        self.assertIsNone(parser.parse("DATA,bad,1,2"))
        with self.assertRaises(SensorProtocolError):
            parser.parse("ERROR,MAX30102_NOT_FOUND")

    def test_detects_device_restart(self) -> None:
        parser = SerialDataParser()
        parser.parse("DATA,5000,50000,1900")
        with self.assertRaises(DeviceClockReset):
            parser.parse("DATA,1000,50000,1900")

    def test_recording_duration_uses_data_timestamps(self) -> None:
        lines = [
            b"DATA,1000,50000,1900",
            b"DATA,2600,50100,1901",
            b"DATA,4600,50200,1902",
        ]
        self.assertAlmostEqual(recording_duration_seconds(lines), 3.6)


class FeatureMLPIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = C1FeatureMLPPredictor()
        cls.capture = synthetic_sensor_capture()

    def test_exported_configuration(self) -> None:
        self.assertEqual(len(self.predictor.feature_cols), 18)
        self.assertEqual(self.predictor.threshold, 0.295)
        self.assertEqual(self.predictor.manifest["calibration_sec"], 300.0)
        self.assertEqual(self.predictor.manifest["calibration_windows"], 49)

    def test_calibration_and_fresh_window_prediction(self) -> None:
        calibration = self.predictor.calibrate(self.capture, start_t=0.0)
        self.assertEqual(calibration.n_windows, 49)
        self.assertEqual(calibration.end_t, 300.0)
        prediction = self.predictor.predict(
            self.capture,
            calibration,
            start_t=300.0,
            end_t=360.0,
        )
        self.assertGreaterEqual(prediction.raw_probability, 0.0)
        self.assertLessEqual(prediction.raw_probability, 1.0)
        self.assertGreaterEqual(prediction.calibrated_probability, 0.0)
        self.assertLessEqual(prediction.calibrated_probability, 1.0)
        self.assertEqual(prediction.window_start_t, 300.0)
        self.assertEqual(prediction.window_end_t, 360.0)

    def test_resampling_uses_a_valid_sample_just_before_the_window_edge(self) -> None:
        timestamps = np.concatenate(
            (np.asarray([-0.01]), np.arange(0.30, 60.31, 0.10))
        )
        values = np.sin(timestamps)
        resampled = _resample_by_time(timestamps, values, 4.0, 0.0, 60.0)
        self.assertEqual(len(resampled), 240)
        self.assertTrue(np.all(np.isfinite(resampled)))


class ServiceConfigurationTests(unittest.TestCase):
    def test_manifest_timing_defaults_are_compatible(self) -> None:
        config = C1ServiceConfig(port="TEST")
        self.assertEqual(config.baud_rate, 230400)
        self.assertGreaterEqual(config.buffer_seconds, 365)
        self.assertEqual(config.smoothing_window, 3)
        self.assertEqual(config.smoothing_required, 2)
        self.assertEqual(ServiceState.RUNNING.value, "running")

    def test_two_of_three_smoothing(self) -> None:
        history = [
            SimpleNamespace(is_stressed=True, calibrated_probability=0.8),
            SimpleNamespace(is_stressed=False, calibrated_probability=0.2),
            SimpleNamespace(is_stressed=True, calibrated_probability=0.7),
        ]
        is_stressed, probability = smooth_prediction_history(history, 2)
        self.assertTrue(is_stressed)
        self.assertAlmostEqual(probability, (0.8 + 0.2 + 0.7) / 3)

    def test_stream_diagnostics_count_valid_and_ignored_lines(self) -> None:
        predictor = SimpleNamespace(
            manifest={"calibration_sec": 300.0, "window_sec": 60.0}
        )
        service = SerialSensorService(
            C1ServiceConfig(port="TEST"), predictor=predictor
        )
        service.restart_calibration(reset_input=False)
        service.ingest_line("READY")
        service.ingest_line("ECG: BPM 72")
        service.ingest_line("DATA,1000,50000,1900")
        service.ingest_line("DATA,1010,50100,1901")

        snapshot = service.snapshot()
        self.assertEqual(snapshot.serial_line_count, 4)
        self.assertEqual(snapshot.invalid_line_count, 1)
        self.assertEqual(snapshot.sample_count, 2)
        self.assertAlmostEqual(snapshot.elapsed_seconds, 0.01)

    def test_recent_samples_are_cropped_and_downsampled_for_live_display(self) -> None:
        predictor = SimpleNamespace(
            manifest={"calibration_sec": 300.0, "window_sec": 60.0}
        )
        service = SerialSensorService(
            C1ServiceConfig(port="TEST"), predictor=predictor
        )
        service.restart_calibration(reset_input=False)
        for index in range(1001):
            service.ingest_line(
                f"DATA,{index * 10},{50000 + index},{1900 + index}"
            )

        samples = service.recent_samples(duration_seconds=8.0, max_points=100)
        self.assertEqual(len(samples), 100)
        self.assertAlmostEqual(samples[0].t_sec, 2.0)
        self.assertAlmostEqual(samples[-1].t_sec, 10.0)
        self.assertEqual(samples[-1].bvp, 51000.0)
        self.assertEqual(samples[-1].eda, 2900.0)

    def test_recent_samples_validates_display_limits(self) -> None:
        service = SerialSensorService(C1ServiceConfig(port="TEST"))
        with self.assertRaisesRegex(ValueError, "duration_seconds"):
            service.recent_samples(duration_seconds=0)
        with self.assertRaisesRegex(ValueError, "max_points"):
            service.recent_samples(max_points=0)

    def test_non_data_serial_stream_fails_instead_of_freezing(self) -> None:
        class InvalidSerial:
            def reset_input_buffer(self) -> None:
                pass

            def readline(self) -> bytes:
                time.sleep(0.002)
                return b"ECG: BPM 72\n"

            def close(self) -> None:
                pass

        predictor = SimpleNamespace(
            manifest={"calibration_sec": 300.0, "window_sec": 60.0}
        )
        service = SerialSensorService(
            C1ServiceConfig(
                port="TEST",
                process_interval=0.01,
                stream_stall_seconds=0.05,
            ),
            predictor=predictor,
            serial_factory=lambda **_: InvalidSerial(),
        )
        try:
            service.start()
            deadline = time.monotonic() + 1.0
            while service.state != ServiceState.ERROR and time.monotonic() < deadline:
                time.sleep(0.01)
            snapshot = service.snapshot()
            self.assertEqual(snapshot.state, ServiceState.ERROR)
            self.assertIn("no valid DATA", snapshot.last_error or "")
            self.assertGreater(snapshot.invalid_line_count, 0)
        finally:
            service.stop()



class BvpSourceTests(unittest.TestCase):
    """The MAX30102 finger-presence check must not be applied to E4 data.

    E4 BVP is already band-passed and centred on zero, so its median sits near
    1 rather than near 50,000 and ``median_ir < MAX30102_MIN_MEDIAN_IR`` would
    condemn every window of a perfectly good recording.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = C1FeatureMLPPredictor()
        bvp_csv, eda_csv = synthetic_e4_capture()
        parser = SerialDataParser()
        samples = [
            parser.parse(line)
            for line in convert_e4_to_firmware_lines(bvp_csv, eda_csv)
        ]
        cls.e4_samples = [item for item in samples if item is not None]

    def test_e4_window_is_not_flagged_weak(self) -> None:
        _, diagnostics = extract_feature_window(
            self.e4_samples,
            0.0,
            60.0,
            self.predictor.feature_cols,
            bvp_source=BVP_SOURCE_E4,
        )
        self.assertLess(diagnostics.median_ir, 20000)
        self.assertNotIn("weak_bvp_or_finger_missing", diagnostics.quality_flags)

    def test_timestamp_value_csv_layout_preserves_full_overlap(self) -> None:
        bvp_raw, eda_raw = synthetic_tabular_e4_capture()
        lines = convert_e4_to_firmware_lines(bvp_raw, eda_raw)
        self.assertGreater(recording_duration_seconds(lines), 399.0)
        self.assertTrue(lines[0].startswith(b"DATA,0,"))

    def test_the_same_window_is_flagged_when_read_as_max30102(self) -> None:
        _, diagnostics = extract_feature_window(
            self.e4_samples,
            0.0,
            60.0,
            self.predictor.feature_cols,
            bvp_source=BVP_SOURCE_MAX30102,
        )
        self.assertIn("weak_bvp_or_finger_missing", diagnostics.quality_flags)

    def test_e4_calibration_succeeds_and_max30102_still_rejects(self) -> None:
        calibration = self.predictor.calibrate(
            self.e4_samples, start_t=0.0, bvp_source=BVP_SOURCE_E4
        )
        self.assertEqual(calibration.n_windows, 49)
        with self.assertRaises(SignalQualityError):
            self.predictor.calibrate(
                self.e4_samples, start_t=0.0, bvp_source=BVP_SOURCE_MAX30102
            )

    def test_live_sensor_data_keeps_the_finger_presence_check(self) -> None:
        """The default must stay MAX30102, so a real dropout is still caught."""

        dropout = synthetic_sensor_capture(duration_seconds=120.0)
        dropout[:, 1] = 500.0  # finger lifted: raw IR collapses
        _, diagnostics = extract_feature_window(
            dropout, 0.0, 60.0, self.predictor.feature_cols
        )
        self.assertIn("weak_bvp_or_finger_missing", diagnostics.quality_flags)

    def test_service_config_validates_the_source(self) -> None:
        self.assertEqual(
            C1ServiceConfig(port="replay:E4", bvp_source=BVP_SOURCE_E4).bvp_source,
            BVP_SOURCE_E4,
        )
        self.assertEqual(C1ServiceConfig(port="COM3").bvp_source, BVP_SOURCE_MAX30102)
        with self.assertRaises(ValueError):
            C1ServiceConfig(port="COM3", bvp_source="empatica")



if __name__ == "__main__":
    unittest.main()
