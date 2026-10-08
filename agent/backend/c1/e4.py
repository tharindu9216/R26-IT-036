"""Convert an Empatica E4 (WESAD-style) BVP.csv + EDA.csv pair into the
``DATA,<millis>,<ir>,<eda>`` firmware-line format the replay pipeline
(``replay.py``) expects, so a real E4 session can be uploaded and replayed
through the exact same calibration/prediction timeline as the ESP32 wearable.

Two input layouts are accepted. Native E4 exports use row 1 = session start
Unix time, row 2 = sample rate in Hz, then one value per row. Prepared WESAD
exports may instead use explicit two-column rows such as
``time_seconds,BVP`` / ``time_seconds,EDA``. BVP is normally 64 Hz and EDA
4 Hz (matching ``bvp_fs``/``eda_fs`` in ``model/c1/run_manifest.json``).
EDA is linearly upsampled onto the overlapping portion of BVP's timestamp
grid: EDA changes slowly, so this is a standard, safe resampling, and it lets
every emitted line carry a real value for both channels, as if they had been
sampled together.

Note: the "weak signal" diagnostic in ``inference.py``
(``median_ir < MAX30102_MIN_MEDIAN_IR``) is a finger-presence test for the
ESP32's raw MAX30102 ADC scale, where a low DC level means the finger has
left the sensor. E4's BVP is already band-passed and centred on zero, so its
median sits near 1 rather than near 50,000 and the same test would condemn
every window. Callers replaying E4 data must therefore pass
``bvp_source=BVP_SOURCE_E4`` (``C1ServiceConfig.bvp_source``), which skips
that one check; the remaining quality flags (``insufficient_valid_ibi``,
``flat_eda_signal``) are scale-free and still apply.
"""

from __future__ import annotations


def _read_e4_channel(raw: bytes, channel_name: str) -> tuple[list[float], list[float]]:
    text = raw.decode("utf-8", errors="ignore").strip()
    rows = [line.strip() for line in text.splitlines() if line.strip()]
    if len(rows) < 2:
        raise ValueError(
            f"{channel_name} CSV does not contain enough rows"
        )

    # Prepared WESAD/tidy CSV: time_seconds,BVP (or EDA), then timestamp,value.
    header = [field.strip().lower() for field in rows[0].split(",")]
    if len(header) >= 2 and header[0] in {
        "time",
        "timestamp",
        "time_seconds",
        "seconds",
    }:
        times: list[float] = []
        values: list[float] = []
        try:
            for row in rows[1:]:
                fields = [field.strip() for field in row.split(",")]
                if len(fields) < 2:
                    continue
                times.append(float(fields[0]))
                values.append(float(fields[1]))
        except ValueError as exc:
            raise ValueError(
                f"Could not parse timestamp/value rows in {channel_name} CSV"
            ) from exc
        if len(times) < 5:
            raise ValueError(f"{channel_name} CSV has too few timestamped values")
        if any(later <= earlier for earlier, later in zip(times, times[1:])):
            raise ValueError(f"{channel_name} timestamps must increase continuously")
        return times, values

    # Native Empatica E4 CSV: absolute start time, sample rate, then values.
    if len(rows) < 3:
        raise ValueError(
            "Native E4 CSV must have a start-time row, a sample-rate row, "
            "and value rows"
        )
    try:
        start_time = float(rows[0].split(",")[0])
        rate = float(rows[1].split(",")[0])
        values = [float(row.split(",")[0]) for row in rows[2:]]
    except ValueError as exc:
        raise ValueError(
            f"Could not parse native E4 {channel_name} header/rate/values"
        ) from exc
    if rate <= 0:
        raise ValueError(f"E4 {channel_name} sample rate must be positive")
    if not values:
        raise ValueError(f"E4 {channel_name} CSV has no sample rows")
    times = [start_time + index / rate for index in range(len(values))]
    return times, values


def _interp_onto(x_new: list[float], x: list[float], y: list[float]) -> list[float]:
    """Dependency-free linear interpolation (this module has no numpy dependency)."""

    result: list[float] = []
    j = 0
    n = len(x)
    for xi in x_new:
        if xi <= x[0]:
            result.append(y[0])
            continue
        if xi >= x[-1]:
            result.append(y[-1])
            continue
        while j + 1 < n and x[j + 1] < xi:
            j += 1
        x0, x1 = x[j], x[j + 1]
        y0, y1 = y[j], y[j + 1]
        t = (xi - x0) / (x1 - x0) if x1 > x0 else 0.0
        result.append(y0 + t * (y1 - y0))
    return result


def convert_e4_to_firmware_lines(bvp_raw: bytes, eda_raw: bytes) -> list[bytes]:
    bvp_times, bvp_values = _read_e4_channel(bvp_raw, "BVP")
    eda_times, eda_values = _read_e4_channel(eda_raw, "EDA")

    # Keep only the interval actually covered by both channels. Explicit-time
    # exports do not necessarily begin on the exact same sample boundary.
    start_t = max(bvp_times[0], eda_times[0])
    end_t = min(bvp_times[-1], eda_times[-1])
    if end_t <= start_t:
        raise ValueError("BVP and EDA recordings do not overlap in time")
    trimmed = [
        (t, v)
        for t, v in zip(bvp_times, bvp_values)
        if start_t <= t <= end_t
    ]
    if len(trimmed) < 5:
        raise ValueError("Not enough overlapping BVP/EDA duration to replay")

    trimmed_times = [t for t, _ in trimmed]
    trimmed_bvp = [v for _, v in trimmed]
    eda_on_bvp_grid = _interp_onto(trimmed_times, eda_times, eda_values)

    return [
        f"DATA,{round((t - start_t) * 1000.0)},{bvp_value:.4f},{eda_value:.4f}".encode()
        for t, bvp_value, eda_value in zip(trimmed_times, trimmed_bvp, eda_on_bvp_grid)
    ]
