"""Replay a recorded firmware capture as a stand-in for a live serial port.

Lets the full C1 timeline (5 min calibration -> 1 min warm-up -> a prediction
every 5 s) run against an uploaded recording instead of a wearable that has to
be worn for every test. The recording is a plain text capture of the
firmware's own ``DATA,<millis>,<MAX30102_IR>,<GSR_ADC>`` lines (i.e. exactly
what a real serial session would have produced) and is replayed at the pace
implied by each line's own ``millis`` field, so the calibration/warm-up
timing on screen behaves identically to a live sensor no matter what sample
rate the recording was captured at.
"""

from __future__ import annotations

import time


UINT32_MODULUS = 2**32


def _extract_millis(line: bytes) -> int | None:
    try:
        _, millis_field, _ir, _eda = line.split(b",")
        return int(millis_field)
    except (ValueError, TypeError):
        return None


class ReplaySerialConnection:
    """Matches the slice of the pyserial ``Serial`` API ``SerialSensorService``
    uses (``readline`` / ``reset_input_buffer`` / ``close``), so it can be
    handed to ``SerialSensorService`` via its ``serial_factory`` hook without
    changing any live-sensor code.

    ``speed`` is a multiplier on the recording's own pace (2.0 replays twice
    as fast); it defaults to 1.0 so calibration timing matches a live sensor
    exactly.
    """

    def __init__(self, lines: list[bytes], speed: float = 1.0) -> None:
        self._lines = lines
        self._index = 0
        self._speed = max(1e-6, float(speed))
        self._last_millis: int | None = None
        self._next_due = time.monotonic()

    def reset_input_buffer(self) -> None:
        pass

    def readline(self) -> bytes:
        if self._index >= len(self._lines):
            # Recording exhausted: behave like a serial read timeout so the
            # reader loop keeps polling instead of erroring out.
            time.sleep(0.05)
            return b""

        line = self._lines[self._index]
        self._index += 1

        millis = _extract_millis(line)
        if millis is not None:
            if self._last_millis is not None:
                delta_seconds = max(0.0, (millis - self._last_millis) / 1000.0)
                self._next_due += delta_seconds / self._speed
            self._last_millis = millis

        now = time.monotonic()
        if now < self._next_due:
            time.sleep(self._next_due - now)
        return line

    def close(self) -> None:
        pass


def parse_recording(raw: bytes) -> list[bytes]:
    """Split an uploaded capture into ``DATA,...`` firmware lines."""

    lines = [
        line.strip()
        for line in raw.replace(b"\r\n", b"\n").split(b"\n")
        if line.strip().startswith(b"DATA,")
    ]
    if not lines:
        raise ValueError(
            "No DATA,<millis>,<ir>,<eda> lines found in the uploaded recording"
        )
    return lines


def recording_duration_seconds(lines: list[bytes]) -> float:
    """Return the usable timestamp span of firmware-format recording lines."""

    first: int | None = None
    previous: int | None = None
    last_unwrapped: int | None = None
    rollover_offset = 0
    for line in lines:
        raw_millis = _extract_millis(line)
        if raw_millis is None or raw_millis < 0 or raw_millis >= UINT32_MODULUS:
            continue
        if previous is not None and raw_millis < previous:
            backwards = previous - raw_millis
            if backwards > UINT32_MODULUS // 2:
                rollover_offset += UINT32_MODULUS
            else:
                raise ValueError("Recording timestamps move backwards")
        unwrapped = raw_millis + rollover_offset
        if first is None:
            first = unwrapped
        previous = raw_millis
        last_unwrapped = unwrapped
    if first is None or last_unwrapped is None:
        raise ValueError("Recording contains no valid DATA timestamps")
    return max(0.0, (last_unwrapped - first) / 1000.0)
