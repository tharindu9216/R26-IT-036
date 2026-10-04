"""Generate a synthetic C1 firmware capture for testing without a wearable.

Produces the same signal shape used by the C1 checkpoint test
(``tests/test_c1.py``'s ``synthetic_sensor_capture``), formatted as real
``DATA,<millis>,<ir>,<eda>`` lines so it can be uploaded through
``POST /api/sensor/upload`` (or the Streamlit sensor gate's "upload a
recording" option) and replayed end-to-end without ever wearing the sensor.
No third-party dependencies: only the stdlib, so it also works before
``requirements.txt`` is installed.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def synthetic_capture(duration_seconds: float = 400.0, dt: float = 0.1) -> list[str]:
    lines: list[str] = []
    steps = round(duration_seconds / dt)
    for step in range(steps + 1):
        t = step * dt
        # Strong raw IR with a stable ~72 BPM pulse plus a small harmonic.
        bvp = (
            50000.0
            + 3500.0 * math.sin(2.0 * math.pi * 1.2 * t)
            + 350.0 * math.sin(2.0 * math.pi * 2.4 * t)
        )
        # Slowly varying ADC-scale EDA with regular phasic responses.
        eda = (
            1900.0
            + 20.0 * math.sin(2.0 * math.pi * 0.01 * t)
            + 5.0 * max(0.0, math.sin(2.0 * math.pi * 0.12 * t))
        )
        millis = round(t * 1000.0)
        lines.append(f"DATA,{millis},{bvp:.1f},{eda:.1f}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Output .txt path")
    parser.add_argument(
        "--duration",
        type=float,
        default=400.0,
        help="Seconds of signal to generate (>=360 covers calibration + warm-up)",
    )
    args = parser.parse_args()

    lines = synthetic_capture(args.duration)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} lines ({args.duration:.0f}s) to {args.output}")


if __name__ == "__main__":
    main()
