import { useId } from "react";
import type { SensorChannel, UploadedSensorRecording } from "../sensorRecording";
import type { SensorWaveformData } from "../types";

function lowerBound(values: number[], target: number): number {
  let low = 0;
  let high = values.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (values[middle] < target) low = middle + 1;
    else high = middle;
  }
  return low;
}

function Waveform({
  channel,
  currentTime,
  label,
  className,
}: {
  channel: SensorChannel;
  currentTime: number;
  label: string;
  className: string;
}) {
  const gradientId = useId().replace(/:/g, "");
  const width = 420;
  const height = 76;
  const windowSeconds = 8;
  const windowStart = Math.max(0, currentTime - windowSeconds);
  const startIndex = Math.max(0, lowerBound(channel.times, windowStart) - 1);
  const firstAtOrAfter = lowerBound(channel.times, currentTime);
  const endIndex = Math.min(
    channel.times.length,
    firstAtOrAfter + (channel.times[firstAtOrAfter] === currentTime ? 1 : 0),
  );
  let samples = channel.values.slice(startIndex, endIndex).map((value, offset) => ({
    time: channel.times[startIndex + offset],
    value,
  }));

  if (samples.length > width) {
    const step = samples.length / width;
    samples = Array.from({ length: width }, (_, index) => samples[Math.floor(index * step)]);
  }
  const sampleValues = samples.map(({ value }) => value);
  const minimum = sampleValues.length ? Math.min(...sampleValues) : 0;
  const maximum = sampleValues.length ? Math.max(...sampleValues) : 1;
  const range = maximum - minimum || Math.max(Math.abs(maximum) * 0.05, 1);
  const xRange = Math.max(currentTime - windowStart, 0.25);
  const points = samples.map(({ time, value }) => {
    const x = ((time - windowStart) / xRange) * width;
    const y = height - 7 - ((value - minimum) / range) * (height - 14);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  const currentValue = sampleValues.at(-1);
  const areaPoints = points ? `0,${height} ${points} ${width},${height}` : "";

  return (
    <section className={`sensor-waveform ${className}`} aria-label={`${label} waveform`}>
      <header>
        <span><i aria-hidden="true" />{label}</span>
        <strong>{currentValue === undefined ? "Waiting…" : `${currentValue.toFixed(2)} ${channel.unit}`}</strong>
      </header>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${label}, latest eight seconds`}>
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="currentColor" stopOpacity=".22" />
            <stop offset="1" stopColor="currentColor" stopOpacity="0" />
          </linearGradient>
        </defs>
        <path className="waveform-grid" d={`M0 ${height / 2}H${width} M${width / 4} 0V${height} M${width / 2} 0V${height} M${width * 0.75} 0V${height}`} />
        {areaPoints && <polygon points={areaPoints} fill={`url(#${gradientId})`} />}
        {points && <polyline className="waveform-line" points={points} />}
      </svg>
    </section>
  );
}

export function SensorWaveforms({
  recording,
  elapsedSeconds,
}: {
  recording: UploadedSensorRecording;
  elapsedSeconds: number;
}) {
  const currentTime = Math.min(Math.max(0, elapsedSeconds), recording.durationSeconds);
  const progress = recording.durationSeconds > 0 ? currentTime / recording.durationSeconds : 0;
  return (
    <div className="sensor-waveforms" aria-label="Uploaded sensor signals">
      <div className="sensor-waveforms-heading">
        <div>
          <strong>Uploaded sensor signals</strong>
          <span>Live view of the latest 8 seconds</span>
        </div>
        <span>{Math.floor(currentTime)}s / {Math.ceil(recording.durationSeconds)}s</span>
      </div>
      <div className="sensor-waveforms-progress" aria-hidden="true">
        <span style={{ width: `${progress * 100}%` }} />
      </div>
      <div className="sensor-waveforms-grid">
        <Waveform channel={recording.bvp} currentTime={currentTime} label="BVP · pulse" className="waveform-bvp" />
        <Waveform channel={recording.eda} currentTime={currentTime} label="EDA · skin response" className="waveform-eda" />
      </div>
    </div>
  );
}

export function LiveSensorWaveforms({
  waveform,
}: {
  waveform: SensorWaveformData;
}) {
  const currentTime = waveform.times.at(-1) ?? 0;
  const bvp: SensorChannel = {
    times: waveform.times,
    values: waveform.bvp,
    unit: waveform.bvp_unit,
  };
  const eda: SensorChannel = {
    times: waveform.times,
    values: waveform.eda,
    unit: waveform.eda_unit,
  };

  return (
    <div className="sensor-waveforms" aria-label="Live wearable sensor signals">
      <div className="sensor-waveforms-heading">
        <div>
          <strong>Live wearable signals</strong>
          <span>Updating from the latest 8 seconds</span>
        </div>
        <span>{waveform.times.length} display points</span>
      </div>
      <div className="sensor-waveforms-live" aria-hidden="true"><span /></div>
      <div className="sensor-waveforms-grid">
        <Waveform channel={bvp} currentTime={currentTime} label="BVP · pulse" className="waveform-bvp" />
        <Waveform channel={eda} currentTime={currentTime} label="EDA · skin response" className="waveform-eda" />
      </div>
    </div>
  );
}
