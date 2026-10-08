import { useEffect, useMemo, useRef, useState } from "react";

interface Props {
  audioUrl: string;
  transcript: string;
  role: "user" | "assistant";
  knownDuration?: number;
  autoPlay?: boolean;
}

const BAR_COUNT = 48;

function fallbackBars(): number[] {
  return Array.from({ length: BAR_COUNT }, (_, index) => {
    const primary = Math.abs(Math.sin(index * 1.37 + 0.8));
    const secondary = Math.abs(Math.cos(index * 0.53 + 1.9));
    return 0.2 + (primary * 0.55 + secondary * 0.25);
  });
}

function waveformBars(samples: Float32Array): number[] {
  if (!samples.length) return fallbackBars();
  const blockSize = Math.max(1, Math.floor(samples.length / BAR_COUNT));
  const amplitudes = Array.from({ length: BAR_COUNT }, (_, barIndex) => {
    const start = barIndex * blockSize;
    const end = Math.min(samples.length, start + blockSize);
    let energy = 0;
    for (let index = start; index < end; index += 1) {
      energy += samples[index] * samples[index];
    }
    return Math.sqrt(energy / Math.max(1, end - start));
  });
  const maximum = Math.max(...amplitudes, 0.0001);
  return amplitudes.map((amplitude) => 0.16 + (amplitude / maximum) * 0.84);
}

function formatTime(value: number): string {
  if (!Number.isFinite(value) || value < 0) return "0:00";
  const seconds = Math.floor(value);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

export function VoiceWaveMessage({
  audioUrl,
  transcript,
  role,
  knownDuration,
  autoPlay = false,
}: Props) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(knownDuration ?? 0);
  const [bars, setBars] = useState<number[]>(fallbackBars);
  const progress = duration > 0 ? Math.min(1, currentTime / duration) : 0;
  const activeBars = Math.round(progress * BAR_COUNT);
  const label = role === "user" ? "Your voice" : "Supportive reply";

  useEffect(() => {
    let cancelled = false;
    async function decodeWaveform() {
      try {
        const response = await fetch(audioUrl);
        if (!response.ok) return;
        const encoded = await response.arrayBuffer();
        const AudioContextClass = window.AudioContext;
        if (!AudioContextClass) return;
        const context = new AudioContextClass();
        try {
          const decoded = await context.decodeAudioData(encoded.slice(0));
          if (!cancelled) setBars(waveformBars(decoded.getChannelData(0)));
        } finally {
          void context.close();
        }
      } catch {
        // A decorative fallback waveform remains if decoding is unavailable.
      }
    }
    void decodeWaveform();
    return () => { cancelled = true; };
  }, [audioUrl]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const updateTime = () => setCurrentTime(audio.currentTime);
    const updateDuration = () => {
      if (Number.isFinite(audio.duration)) setDuration(audio.duration);
    };
    const markPlaying = () => setPlaying(true);
    const markPaused = () => setPlaying(false);
    audio.addEventListener("timeupdate", updateTime);
    audio.addEventListener("loadedmetadata", updateDuration);
    audio.addEventListener("durationchange", updateDuration);
    audio.addEventListener("play", markPlaying);
    audio.addEventListener("pause", markPaused);
    audio.addEventListener("ended", markPaused);
    return () => {
      audio.removeEventListener("timeupdate", updateTime);
      audio.removeEventListener("loadedmetadata", updateDuration);
      audio.removeEventListener("durationchange", updateDuration);
      audio.removeEventListener("play", markPlaying);
      audio.removeEventListener("pause", markPaused);
      audio.removeEventListener("ended", markPaused);
    };
  }, []);

  const timeLabel = useMemo(
    () => playing || currentTime > 0
      ? `${formatTime(currentTime)} / ${formatTime(duration)}`
      : formatTime(duration),
    [currentTime, duration, playing],
  );

  function togglePlayback() {
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) void audio.play();
    else audio.pause();
  }

  function seek(event: React.MouseEvent<HTMLButtonElement>) {
    const audio = audioRef.current;
    if (!audio || !duration) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width));
    audio.currentTime = ratio * duration;
    setCurrentTime(audio.currentTime);
  }

  return (
    <div className={`voice-wave-card is-${role}`}>
      <audio ref={audioRef} src={audioUrl} preload="metadata" autoPlay={autoPlay} />
      <div className="voice-wave-heading">
        <span><i aria-hidden="true" />{label}</span>
        <time>{timeLabel}</time>
      </div>
      <div className="voice-wave-player">
        <button
          type="button"
          className="voice-wave-play"
          onClick={togglePlayback}
          aria-label={playing ? `Pause ${label}` : `Play ${label}`}
        >
          {playing ? <span className="pause-icon" aria-hidden="true" /> : <span className="play-icon" aria-hidden="true" />}
        </button>
        <button
          type="button"
          className="voice-wave-bars"
          onClick={seek}
          aria-label={`Seek within ${label}`}
        >
          {bars.map((height, index) => (
            <i
              key={index}
              className={index < activeBars ? "is-played" : ""}
              style={{ height: `${Math.round(height * 100)}%` }}
            />
          ))}
        </button>
      </div>
      <div className="voice-mini-transcript">
        <span>{role === "user" ? "Transcript" : "Reply transcript"}</span>
        <p>{transcript}</p>
      </div>
    </div>
  );
}
