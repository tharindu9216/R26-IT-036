import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { SensorStatus } from "../types";
import {
  readUploadedSensorRecording,
  type UploadedSensorRecording,
} from "../sensorRecording";
import { HudRing } from "./HudRing";
import { LiveSensorWaveforms, SensorWaveforms } from "./SensorWaveforms";

interface Props {
  onReady: (sensorAvailable: boolean) => void;
  uploadedRecording: UploadedSensorRecording | null;
  onUploadedRecordingChange: (recording: UploadedSensorRecording | null) => void;
}

const CHECK_IN_QUESTIONS = [
  { key: "manageable", text: "Right now, the demands on me feel manageable." },
  { key: "tense", text: "I feel tense or on edge at this moment." },
  { key: "overwhelmed", text: "I feel overwhelmed by what I need to handle." },
  { key: "relaxed", text: "I feel able to relax right now." },
  { key: "worried", text: "I feel worried about what might happen next." },
  { key: "in_control", text: "I feel in control of my current situation." },
] as const;

const CHECK_IN_OPTIONS = ["Not at all", "A little", "Quite a bit", "Very much"];

function formatCountdown(seconds: number): string {
  const totalSeconds = Math.max(0, Math.ceil(seconds));
  const minutes = Math.floor(totalSeconds / 60);
  return `${minutes}:${String(totalSeconds % 60).padStart(2, "0")}`;
}

function SensorSetupProgress({
  isCalibrating,
  progress,
  secondsLeft,
}: {
  isCalibrating: boolean;
  progress: number;
  secondsLeft: number;
}) {
  return (
    <section className="sensor-setup-progress" aria-label="Overall sensor setup progress">
      <header>
        <strong>Overall sensor setup</strong>
        <span>{formatCountdown(secondsLeft)} remaining</span>
      </header>
      <div
        className="sensor-setup-progress-track"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(progress * 100)}
      >
        <span style={{ width: `${progress * 100}%` }} />
        <i aria-hidden="true" />
      </div>
      <div className="sensor-setup-stages">
        <div className={isCalibrating ? "is-active" : "is-complete"}>
          <b>{isCalibrating ? "1" : "✓"}</b>
          <span><strong>Personal calibration</strong><small>First 5 minutes</small></span>
        </div>
        <div className={isCalibrating ? "" : "is-active"}>
          <b>2</b>
          <span><strong>Prediction warm-up</strong><small>Final 1 minute</small></span>
        </div>
      </div>
    </section>
  );
}

function sensorFileSelectionError(files: File[]): string | null {
  const bvpCount = files.filter((file) => file.name.toLowerCase().includes("bvp")).length;
  const edaCount = files.filter((file) => file.name.toLowerCase().includes("eda")).length;
  if (bvpCount > 0 || edaCount > 0) {
    if (bvpCount !== 1 || edaCount !== 1) {
      return `Select one BVP.csv and one EDA.csv file. Currently selected: ${bvpCount} BVP and ${edaCount} EDA.`;
    }
  } else if (files.length !== 1) {
    return "Select one serial capture, or one BVP.csv and one EDA.csv file.";
  }
  return null;
}

export function SensorGate({
  onReady,
  uploadedRecording,
  onUploadedRecordingChange,
}: Props) {
  const [status, setStatus] = useState<SensorStatus>({ mode: "idle", ready: false });
  const [port, setPort] = useState("/dev/ttyUSB0");
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [questionnaireComplete, setQuestionnaireComplete] = useState(false);
  const [questionnaireBusy, setQuestionnaireBusy] = useState(false);
  const [questionIndex, setQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState<Record<string, number>>({});
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    api.questionnaireStatus().then((result) => {
      if (!cancelled) setQuestionnaireComplete(result.completed);
    }).catch(() => {
      // A missing saved result simply means this session should answer again.
    });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const next = await api.sensorStatus(true);
        if (cancelled) return;
        setConnectionError(null);
        setStatus(next);
        if (next.ready && questionnaireComplete) {
          onReady(next.mode === "sensor");
          return;
        }
      } catch (err) {
        if (!cancelled) {
          setConnectionError(
            err instanceof Error ? err.message : "Backend connection was lost",
          );
        }
      }
      if (!cancelled) {
        timerRef.current = window.setTimeout(poll, 1000);
      }
    }

    poll();
    return () => {
      cancelled = true;
      if (timerRef.current) window.clearTimeout(timerRef.current);
    };
  }, [onReady, questionnaireComplete]);

  async function handleStart() {
    setBusy(true);
    setError(null);
    onUploadedRecordingChange(null);
    try {
      setStatus(await api.startSensor(port));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function handleUpload() {
    if (files.length === 0) return;
    const selectionError = sensorFileSelectionError(files);
    if (selectionError) {
      setError(selectionError);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      try {
        onUploadedRecordingChange(await readUploadedSensorRecording(files));
      } catch {
        // The backend remains the authority for validating sensor recordings.
        // If this optional preview cannot parse a valid future file format,
        // uploading and analysis should still proceed normally.
        onUploadedRecordingChange(null);
      }
      setStatus(await api.uploadSensorRecording(files));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function handleSkip() {
    setBusy(true);
    setError(null);
    onUploadedRecordingChange(null);
    try {
      setStatus(await api.skipSensor());
    } finally {
      setBusy(false);
    }
  }

  async function handleReconnect() {
    setBusy(true);
    setError(null);
    try {
      setStatus(await api.disconnectSensor());
      onUploadedRecordingChange(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function handleQuestionAnswer(value: number) {
    if (questionnaireBusy) return;
    const question = CHECK_IN_QUESTIONS[questionIndex];
    const nextAnswers = { ...answers, [question.key]: value };
    setAnswers(nextAnswers);
    setError(null);

    if (questionIndex < CHECK_IN_QUESTIONS.length - 1) {
      setQuestionIndex((current) => current + 1);
      return;
    }

    setQuestionnaireBusy(true);
    try {
      await api.submitQuestionnaire(nextAnswers);
      setQuestionnaireComplete(true);
      if (status.ready) onReady(status.mode === "sensor");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setQuestionnaireBusy(false);
    }
  }

  function questionnaireCard(
    setup?: {
      isCalibrating: boolean;
      progress: number;
      secondsLeft: number;
    },
  ) {
    const question = CHECK_IN_QUESTIONS[questionIndex];
    const progress = (questionIndex + 1) / CHECK_IN_QUESTIONS.length;
    return (
      <div className="card calibration-question-card">
        <div className="questionnaire-layout">
          <aside className="questionnaire-countdown">
            <div className="questionnaire-ring-wrap">
              <HudRing
                size={174}
                progress={setup?.progress}
                variant={setup === undefined ? "idle" : "progress"}
              >
                <span className="hud-value">
                  {setup === undefined
                    ? CHECK_IN_QUESTIONS.length
                    : formatCountdown(setup.secondsLeft)}
                </span>
                <span className="hud-caption">
                  {setup === undefined ? "questions" : "total remaining"}
                </span>
              </HudRing>
            </div>
            <strong>
              {setup
                ? setup.isCalibrating
                  ? "Step 1 of 2 · Calibration"
                  : "Step 2 of 2 · Warm-up"
                : "Wellbeing check-in"}
            </strong>
            <p>
              {setup
                ? "The complete sensor setup takes 6 minutes while you answer."
                : "A short reflection before we begin."}
            </p>
          </aside>

          <section className="questionnaire-content" key={question.key}>
            <span className="eyebrow">A quick check-in</span>
            <div className="questionnaire-progress" aria-hidden="true">
              <span style={{ width: `${progress * 100}%` }} />
            </div>
            <div className="questionnaire-count">
              Question {questionIndex + 1} of {CHECK_IN_QUESTIONS.length}
            </div>
            <h2>{question.text}</h2>
            <p className="muted">Choose the answer that feels closest right now.</p>
            <div className="questionnaire-options">
              {CHECK_IN_OPTIONS.map((option, value) => (
                <button
                  type="button"
                  key={option}
                  onClick={() => handleQuestionAnswer(value)}
                  disabled={questionnaireBusy}
                >
                  <span>{option}</span>
                  <i aria-hidden="true">→</i>
                </button>
              ))}
            </div>
            <p className="questionnaire-note">
              Your answers are the main signal in the combined wellbeing check.
            </p>
            {error && <p className="error">{error}</p>}
          </section>
        </div>
        {setup && (
          <SensorSetupProgress
            isCalibrating={setup.isCalibrating}
            progress={setup.progress}
            secondsLeft={setup.secondsLeft}
          />
        )}
        {sensorWaveforms()}
      </div>
    );
  }

  function sensorWaveforms() {
    if (status.mode !== "sensor") return null;
    if (uploadedRecording) {
      return (
        <SensorWaveforms
          recording={uploadedRecording}
          elapsedSeconds={status.elapsed_seconds}
        />
      );
    }
    if (status.sensor_waveform?.times.length) {
      return <LiveSensorWaveforms waveform={status.sensor_waveform} />;
    }
    return null;
  }

  if (status.mode === "idle") {
    return (
      <div className="card">
        <span className="eyebrow">Optional wellbeing signal</span>
        <h2>Would you like to connect your wearable?</h2>
        <p className="muted">
          A connected sensor can quietly add stress insights while you chat.
          It takes about <strong>6 minutes</strong> to learn your personal
          baseline. You can also continue without it.
        </p>

        <label className="field">
          <span>Wearable connection</span>
          <input
            value={port}
            onChange={(event) => setPort(event.target.value)}
            disabled={busy}
          />
        </label>
        <div className="actions">
          <button className="primary" onClick={handleStart} disabled={busy}>
            Connect wearable
          </button>
          <button className="ghost" onClick={handleSkip} disabled={busy}>
            Continue without it
          </button>
        </div>

        <div className="divider">For research testing</div>

        <section className="sensor-upload" aria-labelledby="sensor-upload-title">
          <div className="sensor-upload-heading">
            <div>
              <strong id="sensor-upload-title">Upload a recorded session</strong>
              <span>Use saved sensor data instead of a wearable.</span>
            </div>
            <span className="sensor-upload-optional">Optional</span>
          </div>

          <label className={`sensor-file-picker${files.length ? " has-files" : ""}`}>
            <input
              type="file"
              multiple
              accept=".txt,.log,.csv"
              disabled={busy}
              onChange={(event) =>
                {
                  const selected = Array.from(event.target.files ?? []);
                  setFiles(selected);
                  onUploadedRecordingChange(null);
                  setError(sensorFileSelectionError(selected));
                }
              }
            />
            <span className="sensor-file-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24">
                <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5" />
                <path d="M5 14v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4" />
              </svg>
            </span>
            <span className="sensor-file-copy">
              <strong>
                {files.length
                  ? `${files.length} file${files.length === 1 ? "" : "s"} selected`
                  : "Choose sensor files"}
              </strong>
              <small>
                {files.length
                  ? files.map((file) => file.name).join("  •  ")
                  : "Select BVP.csv and EDA.csv together"}
              </small>
            </span>
            <span className="sensor-browse">Browse</span>
          </label>

          <div className="sensor-upload-note">
            <span aria-hidden="true">✓</span>
            At least 6 minutes of matching BVP and EDA data
          </div>
        </section>
        <div className="actions">
          <button
            className="primary"
            onClick={handleUpload}
            disabled={busy || files.length === 0}
          >
            Upload &amp; start
          </button>
        </div>

        {error && <p className="error">{error}</p>}
        {connectionError && (
          <p className="error">Cannot reach the backend: {connectionError}</p>
        )}
      </div>
    );
  }

  if (status.mode === "sensor" && status.state === "error") {
    return (
      <div className="card">
        <h2>Sensor error</h2>
        <p className="error">{status.error}</p>
        <p className="muted sensor-diagnostics">
          Received {status.sample_count.toLocaleString()} valid samples from{" "}
          {status.serial_line_count.toLocaleString()} serial lines
          {status.invalid_line_count > 0
            ? ` · ${status.invalid_line_count.toLocaleString()} lines were ignored`
            : ""}.
        </p>
        <div className="actions">
          <button className="primary" onClick={handleReconnect} disabled={busy}>
            Reconnect sensor
          </button>
          <button className="ghost" onClick={handleSkip} disabled={busy}>
            Continue without it
          </button>
        </div>
        {error && <p className="error">{error}</p>}
      </div>
    );
  }

  if (status.mode === "skipped" && !questionnaireComplete) {
    return questionnaireCard();
  }

  if (status.mode === "sensor") {
    const isCalibrating = status.state === "connecting" || status.state === "calibrating";
    const stageRemaining = isCalibrating
      ? status.calibration_remaining_seconds
      : status.warmup_remaining_seconds;
    const calibrationTotal = status.calibration_total_seconds || 300;
    const warmupTotal = 60;
    const setupTotal = calibrationTotal + warmupTotal;
    const setupRemaining = Math.min(
      setupTotal,
      Math.max(0, isCalibrating ? stageRemaining + warmupTotal : stageRemaining),
    );
    const setupProgress = Math.min(1, Math.max(0, 1 - setupRemaining / setupTotal));
    const label = isCalibrating
      ? "Step 1 of 2 · Personal calibration"
      : "Step 2 of 2 · Prediction warm-up";
    const hasWaveforms = Boolean(
      uploadedRecording || status.sensor_waveform?.times.length,
    );

    if (!questionnaireComplete) {
      return questionnaireCard({
        isCalibrating,
        progress: setupProgress,
        secondsLeft: setupRemaining,
      });
    }

    return (
      <div className={`card calibration-card${isCalibrating ? "" : " warmup-card"}${hasWaveforms ? " has-waveforms" : ""}`}>
        <span className="eyebrow">6-minute sensor setup</span>
        <h2>{isCalibrating ? "Calibrating your wearable" : "Final prediction warm-up"}</h2>
        <div className="calibration-visual">
          {!isCalibrating && (
            <div className="warmup-orbits" aria-hidden="true">
              <i /><i /><i />
            </div>
          )}
          <HudRing size={200} progress={setupProgress} variant="progress">
            <span className="hud-value">{formatCountdown(setupRemaining)}</span>
            <span className="hud-caption">total remaining</span>
          </HudRing>
        </div>
        <p className="muted mono" style={{ textAlign: "center" }}>
          {label}
        </p>
        <SensorSetupProgress
          isCalibrating={isCalibrating}
          progress={setupProgress}
          secondsLeft={setupRemaining}
        />
        <p className="check-in-complete"><span>✓</span> Wellbeing check-in complete</p>
        <p className="sensor-stream-status" aria-live="polite">
          <i /> Receiving sensor data · {status.sample_count.toLocaleString()} samples
          {status.elapsed_seconds > 0
            ? ` · ${Math.floor(status.elapsed_seconds)}s captured`
            : ""}
        </p>
        {sensorWaveforms()}
        {connectionError && (
          <p className="error">
            Live status connection lost: {connectionError}. Check that the backend is running.
          </p>
        )}
      </div>
    );
  }

  return null;
}
