import { useEffect, useState, type ReactNode } from "react";
import { api } from "../api";
import type { Account, ChatReadiness } from "../types";
import { ModeSelect } from "./ModeSelect";
import { Chat } from "./Chat";
import { VoiceChat } from "./VoiceChat";
import { DASSCheckIn } from "./DASSCheckIn";
import { CareTeam } from "./CareTeam";

function SensorReadyGate({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<ChatReadiness | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    let timer: number | undefined;
    async function poll() {
      try {
        const next = await api.chatReadiness();
        if (!active) return;
        setStatus(next);
        setError("");
      } catch {
        if (!active) return;
        setStatus(null);
        setError("We could not check your wearable. Checking again automatically…");
      }
      if (active) timer = window.setTimeout(poll, 1000);
    }
    void poll();
    return () => { active = false; window.clearTimeout(timer); };
  }, []);

  if (status?.ready) return children;
  const seconds = status?.state === "calibrating"
    ? status.calibration_remaining_seconds : status?.warmup_remaining_seconds ?? 0;
  const remaining = Math.max(0, Math.ceil(seconds));
  return <section className="card portal-panel">
    <span className="eyebrow">GETTING YOUR WEARABLE READY</span>
    <h2>{status?.state === "waiting_for_assessment" ? "Please complete your assessment" : status?.state === "calibrating" ? "Personal calibration in progress" : status?.state === "warming_up" ? "Waiting for your first reading" : "Waiting for your wearable"}</h2>
    <p role="status">{status?.message ?? "Checking your wearable…"}</p>
    {remaining > 0 && <p role="timer">Time remaining: {Math.floor(remaining / 60)}:{String(remaining % 60).padStart(2, "0")}</p>}
    <p>Your conversation will open after calibration and the first sensor reading are ready. Please keep wearing the sensor.</p>
    {error && <p className="error" role="alert">{error}</p>}
  </section>;
}

function Prepare({ mode, onReady }: { mode: "text" | "voice"; onReady: () => void }) {
  const [attempt, setAttempt] = useState(0);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    let timer: number | undefined;
    async function prepare() {
      try {
        const status = await (mode === "text" ? api.prepareTextModels() : api.prepareVoiceModels());
        if (!active) return;
        if (status.ready) onReady();
        else if (Object.values(status.models).some(model => model.status === "error")) setError("Your conversation could not be prepared. Please try again.");
        else timer = window.setTimeout(prepare, 2000);
      } catch { if (active) setError("We could not open your conversation. Please try again."); }
    }
    void prepare();
    return () => { active = false; window.clearTimeout(timer); };
  }, [mode, attempt, onReady]);
  return <section className="card portal-panel"><span className="eyebrow">SETTLE IN</span><h2>Getting your conversation ready…</h2><p role="status">This may take a little while the first time. Take a breath; there’s no rush.</p>{error && <><p className="error" role="alert">{error}</p><button onClick={() => { setError(""); setAttempt(attempt + 1); }}>Try again</button></>}</section>;
}

export function PatientPortal({ user, onUpdate }: { user: Account; onUpdate: (user: Account) => void }) {
  const [screen, setScreen] = useState<"home" | "check-in" | "prepare" | "chat">("home");
  const [mode, setMode] = useState<"text" | "voice">("text");
  return <div className="patient-portal">
    {screen !== "home" && <button className="portal-back" onClick={() => setScreen("home")}>← Back to your space</button>}
    {screen === "home" && <><ModeSelect onSelect={selected => { setMode(selected); setScreen("check-in"); }} /><CareTeam user={user} onUpdate={onUpdate} /></>}
    {screen === "check-in" && <DASSCheckIn patientId={user.id} onDone={() => setScreen("prepare")} />}
    {(screen === "prepare" || screen === "chat") && <SensorReadyGate>
      {screen === "prepare" ? <Prepare mode={mode} onReady={() => setScreen("chat")} /> : mode === "text" ? <Chat /> : <VoiceChat />}
    </SensorReadyGate>}
  </div>;
}
