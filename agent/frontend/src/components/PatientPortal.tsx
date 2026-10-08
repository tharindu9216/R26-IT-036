import { useEffect, useState } from "react";
import { api } from "../api";
import type { Account } from "../types";
import { ModeSelect } from "./ModeSelect";
import { Chat } from "./Chat";
import { VoiceChat } from "./VoiceChat";

const questions = [
  ["manageable", "Right now, the demands on me feel manageable."],
  ["tense", "I feel tense or on edge at this moment."],
  ["overwhelmed", "I feel overwhelmed by what I need to handle."],
  ["relaxed", "I feel able to relax right now."],
  ["worried", "I feel worried about what might happen next."],
  ["in_control", "I feel in control of my current situation."],
];
const options = ["Not at all", "A little", "Quite a bit", "Very much"];

function CheckIn({ onDone }: { onDone: () => void }) {
  const [answers, setAnswers] = useState<Record<string, number>>({});
  const [index, setIndex] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function answer(value: number) {
    const next = { ...answers, [questions[index][0]]: value };
    setAnswers(next);
    if (index < questions.length - 1) { setIndex(index + 1); return; }
    setBusy(true); setError("");
    try { await api.submitQuestionnaire(next); onDone(); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "Could not save your check-in."); }
    finally { setBusy(false); }
  }
  return <section className="card portal-panel check-in"><span className="eyebrow">A MOMENT TO CHECK IN · {index + 1} OF {questions.length}</span><h2>{questions[index][1]}</h2><p>Choose what feels closest to how you feel right now.</p><div className="check-in-options">{options.map((label, value) => <button key={label} disabled={busy} onClick={() => answer(value)}>{label}</button>)}</div>{error && <p role="alert" className="error">{error}</p>}<button className="portal-link" disabled={busy} onClick={onDone}>Skip for now</button></section>;
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
  const [doctorEmail, setDoctorEmail] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  async function share(remove = false) {
    setBusy(true); setNotice("");
    try {
      if (remove) { await api.unlinkDoctor(); setNotice("Your doctor can no longer access your activity."); }
      else { const doctor = await api.linkDoctor(doctorEmail); setNotice(`Your activity is now shared with ${doctor.name}.`); }
      onUpdate(await api.me());
    } catch (caught) { setNotice(caught instanceof Error ? caught.message : "Could not update sharing."); }
    finally { setBusy(false); }
  }
  return <div className="patient-portal">
    {screen !== "home" && <button className="portal-back" onClick={() => setScreen("home")}>← Back to your space</button>}
    {screen === "home" && <><ModeSelect onSelect={selected => { setMode(selected); setScreen("check-in"); }} /><details className="card sharing-panel"><summary>Your care team {user.doctor_id ? "· Sharing enabled" : "· Not sharing"}</summary><p>Choose whether a doctor can review your conversations, check-ins, wearable readings, and AI insights. Linking a doctor shares your retained activity. You can stop sharing at any time.</p><form className="portal-form" onSubmit={event => { event.preventDefault(); void share(); }}><label>Your doctor’s email<input type="email" required value={doctorEmail} onChange={event => setDoctorEmail(event.target.value)} placeholder="doctor@example.com" /></label><button disabled={busy}>{user.doctor_id ? "Change doctor and share" : "Share with my doctor"}</button></form>{user.doctor_id && <button className="portal-link" disabled={busy} onClick={() => share(true)}>Stop sharing</button>}{notice && <p role="status">{notice}</p>}</details></>}
    {screen === "check-in" && <CheckIn onDone={() => setScreen("prepare")} />}
    {screen === "prepare" && <Prepare mode={mode} onReady={() => setScreen("chat")} />}
    {screen === "chat" && (mode === "text" ? <Chat /> : <VoiceChat />)}
  </div>;
}
