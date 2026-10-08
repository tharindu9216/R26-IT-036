import { useEffect, useState } from "react";
import { api, selectPatient } from "../api";
import type { ChatMessage, PatientSummary, SensorStatus, XAIResponse } from "../types";
import { XAIView } from "./XAIView";
import { LiveSensorWaveforms } from "./SensorWaveforms";

function SensorPanel() {
  const [status, setStatus] = useState<SensorStatus | null>(null);
  const [port, setPort] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    let timer: number | undefined;
    async function poll() {
      try { const next = await api.sensorStatus(true); if (active) setStatus(next); }
      catch (caught) { if (active) setError(caught instanceof Error ? caught.message : "Could not read sensor status."); }
      if (active) timer = window.setTimeout(poll, 3000);
    }
    void poll();
    return () => { active = false; window.clearTimeout(timer); };
  }, []);
  async function run(action: () => Promise<SensorStatus>) {
    setBusy(true); setError("");
    try { setStatus(await action()); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "Sensor request failed."); }
    finally { setBusy(false); }
  }
  return <section className="card portal-panel sensor-review"><span className="eyebrow">WEARABLE · SELECTED PATIENT</span><h2>Sensor signals</h2><p>Connect a wearable or replay a recording belonging to this patient. One recording can run at a time.</p>
    <form className="sensor-actions" onSubmit={event => { event.preventDefault(); void run(() => api.startSensor(port)); }}><label>Serial port<input required value={port} onChange={event => setPort(event.target.value)} placeholder="/dev/ttyUSB0 or COM3" /></label><button disabled={busy}>Connect wearable</button></form>
    <label className="portal-upload">Replay sensor recording<input type="file" multiple accept=".csv,.txt" disabled={busy} onChange={event => { const files = Array.from(event.target.files ?? []); if (files.length) void run(() => api.uploadSensorRecording(files)); event.target.value = ""; }} /></label>
    <small>Choose a firmware capture, or both BVP.csv and EDA.csv. At least six minutes are required.</small>
    {status && <p className="sensor-status" role="status">{status.mode === "sensor" ? `Wearable: ${status.state.replaceAll("_", " ")}` : "No active wearable for this patient."}</p>}
    {status?.mode === "sensor" && <>
      {status.error && <p className="error">{status.error}</p>}
      {status.stream_stalled && <p className="error">The stream has stopped updating. Check the device connection.</p>}
      {status.state === "calibrating" && <p>Personal calibration: {Math.ceil(status.calibration_remaining_seconds)} seconds remaining.</p>}
      {status.state === "warming_up" && <p>First prediction: {Math.ceil(status.warmup_remaining_seconds)} seconds remaining.</p>}
      {status.sensor_stress_probability !== undefined && <p><strong>Wearable stress estimate: {(status.sensor_stress_probability * 100).toFixed(1)}%</strong></p>}
      {status.sensor_waveform && <LiveSensorWaveforms waveform={status.sensor_waveform} />}
      <div className="sensor-actions"><button disabled={busy} onClick={() => run(api.restartSensor)}>Recalibrate</button><button disabled={busy} onClick={() => run(api.disconnectSensor)}>Disconnect</button></div>
    </>}
    {error && <p className="error" role="alert">{error}</p>}
  </section>;
}

function ActivityTurn({ message, input }: { message: ChatMessage; input: string }) {
  const [explanation, setExplanation] = useState<XAIResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const trace = message.trace;
  async function explain() {
    if (!trace?.turn_id) return;
    setBusy(true); setError("");
    try { setExplanation(await api.explainChat(input, trace.turn_id)); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "Explanation unavailable."); }
    finally { setBusy(false); }
  }
  return <article className="activity-turn"><header><span className="eyebrow">{trace?.mode === "voice" ? "VOICE CHECK-IN" : "WRITTEN CHECK-IN"}</span>{message.created_at && <time dateTime={new Date(message.created_at * 1000).toISOString()}>{new Date(message.created_at * 1000).toLocaleString()}</time>}</header><div className="activity-message"><strong>Patient</strong><p>{input}</p></div><div className="activity-message reply"><strong>Supportive response</strong><p>{message.text}</p></div>
    {trace ? <details className="decision-record"><summary>How this response was generated · {trace.steps.length} stages</summary><ol>{trace.steps.map(step => <li key={step.key}><h4>{step.title}</h4><small>{step.component} · {step.status}</small><p>{step.summary ?? step.detail}</p>{step.outputs?.length ? <dl>{step.outputs.map((output, index) => <div key={index}><dt>{output.label}</dt><dd>{output.value}</dd></div>)}</dl> : null}{step.reasons?.length ? <ul>{step.reasons.map(reason => <li key={reason.code}>{reason.text}</li>)}</ul> : null}</li>)}</ol></details> : <p className="portal-note">No decision record was saved for this reply.</p>}
    {message.explanation_available ? <button disabled={busy} onClick={() => explanation ? setExplanation(null) : void explain()}>{busy ? "Computing explanation…" : explanation ? "Hide AI explanation" : "Explain predictions (XAI)"}</button> : <p className="portal-note">Live explanations have expired or are unavailable. Saved decision records remain above.</p>}
    {error && <p className="error" role="alert">{error}</p>}{explanation && <XAIView result={explanation} />}
  </article>;
}

function PatientReview({ patient }: { patient: PatientSummary }) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [tab, setTab] = useState<"activity" | "sensor">("activity");
  useEffect(() => {
    let active = true;
    let timer: number | undefined;
    async function refresh() {
      try { const result = await api.activity(); if (active) { setMessages(result); setError(""); } }
      catch (caught) { if (active) { setMessages([]); setError(caught instanceof Error ? caught.message : "Could not load activity."); } }
      finally { if (active) { setLoading(false); timer = window.setTimeout(refresh, 10000); } }
    }
    void refresh();
    return () => { active = false; window.clearTimeout(timer); };
  }, []);
  return <section className="patient-review"><header className="review-heading"><div><span className="eyebrow">PATIENT OVERVIEW</span><h2>{patient.name}</h2><p>{patient.email}</p></div><span className="sharing-badge">Access shared by patient</span></header><nav className="review-tabs" aria-label="Patient information"><button aria-pressed={tab === "activity"} onClick={() => setTab("activity")}>Activity & explanations</button><button aria-pressed={tab === "sensor"} onClick={() => setTab("sensor")}>Sensor signals</button></nav>
    {error ? <p className="error" role="alert">{error}</p> : tab === "sensor" ? <SensorPanel /> : <section className="card portal-panel"><div className="activity-heading"><h3>Recent conversations</h3><small>Updates every 10 seconds</small></div>{loading && <p role="status">Loading activity…</p>}{!loading && messages.length === 0 && <div className="portal-empty"><h3>No activity yet</h3><p>This patient’s conversations will appear here after their first check-in.</p></div>}{messages.map((message, index) => message.role === "assistant" ? <ActivityTurn key={message.trace?.turn_id ?? `${message.created_at}-${index}`} message={message} input={messages[index - 1]?.role === "user" ? messages[index - 1].text : "No saved input"} /> : null)}</section>}
  </section>;
}

export function DoctorPortal() {
  const [patients, setPatients] = useState<PatientSummary[]>([]);
  const [selected, setSelected] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  useEffect(() => {
    let active = true;
    let timer: number | undefined;
    async function refresh() {
      try { const result = await api.patients(); if (active) { setPatients(result); setError(""); } }
      catch (caught) { if (active) { setPatients([]); setError(caught instanceof Error ? caught.message : "Could not load patients."); } }
      finally { if (active) { setLoading(false); timer = window.setTimeout(refresh, 10000); } }
    }
    void refresh();
    return () => { active = false; window.clearTimeout(timer); selectPatient(""); };
  }, []);
  const patient = patients.find(item => item.id === selected);
  return <div className="doctor-portal"><header className="doctor-heading"><div><span className="eyebrow">YOUR CARE WORKSPACE</span><h1>A clearer view of their wellbeing.</h1><p>Review patient activity and the evidence behind each supportive response.</p></div><span className="patient-count">{patients.length}<small>linked patients</small></span></header>
    {error && <p className="error" role="alert">{error}</p>}
    <div className="doctor-layout"><aside className="card patient-list"><h2>Your patients</h2><label className="patient-search">Find a patient<input type="search" value={search} onChange={event => setSearch(event.target.value)} placeholder="Search name or email" /></label>{loading && <p role="status">Loading patients…</p>}{!loading && patients.length === 0 && <p>Patients appear when they share access using your email in their care team settings.</p>}{patients.filter(item => `${item.name} ${item.email}`.toLowerCase().includes(search.toLowerCase())).map(item => <button key={item.id} className={selected === item.id ? "selected" : ""} onClick={() => { selectPatient(item.id); setSelected(item.id); }}><strong>{item.name}</strong><span>{item.email}</span><small>{item.message_count} messages{item.latest_activity ? ` · ${new Date(item.latest_activity * 1000).toLocaleDateString()}` : " · No activity yet"}</small></button>)}</aside>
      {patient ? <PatientReview key={patient.id} patient={patient} /> : <section className="card portal-empty"><span aria-hidden="true">♡</span><h2>Select a patient</h2><p>Open a patient’s activity to review their conversations, signals, and explanations.</p></section>}
    </div>
  </div>;
}
