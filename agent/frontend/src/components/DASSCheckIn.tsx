import { useEffect, useState } from "react";
import { api } from "../api";
import type { DASS21Form, DASS21Status } from "../types";

function readDraft(key: string): Record<string, number> {
  try {
    const draft: unknown = JSON.parse(sessionStorage.getItem(key) ?? "{}");
    if (!draft || typeof draft !== "object" || Array.isArray(draft)) return {};
    return Object.fromEntries(Object.entries(draft).filter(([id, value]) =>
      /^(?:[1-9]|1[0-9]|2[01])$/.test(id) && Number.isInteger(value) && Number(value) >= 0 && Number(value) <= 3));
  } catch { return {}; }
}

export function DASSCheckIn({ patientId, onDone }: { patientId: string; onDone: () => void }) {
  const draftKey = `dass21-adult-en-v1:${patientId}`;
  const [answers, setAnswers] = useState<Record<string, number>>(() => readDraft(draftKey));
  const [index, setIndex] = useState(0);
  const [form, setForm] = useState<DASS21Form | null>(null);
  const [completion, setCompletion] = useState<DASS21Status | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let active = true;
    Promise.all([api.dass21Form(), api.dass21Status()]).then(([definition, status]) => {
      if (active) { setForm(definition); setCompletion(status); setError(""); }
    }).catch(() => { if (active) setError("We could not open your assessment. Please try again."); });
    return () => { active = false; };
  }, [attempt]);

  function answer(value: number) {
    if (!form) return;
    const next = { ...answers, [form.questions[index].id]: value };
    setAnswers(next);
    try { sessionStorage.setItem(draftKey, JSON.stringify(next)); } catch { /* Keep in memory when storage is unavailable. */ }
  }

  async function next() {
    if (!form) return;
    setError("");
    if (answers[form.questions[index].id] === undefined) return;
    if (index < form.questions.length - 1) { setIndex(index + 1); return; }
    const missing = form.questions.findIndex(question => answers[question.id] === undefined);
    if (missing >= 0) { setIndex(missing); setError("Please answer each statement before completing your assessment."); return; }
    setBusy(true);
    try {
      setCompletion(await api.submitDASS21(answers));
      try { sessionStorage.removeItem(draftKey); } catch { /* The saved server assessment remains available. */ }
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Could not save your assessment."); }
    finally { setBusy(false); }
  }

  if (!form || !completion) return <section className="card portal-panel">
    <h2>Getting your assessment ready…</h2>
    {error ? <><p className="error" role="alert">{error}</p><button onClick={() => setAttempt(attempt + 1)}>Try again</button></> : <p role="status">Loading your check-in.</p>}
  </section>;

  if (completion.completed) return <section className="card portal-panel">
    <span className="eyebrow">DASS-21 · PAST-WEEK CHECK-IN</span>
    <h2>Assessment completed</h2>
    <p>Your answers have been saved for your linked doctor to review.</p>
    {completion.completed_at && <p>Completed: {new Date(completion.completed_at * 1000).toLocaleString()}</p>}
    <p>You can use this assessment for both text and voice conversations. You will be asked again after seven days.</p>
    <button onClick={onDone}>Continue to your conversation</button>
    <button className="portal-link" onClick={() => { setCompletion({ completed: false }); setAnswers({}); setIndex(0); try { sessionStorage.removeItem(draftKey); } catch { /* Optional draft storage. */ } }}>Take the assessment again</button>
  </section>;

  const question = form.questions[index];
  const selected = answers[question.id];
  return <section className="card portal-panel check-in dass-check-in">
    <span className="eyebrow">DASS-21 · QUESTION {index + 1} OF {form.questions.length}</span>
    <progress max={form.questions.length} value={index + 1} aria-label="Assessment progress" />
    <p><strong>{form.timeframe}</strong></p>
    <p>{form.instructions}</p>
    <h2>{question.text}</h2>
    <div className="check-in-options">{form.options.map(option => <button key={option.value} disabled={busy} aria-pressed={selected === option.value} onClick={() => answer(option.value)}><strong>{option.value}</strong> {option.label}</button>)}</div>
    {error && <p className="error" role="alert">{error}</p>}
    <div className="dass-navigation">
      <button disabled={busy || index === 0} onClick={() => setIndex(index - 1)}>Back</button>
      <button disabled={busy || selected === undefined} onClick={() => void next()}>{busy ? "Saving…" : index === form.questions.length - 1 ? "Complete assessment" : "Next"}</button>
    </div>
  </section>;
}
