import { useEffect, useState } from "react";
import { api } from "../api";
import type { DASS21Assessment, DASS21Form, DASS21Scale } from "../types";

const scales: DASS21Scale[] = ["depression", "anxiety", "stress"];

export function DASSReview() {
  const [assessments, setAssessments] = useState<DASS21Assessment[]>([]);
  const [form, setForm] = useState<DASS21Form | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    let timer: number | undefined;
    async function refresh() {
      try {
        const result = await api.dass21History();
        if (active) { setAssessments(result.assessments); setForm(result.form); setError(""); }
      } catch {
        if (active) { setAssessments([]); setForm(null); setError("Could not load this patient's assessments."); }
      } finally {
        if (active) { setLoading(false); timer = window.setTimeout(refresh, 10000); }
      }
    }
    void refresh();
    return () => { active = false; window.clearTimeout(timer); };
  }, []);
  return <section className="card portal-panel dass-review">
    <span className="eyebrow">PAST-WEEK ASSESSMENTS</span>
    <h2>DASS-21 history</h2>
    <p>Depression, anxiety and stress symptom severity over the past week. These scores are not diagnoses and are kept separate from live wearable estimates.</p>
    {loading && <p role="status">Loading assessments…</p>}
    {error && <p className="error" role="alert">{error}</p>}
    {!loading && !error && assessments.length === 0 && <p>No completed DASS-21 assessment has been shared by this patient yet.</p>}
    {assessments.map((assessment, index) => <details className="dass-assessment" key={assessment.assessment_id} open={index === 0}>
      <summary>{new Date(assessment.completed_at * 1000).toLocaleString()}{index === 0 ? " · Latest assessment" : ""}</summary>
      <div className="dass-table-wrap"><table><caption>Each seven-item subscale sum is multiplied by two.</caption><thead><tr><th scope="col">Scale</th><th scope="col">Raw / 21</th><th scope="col">Score / 42</th><th scope="col">Severity</th></tr></thead><tbody>{scales.map(scale => <tr key={scale}><th scope="row" className="dass-scale">{scale}</th><td>{assessment.scores[scale].raw_score}</td><td>{assessment.scores[scale].score}</td><td>{assessment.scores[scale].severity}</td></tr>)}</tbody></table></div>
      <details><summary>Review the 21 responses</summary><ol>{form?.questions.map(question => <li key={question.id}><p>{question.text}</p><strong>{assessment.answers[question.id]} — {form.options.find(option => option.value === assessment.answers[question.id])?.label}</strong></li>)}</ol></details>
    </details>)}
  </section>;
}
