import React, { useEffect, useState } from "react";
import { AlertCircle, Eye, Loader2 } from "lucide-react";
import { explainDiaryEntry } from "../../api";
import TokenEvidence from "./TokenEvidence";

export default function FusionExplanation({ entryId }) {
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setExplanation(null);
    setError("");
    setLoading(false);
  }, [entryId]);

  async function handleExplain() {
    if (loading) return;
    setLoading(true);
    setError("");
    try {
      setExplanation(await explainDiaryEntry(entryId));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  if (!explanation) {
    return (
      <div className="fusion-explain-action">
        <button type="button" onClick={handleExplain} disabled={loading}>
          {loading ? <Loader2 size={14} className="spin" /> : <Eye size={14} />}
          {loading ? "Generating combined XAI…" : "Explain this combined result"}
        </button>
        <span>Runs IG + SHAP + LIME consensus; this can take several minutes.</span>
        {error && <p role="alert">{error}</p>}
      </div>
    );
  }

  return (
    <div className="fusion-explanation">
      <div className="fusion-rule-trace">
        <strong>Why this result was selected</strong>
        <p>{explanation.fusion_reason}</p>
      </div>

      <div className="fusion-evidence-grid">
        <TokenEvidence
          title="Stress-head evidence"
          items={explanation.stress_evidence}
          tone="stress"
          features={explanation.stress_feature_evidence}
          limeR2={explanation.stress_lime_r2}
        />
        <TokenEvidence
          title="CBT-head evidence"
          items={explanation.cbt_evidence}
          tone="pattern"
          features={explanation.cbt_feature_evidence}
          limeR2={explanation.cbt_lime_r2}
        />
      </div>

      {explanation.overlap_evidence.length > 0 && (
        <TokenEvidence
          title="Evidence shared by both heads"
          items={explanation.overlap_evidence}
          tone="overlap"
          showConsensusLabel={false}
        />
      )}

      {explanation.errors.length > 0 && (
        <div className="fusion-xai-warning" role="status">
          <AlertCircle size={14} />
          <span>
            The rule trace is available, but some combined XAI evidence could not be generated.
          </span>
        </div>
      )}

      <p className="fusion-disclaimer">{explanation.disclaimer}</p>
    </div>
  );
}
