import React from "react";

function scoreClass(value) {
  if (value > 0) return "score-positive";
  if (value < 0) return "score-negative";
  return "";
}

function MethodEvidenceTable({ features, limeR2 }) {
  if (!features?.length) return null;
  return (
    <div className="method-evidence">
      <div className="method-evidence-meta">
        <strong>Combined word evidence</strong>
        {Number.isFinite(limeR2) && <span>LIME local R² {limeR2.toFixed(3)}</span>}
      </div>
      <div className="method-evidence-scroll">
        <table>
          <thead>
            <tr>
              <th>Word</th>
              <th>IG</th>
              <th>SHAP</th>
              <th>LIME</th>
              <th>Combined</th>
              <th>Agree</th>
            </tr>
          </thead>
          <tbody>
            {features.slice(0, 10).map((feature, index) => (
              <tr key={`${feature.word}-${index}`}>
                <th>{feature.word}</th>
                {[
                  feature.integrated_gradients,
                  feature.shap,
                  feature.lime,
                  feature.consensus,
                ].map((score, scoreIndex) => (
                  <td className={scoreClass(score)} key={scoreIndex}>
                    {score.toFixed(3)}
                  </td>
                ))}
                <td>{Math.round(feature.agreement * 100)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="method-evidence-note">
        Positive scores support the explained class; negative scores oppose it.
      </p>
    </div>
  );
}

export default function TokenEvidence({
  title,
  items,
  tone,
  features,
  limeR2,
  showConsensusLabel = true,
}) {
  return (
    <div className={`token-evidence evidence-${tone}`}>
      <div className="token-evidence-title">
        <strong>{title}</strong>
        {showConsensusLabel && (
          <span>Representative DeBERTa-v3 · IG + SHAP + LIME consensus</span>
        )}
      </div>
      {items?.length ? (
        <div className="token-evidence-list">
          {items.map((item, index) => (
            <span key={`${item.text}-${index}`} title={`Attribution ${item.score.toFixed(3)}`}>
              {item.text}
            </span>
          ))}
        </div>
      ) : (
        <p>No clear consensus phrase passed the attribution threshold.</p>
      )}
      <MethodEvidenceTable features={features} limeR2={limeR2} />
    </div>
  );
}
