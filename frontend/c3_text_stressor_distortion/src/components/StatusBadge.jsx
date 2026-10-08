import React from "react";

// A quiet companion tag, not an alert — a colored dot plus label (never
// color alone) so it still passes as accessible status encoding, but reads
// as a gentle note about the entry rather than a pass/fail test result.
export default function StatusBadge({ isNegative, positiveLabel, negativeLabel }) {
  const tone = isNegative ? "critical" : "good";
  const label = isNegative ? negativeLabel : positiveLabel;
  return (
    <span className={`insight-tag insight-${tone}`}>
      <span className="insight-dot" />
      {label}
    </span>
  );
}
