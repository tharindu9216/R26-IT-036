import React from "react";
import { fusionForEntry, fusionTone } from "./fusionUtils";

export default function FusionBadge({ entry, fusion: suppliedFusion }) {
  const fusion = suppliedFusion ?? fusionForEntry(entry);
  return (
    <span className={`fusion-badge fusion-${fusionTone(fusion.state)}`}>
      <span className="fusion-badge-mark" aria-hidden="true" />
      {fusion.label}
    </span>
  );
}
