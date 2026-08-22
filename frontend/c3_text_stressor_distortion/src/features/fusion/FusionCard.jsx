import React from "react";
import { Network } from "lucide-react";
import FusionBadge from "./FusionBadge";
import FusionExplanation from "./FusionExplanation";
import { fusionForEntry } from "./fusionUtils";

export default function FusionCard({ entry }) {
  const fusion = fusionForEntry(entry);
  return (
    <section className="fusion-card" aria-label="Combined automated reflection">
      <div className="fusion-card-heading">
        <span aria-hidden="true"><Network size={16} /></span>
        <div>
          <strong>Combined reflection pattern</strong>
          <p>{fusion.summary}</p>
        </div>
      </div>
      <FusionBadge fusion={fusion} />
      <FusionExplanation entryId={entry.id} />
    </section>
  );
}
