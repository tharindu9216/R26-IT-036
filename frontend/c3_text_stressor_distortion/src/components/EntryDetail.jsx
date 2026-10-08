import React from "react";
import { BookOpen, Sparkles } from "lucide-react";
import { deriveTitle, formatFullDate } from "../utils";
import StatusBadge from "./StatusBadge";
import TopicTag from "./TopicTag";
import ConfirmDeleteButton from "./ConfirmDeleteButton";
import FusionBadge from "../features/fusion/FusionBadge";
import FusionCard from "../features/fusion/FusionCard";

export default function EntryDetail({ entry, onDelete }) {
  if (!entry) {
    return (
      <div className="entry-detail-pane entry-detail-empty">
        <BookOpen size={30} />
        <p>Select an entry to read it.</p>
      </div>
    );
  }

  return (
    <div className="entry-detail-pane">
      <div className="entry-detail-header">
        <div>
          <p className="entry-detail-date">{formatFullDate(entry.created_at)}</p>
          <h2>{deriveTitle(entry)}</h2>
        </div>
        <ConfirmDeleteButton onConfirm={() => onDelete(entry.id)} label="Delete entry" />
      </div>

      <p className="entry-detail-content">{entry.content}</p>

      <div className="detail-insights">
        <div className="detail-insights-heading">
          <Sparkles size={15} />
          <div>
            <strong>Automated reflection notes</strong>
            <span>Helpful prompts, not a diagnosis or statement of fact.</span>
          </div>
        </div>
        <div className="entry-tags">
          <TopicTag topic={entry.topic} />
          <StatusBadge
            isNegative={entry.stress.is_stressed}
            positiveLabel="Calm"
            negativeLabel={entry.stress.label}
          />
          <StatusBadge
            isNegative={entry.distortion.has_distortion}
            positiveLabel="Clear thinking"
            negativeLabel={entry.distortion.label}
          />
          <FusionBadge entry={entry} />
        </div>
      </div>

      <FusionCard entry={entry} />
    </div>
  );
}
