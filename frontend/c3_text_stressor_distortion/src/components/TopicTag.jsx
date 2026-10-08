import React from "react";
import { Tag } from "lucide-react";

// A life-theme tag (Work, Relationships, Money...) — categorical identity,
// not a good/bad status, so it wears a neutral color and an icon instead of
// the status dot used by StatusBadge. Omitted entirely when the topic model
// wasn't confident enough to assign one.
export default function TopicTag({ topic }) {
  if (!topic) return null;
  const label = topic.theme.split(" · ")[0];
  return (
    <span
      className="topic-tag"
      title={`${topic.theme} (${Math.round(topic.similarity * 100)}% match)`}
    >
      <Tag size={11} />
      {label}
    </span>
  );
}
