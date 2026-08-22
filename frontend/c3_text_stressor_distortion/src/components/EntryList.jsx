import React from "react";
import { deriveTitle, dayNumber, formatTime, groupByMonth, snippet } from "../utils";

export default function EntryList({ entries, selectedId, onSelect }) {
  const groups = groupByMonth(entries);

  return (
    <div className="entry-list-pane" aria-label="Entry list">
      <div className="entry-list-summary">
        <span>{entries.length} {entries.length === 1 ? "reflection" : "reflections"}</span>
        <small>Select one to read</small>
      </div>
      {groups.map((group) => (
        <div className="entry-list-group" key={group.key}>
          <p className="entry-list-month">{group.label}</p>
          {group.entries.map((entry) => (
            <button
              type="button"
              key={entry.id}
              className={`entry-list-row${entry.id === selectedId ? " active" : ""}`}
              onClick={() => onSelect(entry.id)}
            >
              <span className="day-tile">{dayNumber(entry.created_at)}</span>
              <span className="entry-list-text">
                <span className="entry-list-title">{deriveTitle(entry)}</span>
                {entry.title && (
                  <span className="entry-list-snippet">{snippet(entry.content, 60)}</span>
                )}
              </span>
              <span className="entry-list-time">{formatTime(entry.created_at)}</span>
            </button>
          ))}
        </div>
      ))}
    </div>
  );
}
