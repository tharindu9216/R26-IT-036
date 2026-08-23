// Shared formatting helpers for entry lists/detail views.

export function deriveTitle(entry) {
  if (entry.title && entry.title.trim()) return entry.title.trim();
  const words = entry.content.trim().split(/\s+/);
  const short = words.slice(0, 8).join(" ");
  return words.length > 8 ? `${short}...` : short;
}

export function snippet(content, maxLen = 90) {
  const clean = content.trim().replace(/\s+/g, " ");
  return clean.length > maxLen ? `${clean.slice(0, maxLen)}...` : clean;
}

export function formatTime(dateStr) {
  return new Date(dateStr).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
}

export function formatFullDate(dateStr) {
  return new Date(dateStr).toLocaleDateString(undefined, {
    weekday: "long",
    month: "long",
    day: "numeric",
    year: "numeric",
  });
}

export function monthLabel(dateStr) {
  return new Date(dateStr).toLocaleDateString(undefined, { month: "long", year: "numeric" });
}

export function dayNumber(dateStr) {
  return new Date(dateStr).getDate();
}

export function isToday(dateStr) {
  return new Date(dateStr).toDateString() === new Date().toDateString();
}

export function groupByMonth(entries) {
  const groups = [];
  let lastKey = null;
  for (const entry of entries) {
    const date = new Date(entry.created_at);
    const key = `${date.getFullYear()}-${date.getMonth()}`;
    if (key !== lastKey) {
      groups.push({ key, label: monthLabel(entry.created_at), entries: [] });
      lastKey = key;
    }
    groups[groups.length - 1].entries.push(entry);
  }
  return groups;
}
