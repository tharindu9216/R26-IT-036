import React, { useEffect, useMemo, useState } from "react";
import {
  BookHeart,
  BrainCircuit,
  CalendarDays,
  ChartColumn,
  CheckCircle2,
  Feather,
  History,
  Loader2,
  NotebookPen,
  Search,
  Sparkles,
  X,
} from "lucide-react";
import {
  createDiaryEntry,
  deleteDiaryEntry,
  getDiarySummary,
  listDiaryEntries,
} from "./api";
import { formatTime, isToday } from "./utils";
import StatusBadge from "./components/StatusBadge";
import TopicTag from "./components/TopicTag";
import EntryList from "./components/EntryList";
import EntryDetail from "./components/EntryDetail";
import TrendChart from "./components/TrendChart";
import ConfirmDeleteButton from "./components/ConfirmDeleteButton";
import FusionBadge from "./features/fusion/FusionBadge";

const NAV_ITEMS = [
  { id: "today", label: "Today", description: "Write and reflect", icon: CalendarDays },
  { id: "timeline", label: "Timeline", description: "Revisit your days", icon: History },
  { id: "insights", label: "Insights", description: "Notice patterns", icon: ChartColumn },
];

export default function App() {
  const [page, setPage] = useState("today");
  const [entries, setEntries] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    refresh();
  }, []);

  async function refresh() {
    setError("");
    try {
      const data = await listDiaryEntries();
      setEntries(data);
    } catch (err) {
      setError(err.message);
      setEntries([]);
    }
  }

  async function handleCreate(title, content) {
    const entry = await createDiaryEntry(content, title);
    setError("");
    setEntries((current) => [entry, ...(current ?? [])]);
    return entry;
  }

  async function handleDelete(id) {
    const previous = entries;
    setEntries((current) => (current ?? []).filter((entry) => entry.id !== id));
    try {
      await deleteDiaryEntry(id);
      setError("");
    } catch (err) {
      setError(err.message);
      setEntries(previous);
      throw err;
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="Main navigation">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <BookHeart size={23} />
          </div>
          <div className="brand-copy">
            <span>Daily reflection</span>
            <h1>My Diary</h1>
            <p>
              {entries === null
                ? "Your pages are loading"
                : `${entries.length} entr${entries.length === 1 ? "y" : "ies"} so far`}
            </p>
          </div>
        </div>

        <nav className="nav-tabs">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const active = page === item.id;
            return (
              <button
                key={item.id}
                className={active ? "active" : ""}
                onClick={() => setPage(item.id)}
                aria-label={item.label}
                aria-current={active ? "page" : undefined}
              >
                <span className="nav-icon" aria-hidden="true">
                  <Icon size={18} />
                </span>
                <span className="nav-copy">
                  <strong>{item.label}</strong>
                  <small>{item.description}</small>
                </span>
              </button>
            );
          })}
        </nav>

        <div className="sidebar-note">
          <Sparkles size={16} aria-hidden="true" />
          <div>
            <strong>Gentle, automated notes</strong>
            <p>Patterns may support reflection, but they are not a diagnosis.</p>
          </div>
        </div>
      </aside>

      <main className="workspace">
        {error && (
          <div className="error-banner" role="alert">
            <span>{error}</span>
            <button type="button" onClick={() => setError("")} aria-label="Dismiss error">
              <X size={16} />
            </button>
          </div>
        )}
        {page === "today" && (
          <TodayPage entries={entries} onCreate={handleCreate} onDelete={handleDelete} />
        )}
        {page === "timeline" && <TimelinePage entries={entries} onDelete={handleDelete} />}
        {page === "insights" && <InsightsPage />}
      </main>
    </div>
  );
}

function TodayPage({ entries, onCreate, onDelete }) {
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState("");
  const [saveSuccess, setSaveSuccess] = useState(false);
  const todayLine = useMemo(
    () =>
      new Date().toLocaleDateString(undefined, {
        weekday: "long",
        month: "long",
        day: "numeric",
      }),
    []
  );

  const todaysEntries = useMemo(
    () => (entries ?? []).filter((entry) => isToday(entry.created_at)),
    [entries]
  );
  const wordCount = content.trim() ? content.trim().split(/\s+/).length : 0;

  useEffect(() => {
    if (!saveSuccess) return undefined;
    const timeoutId = window.setTimeout(() => setSaveSuccess(false), 3200);
    return () => window.clearTimeout(timeoutId);
  }, [saveSuccess]);

  async function handleSubmit(event) {
    event.preventDefault();
    const text = content.trim();
    if (!text || saving) return;

    const submittedTitle = title.trim();
    setSaving(true);
    setSaveError("");
    setSaveSuccess(false);
    try {
      await onCreate(submittedTitle, text);
      setTitle("");
      setContent("");
      setSaveSuccess(true);
    } catch (err) {
      setSaveError(err.message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="page today-page">
      <header className="page-intro today-intro">
        <div>
          <p className="eyebrow">Your daily page</p>
          <h2>A little space for today.</h2>
          <p>{todayLine} · You can start with one honest sentence.</p>
        </div>
        <div className="today-count" aria-label={`${todaysEntries.length} entries today`}>
          <span>{todaysEntries.length}</span>
          <small>{todaysEntries.length === 1 ? "entry" : "entries"} today</small>
        </div>
      </header>

      <form className="composer" onSubmit={handleSubmit}>
        <div className="composer-topline">
          <div className="composer-heading">
            <span className="composer-icon" aria-hidden="true">
              <NotebookPen size={18} />
            </span>
            <div>
              <span>New reflection</span>
              <small>Only write what feels useful</small>
            </div>
          </div>
          {saveSuccess && (
            <span className="save-success" role="status">
              <CheckCircle2 size={15} /> Saved
            </span>
          )}
        </div>

        <input
          className="composer-title"
          value={title}
          onChange={(event) => setTitle(event.target.value)}
          placeholder="Give this page a title (optional)"
          aria-label="Entry title"
          maxLength={200}
        />
        <textarea
          value={content}
          onChange={(event) => setContent(event.target.value)}
          placeholder="Start writing here…"
          aria-label="New diary entry"
          rows={8}
          maxLength={8000}
        />
        <div className="composer-footer">
          <span className="composer-hint">
            <Feather size={13} />
            {wordCount === 0 ? "There is no right way to begin." : `${wordCount} ${wordCount === 1 ? "word" : "words"}`}
          </span>
          <button type="submit" disabled={saving || !content.trim()}>
            {saving ? <Loader2 size={16} className="spin" /> : <Feather size={15} />}
            {saving ? "Saving…" : "Save reflection"}
          </button>
        </div>
      </form>

      {saveError && <div className="error-line" role="alert">{saveError}</div>}

      <div className="section-heading">
        <div>
          <p className="eyebrow">Today</p>
          <h3>Your reflections</h3>
        </div>
        {todaysEntries.length > 0 && <span>{todaysEntries.length} saved</span>}
      </div>

      {entries === null ? (
        <LoadingState label="Opening your diary…" />
      ) : todaysEntries.length > 0 ? (
        <div className="today-entries">
          {todaysEntries.map((entry) => (
            <article className="entry-card" key={entry.id}>
              <div className="entry-card-header">
                <time dateTime={entry.created_at}>{formatTime(entry.created_at)}</time>
                <ConfirmDeleteButton onConfirm={() => onDelete(entry.id)} label="Delete entry" compact />
              </div>
              {entry.title && <h4 className="entry-card-title">{entry.title}</h4>}
              <p className="entry-content">{entry.content}</p>
              <div className="entry-signals">
                <span>Automated reflection notes</span>
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
            </article>
          ))}
        </div>
      ) : (
        <div className="empty-card">
          <Feather size={23} />
          <div>
            <strong>Your page is still open.</strong>
            <p>When you save a reflection, it will appear here.</p>
          </div>
        </div>
      )}
    </section>
  );
}

function TimelinePage({ entries, onDelete }) {
  const [selectedId, setSelectedId] = useState(null);
  const [query, setQuery] = useState("");

  const visibleEntries = useMemo(() => {
    const allEntries = entries ?? [];
    const normalizedQuery = query.trim().toLowerCase();
    if (!normalizedQuery) return allEntries;
    return allEntries.filter((entry) => {
      const searchText = [
        entry.title,
        entry.content,
        entry.topic?.theme,
        entry.stress?.label,
        entry.distortion?.label,
        entry.fusion?.label,
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return searchText.includes(normalizedQuery);
    });
  }, [entries, query]);

  useEffect(() => {
    if (visibleEntries.length === 0) {
      if (selectedId !== null) setSelectedId(null);
      return;
    }
    if (!visibleEntries.some((entry) => entry.id === selectedId)) {
      setSelectedId(visibleEntries[0].id);
    }
  }, [visibleEntries, selectedId]);

  const selected = visibleEntries.find((entry) => entry.id === selectedId) ?? visibleEntries[0];

  async function handleDelete(id) {
    const remaining = visibleEntries.filter((entry) => entry.id !== id);
    setSelectedId(remaining.length > 0 ? remaining[0].id : null);
    await onDelete(id);
  }

  return (
    <section className="page timeline-page">
      <header className="page-intro timeline-intro">
        <div>
          <p className="eyebrow">Your story over time</p>
          <h2>Timeline</h2>
          <p>Return to a thought, a feeling, or a day you want to remember.</p>
        </div>
        <label className="search-field">
          <Search size={16} aria-hidden="true" />
          <span className="sr-only">Search diary entries</span>
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search your entries"
          />
        </label>
      </header>

      {entries === null ? (
        <LoadingState label="Opening your timeline…" />
      ) : entries.length === 0 ? (
        <div className="empty-card timeline-empty">
          <History size={26} />
          <div>
            <strong>Your timeline begins with your first reflection.</strong>
            <p>Go to Today whenever you are ready to write.</p>
          </div>
        </div>
      ) : visibleEntries.length === 0 ? (
        <div className="empty-card timeline-empty">
          <Search size={24} />
          <div>
            <strong>No reflections match “{query}”.</strong>
            <p>Try another word or clear your search.</p>
          </div>
          <button type="button" className="text-button" onClick={() => setQuery("")}>Clear search</button>
        </div>
      ) : (
        <div className="timeline-layout">
          <EntryList entries={visibleEntries} selectedId={selected?.id ?? null} onSelect={setSelectedId} />
          <EntryDetail entry={selected} onDelete={handleDelete} />
        </div>
      )}
    </section>
  );
}

function InsightsPage() {
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getDiarySummary()
      .then(setSummary)
      .catch((err) => setError(err.message));
  }, []);

  return (
    <section className="page insights-page">
      <header className="page-intro insights-intro">
        <div>
          <p className="eyebrow">Patterns across your pages</p>
          <h2>Reflection insights</h2>
          <p>See broad changes over time without turning a single entry into a conclusion.</p>
        </div>
      </header>

      <div className="insight-notice">
        <Sparkles size={18} aria-hidden="true" />
        <div>
          <strong>A guide for reflection—not a clinical assessment.</strong>
          <p>These summaries come from automated models and can be wrong. Use them as prompts, not facts.</p>
        </div>
      </div>

      {error ? (
        <div className="error-line" role="alert">{error}</div>
      ) : !summary ? (
        <LoadingState label="Gathering your patterns…" />
      ) : (
        <>
          <div className="stat-tiles">
            <StatTile
              icon={NotebookPen}
              label="Reflections saved"
              value={summary.total_entries}
              sub="Across your diary"
            />
            <StatTile
              icon={CalendarDays}
              label="Stress signals"
              value={`${summary.stressed_pct.toFixed(0)}%`}
              sub={`${summary.stressed_count} of ${summary.total_entries} entries`}
              progress={summary.stressed_pct}
              tone="warm"
            />
            <StatTile
              icon={BrainCircuit}
              label="Thinking-pattern signals"
              value={`${summary.distortion_pct.toFixed(0)}%`}
              sub={`${summary.distortion_count} of ${summary.total_entries} entries`}
              progress={summary.distortion_pct}
              tone="plum"
            />
          </div>

          {summary.total_entries === 0 ? (
            <div className="empty-card timeline-empty">
              <ChartColumn size={26} />
              <div>
                <strong>Your patterns will take shape over time.</strong>
                <p>Write a few reflections before drawing meaning from a trend.</p>
              </div>
            </div>
          ) : (
            <div className="trend-grid">
              <TrendChart
                title="Stress signals · last 14 days"
                data={summary.trend}
                positiveKey="stressed"
                positiveLabel="Stress signal"
                negativeLabel="No stress signal"
              />
              <TrendChart
                title="Thinking patterns · last 14 days"
                data={summary.trend}
                positiveKey="distorted"
                positiveLabel="Pattern detected"
                negativeLabel="No pattern detected"
              />
            </div>
          )}
        </>
      )}
    </section>
  );
}

function StatTile({ icon: Icon, label, value, sub, progress, tone = "ink" }) {
  return (
    <article className={`stat-tile card stat-${tone}`}>
      <div className="stat-topline">
        <span className="stat-icon" aria-hidden="true"><Icon size={17} /></span>
        <p className="stat-label">{label}</p>
      </div>
      <p className="stat-value">{value}</p>
      <p className="stat-sub">{sub}</p>
      {typeof progress === "number" && (
        <div className="stat-progress" aria-label={`${label}: ${progress.toFixed(0)} percent`}>
          <span style={{ width: `${Math.max(0, Math.min(progress, 100))}%` }} />
        </div>
      )}
    </article>
  );
}

function LoadingState({ label }) {
  return (
    <div className="empty-state" role="status">
      <Loader2 size={23} className="spin" />
      <p>{label}</p>
    </div>
  );
}
