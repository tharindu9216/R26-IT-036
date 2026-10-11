import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { StressSourceKey, WellbeingSummary as Summary } from "../types";

const PERIODS = [7, 14, 30] as const;

// Plotted sources. Colour follows the source, never its position, so hiding a
// source with no readings never repaints the others. Facial stress is left
// out of the chart on purpose: it is the least validated model (see
// config.yaml facial_stress) and only appears as a tile.
const CHART_SERIES: { key: StressSourceKey; label: string; color: string }[] = [
  { key: "combined", label: "Check-in + wearable", color: "var(--viz-series-1)" },
  { key: "text", label: "Written words", color: "var(--viz-series-2)" },
  { key: "voice", label: "Voice tone", color: "var(--viz-series-3)" },
];

const SOURCE_NOTES: Partial<Record<StressSourceKey, string>> = {
  questionnaire: "Six-question check-in",
  sensor: "BVP/EDA wearable",
  face: "Low-reliability camera model",
};

const EMOTION_LABELS: Record<string, string> = {
  neutral: "Neutral", joy: "Joy", sadness: "Sadness", fear: "Fear", anger: "Anger",
};

function percent(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : `${Math.round(value * 100)}%`;
}

function dayLabel(iso: string, long = false): string {
  const date = new Date(`${iso}T00:00:00`);
  return date.toLocaleDateString(undefined, long
    ? { weekday: "long", month: "short", day: "numeric" }
    : { weekday: "short", day: "numeric" });
}

/** "▲ 12 pts" against the previous period, or nothing to compare with. */
function Change({ value, unit = "pts", higherIsWorse = true }: { value: number | null; unit?: string; higherIsWorse?: boolean }) {
  if (value === null) return <span className="summary-change">No earlier data to compare</span>;
  const points = Math.round(value * (unit === "pts" ? 100 : 1));
  if (points === 0) return <span className="summary-change">No change from the previous period</span>;
  const worse = higherIsWorse ? points > 0 : points < 0;
  return (
    <span className={`summary-change ${worse ? "worse" : "better"}`}>
      <span aria-hidden="true">{points > 0 ? "▲" : "▼"}</span> {Math.abs(points)} {unit} vs previous period
    </span>
  );
}

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T | null>(null);
  const [width, setWidth] = useState(640);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(280, entry.contentRect.width)));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

function StressTrend({ summary }: { summary: Summary }) {
  const [containerRef, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const series = CHART_SERIES.filter(item => summary.daily.some(day => day.stress[item.key] !== null));
  const height = 240;
  // Narrow screens keep only the value at each line's end; the legend above
  // and the table view still name every line.
  const compact = width < 520;
  const pad = { top: 16, right: compact ? 44 : 118, bottom: 30, left: 40 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const count = summary.daily.length;
  const x = (index: number) => pad.left + (count === 1 ? plotWidth / 2 : (index / (count - 1)) * plotWidth);
  const y = (value: number) => pad.top + (1 - value) * plotHeight;
  const labelEvery = count <= 7 && width >= 420 ? 1 : count <= 14 ? 2 : 5;

  if (series.length === 0) {
    return <p className="summary-empty">No stress readings in this period yet.</p>;
  }

  function path(key: StressSourceKey): string {
    // A day without a reading breaks the line rather than being drawn as calm.
    let d = "";
    let pen = false;
    summary.daily.forEach((day, index) => {
      const value = day.stress[key];
      if (value === null) { pen = false; return; }
      d += `${pen ? "L" : "M"}${x(index).toFixed(1)},${y(value).toFixed(1)}`;
      pen = true;
    });
    return d;
  }

  function lastPoint(key: StressSourceKey) {
    for (let index = count - 1; index >= 0; index -= 1) {
      const value = summary.daily[index].stress[key];
      if (value !== null) return { index, value };
    }
    return null;
  }

  // Direct labels at each line's last reading, nudged apart so they never overlap.
  const endLabels = series
    .map(item => ({ item, point: lastPoint(item.key) }))
    .filter((entry): entry is { item: typeof series[number]; point: { index: number; value: number } } => entry.point !== null)
    .map(entry => ({ ...entry, labelY: y(entry.point.value) }))
    .sort((a, b) => a.labelY - b.labelY);
  for (let i = 1; i < endLabels.length; i += 1) {
    endLabels[i].labelY = Math.max(endLabels[i].labelY, endLabels[i - 1].labelY + 15);
  }

  function pick(clientX: number, rect: DOMRect) {
    const relative = ((clientX - rect.left) / rect.width) * width;
    const index = count === 1 ? 0 : Math.round(((relative - pad.left) / plotWidth) * (count - 1));
    setHover(Math.min(count - 1, Math.max(0, index)));
  }

  const hovered = hover === null ? null : summary.daily[hover];
  const tooltipLeft = hover === null ? 0 : Math.min(Math.max(x(hover) - 90, 0), width - 190);

  return (
    <div className="summary-chart" ref={containerRef}>
      <ul className="summary-legend" aria-label="Chart legend">
        {series.map(item => (
          <li key={item.key}><i style={{ background: item.color }} aria-hidden="true" />{item.label}</li>
        ))}
        <li className="summary-legend-threshold"><i aria-hidden="true" />Stress threshold (50%)</li>
      </ul>
      <svg
        width={width}
        height={height}
        role="img"
        aria-label={`Daily average stress by source, ${dayLabel(summary.start)} to ${dayLabel(summary.end)}. Use the arrow keys to read each day.`}
        tabIndex={0}
        onPointerMove={event => pick(event.clientX, event.currentTarget.getBoundingClientRect())}
        onPointerLeave={() => setHover(null)}
        onFocus={() => setHover(count - 1)}
        onBlur={() => setHover(null)}
        onKeyDown={event => {
          if (event.key === "ArrowLeft") setHover(current => Math.max(0, (current ?? count) - 1));
          if (event.key === "ArrowRight") setHover(current => Math.min(count - 1, (current ?? -1) + 1));
        }}
      >
        {[0, 0.25, 0.5, 0.75, 1].map(tick => (
          <g key={tick}>
            <line x1={pad.left} x2={pad.left + plotWidth} y1={y(tick)} y2={y(tick)}
              className={tick === 0.5 ? "summary-threshold" : "summary-grid"} />
            <text x={pad.left - 8} y={y(tick)} className="summary-axis" textAnchor="end" dominantBaseline="middle">
              {Math.round(tick * 100)}%
            </text>
          </g>
        ))}
        {summary.daily.map((day, index) => index % labelEvery === 0 || index === count - 1 ? (
          <text key={day.date} x={x(index)} y={height - 8} className="summary-axis" textAnchor="middle">
            {dayLabel(day.date)}
          </text>
        ) : null)}
        {hover !== null && <line className="summary-crosshair" x1={x(hover)} x2={x(hover)} y1={pad.top} y2={pad.top + plotHeight} />}
        {series.map(item => (
          <g key={item.key}>
            <path d={path(item.key)} fill="none" stroke={item.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
            {summary.daily.map((day, index) => {
              const value = day.stress[item.key];
              if (value === null) return null;
              const isolated = (index === 0 || summary.daily[index - 1].stress[item.key] === null)
                && (index === count - 1 || summary.daily[index + 1].stress[item.key] === null);
              if (!isolated && hover !== index) return null;
              return <circle key={day.date} cx={x(index)} cy={y(value)} r={4.5} fill={item.color} stroke="var(--surface-solid)" strokeWidth={2} />;
            })}
          </g>
        ))}
        {endLabels.map(({ item, point, labelY }) => (
          <text key={item.key} x={x(point.index) + 10} y={labelY} className="summary-end-label" dominantBaseline="middle">
            {compact ? percent(point.value) : `${item.label} ${percent(point.value)}`}
          </text>
        ))}
      </svg>
      {hovered && (
        <div className="summary-tooltip" style={{ left: tooltipLeft }} role="status">
          <strong>{dayLabel(hovered.date, true)}</strong>
          {series.map(item => (
            <div key={item.key}>
              <i style={{ background: item.color }} aria-hidden="true" />
              <b>{percent(hovered.stress[item.key])}</b> {item.label}
            </div>
          ))}
          <small>{hovered.check_ins} check-in{hovered.check_ins === 1 ? "" : "s"}</small>
        </div>
      )}
    </div>
  );
}

function DailyTable({ summary }: { summary: Summary }) {
  return (
    <div className="summary-table-wrap">
      <table className="summary-table">
        <thead>
          <tr><th scope="col">Day</th><th scope="col">Check-ins</th><th scope="col">Check-in + wearable</th><th scope="col">Written words</th><th scope="col">Voice tone</th><th scope="col">Face</th><th scope="col">Main emotion</th><th scope="col">Safety</th></tr>
        </thead>
        <tbody>
          {[...summary.daily].reverse().map(day => (
            <tr key={day.date}>
              <th scope="row">{dayLabel(day.date)}</th>
              <td>{day.check_ins}</td>
              <td>{percent(day.stress.combined)}</td>
              <td>{percent(day.stress.text)}</td>
              <td>{percent(day.stress.voice)}</td>
              <td>{percent(day.stress.face)}</td>
              <td>{day.dominant_emotion ? EMOTION_LABELS[day.dominant_emotion] ?? day.dominant_emotion : "—"}</td>
              <td>{day.crisis ? `${day.crisis} crisis` : day.contacts_offered ? `${day.contacts_offered} contact offer` : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function WellbeingSummary() {
  const [days, setDays] = useState<number>(7);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [error, setError] = useState("");
  const [showTable, setShowTable] = useState(false);

  useEffect(() => {
    let active = true;
    setError("");
    api.wellbeingSummary(days)
      .then(result => { if (active) setSummary(result); })
      .catch((caught: unknown) => { if (active) setError(caught instanceof Error ? caught.message : "Could not load the summary."); });
    return () => { active = false; };
  }, [days]);

  if (error) return <section className="card portal-panel"><p className="error" role="alert">{error}</p></section>;
  if (!summary) return <section className="card portal-panel"><p role="status">Loading summary…</p></section>;

  const { totals } = summary;
  const sources = Object.fromEntries(summary.sources.map(item => [item.key, item]));
  const combined = sources.combined;
  const otherSources = summary.sources.filter(item => item.key in SOURCE_NOTES);
  const negativeChange = summary.negative_share !== null && summary.previous_negative_share !== null
    ? summary.negative_share - summary.previous_negative_share
    : null;
  const safetyCount = totals.crisis + totals.contacts_offered;

  return (
    <section className="wellbeing-summary">
      <div className="summary-toolbar">
        <div>
          <h3>Wellbeing summary</h3>
          <small>{dayLabel(summary.start, true)} – {dayLabel(summary.end, true)}</small>
        </div>
        <div className="summary-periods" role="group" aria-label="Summary period">
          {PERIODS.map(period => (
            <button key={period} aria-pressed={days === period} onClick={() => setDays(period)}>{period} days</button>
          ))}
        </div>
      </div>

      {totals.check_ins === 0 ? (
        <div className="card portal-empty">
          <h3>No check-ins in the last {summary.days} days</h3>
          <p>The summary fills in as this patient uses written or voice check-ins.</p>
        </div>
      ) : <>
        <div className="summary-tiles">
          <article className="card summary-tile">
            <span>Check-ins</span>
            <strong>{totals.check_ins}</strong>
            <small>On {totals.active_days} of {summary.days} days · {totals.text} written, {totals.voice} voice</small>
            <Change value={totals.previous_check_ins ? totals.check_ins - totals.previous_check_ins : null} unit="check-ins" higherIsWorse={false} />
          </article>
          <article className="card summary-tile">
            <span>Average stress (check-in + wearable)</span>
            <strong>{percent(combined?.average)}</strong>
            <small>{combined?.readings ?? 0} readings · 50% and above counts as stressed</small>
            <Change value={combined?.change ?? null} />
          </article>
          <article className="card summary-tile">
            <span>Negative emotions</span>
            <strong>{percent(summary.negative_share)}</strong>
            <small>Share of check-ins read as sadness, fear or anger</small>
            <Change value={negativeChange} />
          </article>
          <article className={`card summary-tile${totals.crisis ? " alert" : ""}`}>
            <span>Safety</span>
            <strong>{safetyCount}</strong>
            <small>
              {totals.crisis ? <><span aria-hidden="true">⚠ </span>{totals.crisis} crisis response{totals.crisis === 1 ? "" : "s"}</> : "No crisis responses"}
              {" · "}{totals.contacts_offered} support-contact offer{totals.contacts_offered === 1 ? "" : "s"}
            </small>
          </article>
        </div>

        <article className="card portal-panel">
          <div className="activity-heading">
            <h3>Daily stress by source</h3>
            <button className="portal-link" onClick={() => setShowTable(value => !value)}>
              {showTable ? "Show chart" : "Show as table"}
            </button>
          </div>
          <p className="summary-note">Each line is one model’s daily average, kept separate because they measure different things. Gaps are days without that reading.</p>
          {showTable ? <DailyTable summary={summary} /> : <StressTrend summary={summary} />}
        </article>

        <div className="summary-columns">
          <article className="card portal-panel">
            <h3>Emotion mix</h3>
            <ul className="summary-bars">
              {summary.emotions.map(item => (
                <li key={item.label}>
                  <span>{EMOTION_LABELS[item.label] ?? item.label}</span>
                  <div className="summary-bar-track"><i style={{ width: `${Math.max(item.share * 100, 2)}%` }} /></div>
                  <b>{Math.round(item.share * 100)}%</b>
                  <small>{item.count}</small>
                </li>
              ))}
            </ul>
          </article>
          <article className="card portal-panel">
            <h3>Other readings</h3>
            <dl className="summary-readings">
              {otherSources.map(item => (
                <div key={item.key}>
                  <dt>{item.label}<small>{SOURCE_NOTES[item.key]} · {item.readings} readings</small></dt>
                  <dd>{percent(item.average)}</dd>
                </div>
              ))}
            </dl>
            <h3>Worth a look</h3>
            <ul className="summary-flags">
              <li><b>{totals.high_deviation}</b> sharp mood swings from the previous check-in</li>
              <li><b>{totals.high_intensity_forecasts}</b> check-ins forecast to intensify next</li>
              <li><b>{totals.out_of_scope}</b> requests outside emotional support</li>
            </ul>
          </article>
        </div>

        {summary.events.length > 0 && (
          <article className="card portal-panel">
            <h3>Safety events</h3>
            <ul className="summary-events">
              {summary.events.map(event => (
                <li key={event.time + event.kind} className={event.kind}>
                  <span aria-hidden="true">{event.kind === "crisis" ? "⚠" : "☎"}</span>
                  <b>{event.kind === "crisis" ? "Crisis response" : "Support contacts offered"}</b>
                  <time dateTime={event.time}>{new Date(event.time).toLocaleString()}</time>
                  <small>{event.mode === "voice" ? "Voice" : "Written"} check-in</small>
                </li>
              ))}
            </ul>
          </article>
        )}
      </>}
      <p className="summary-note">Model estimates to support your judgement, not a diagnosis. Message text is not kept beyond a day; these figures are kept for 90 days.</p>
    </section>
  );
}
