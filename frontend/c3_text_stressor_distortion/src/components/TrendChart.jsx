import React, { useState } from "react";
import { Table2, BarChart3 } from "lucide-react";

const WIDTH = 640;
const HEIGHT = 200;
const PADDING_LEFT = 34;
const PADDING_BOTTOM = 24;
const PADDING_TOP = 12;
const BAR_MAX_THICKNESS = 22;
const SEGMENT_GAP = 2;

function niceMax(value) {
  if (value <= 0) return 1;
  const magnitude = 10 ** Math.floor(Math.log10(value));
  return Math.ceil(value / magnitude) * magnitude;
}

function formatDay(dateStr) {
  const date = new Date(`${dateStr}T00:00:00`);
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

// Stacked daily trend: calm vs. flagged (stressed or distortion) entries.
// Uses status tokens (good/critical) because the series *means* good/bad —
// per dataviz skill's collision rule, not categorical identity.
export default function TrendChart({ title, data, positiveKey, positiveLabel, negativeLabel }) {
  const [showTable, setShowTable] = useState(false);
  const [hoverIndex, setHoverIndex] = useState(null);

  const plotWidth = WIDTH - PADDING_LEFT - 8;
  const plotHeight = HEIGHT - PADDING_TOP - PADDING_BOTTOM;
  const maxTotal = niceMax(Math.max(1, ...data.map((d) => d.total)));
  const slotWidth = plotWidth / data.length;
  const barWidth = Math.min(BAR_MAX_THICKNESS, slotWidth * 0.55);

  const yTicks = [0, Math.round(maxTotal / 2), maxTotal];

  return (
    <div className="trend-chart card">
      <div className="trend-chart-header">
        <h3>{title}</h3>
        <button
          type="button"
          className="ghost-button"
          onClick={() => setShowTable((current) => !current)}
          aria-pressed={showTable}
        >
          {showTable ? <BarChart3 size={14} /> : <Table2 size={14} />}
          {showTable ? "View chart" : "View table"}
        </button>
      </div>

      <div className="trend-legend" role="list" aria-label="Legend">
        <span className="trend-legend-item" role="listitem">
          <span className="legend-swatch legend-good" /> {negativeLabel}
        </span>
        <span className="trend-legend-item" role="listitem">
          <span className="legend-swatch legend-critical" /> {positiveLabel}
        </span>
      </div>

      {showTable ? (
        <div className="trend-table-wrap">
          <table className="trend-table">
            <thead>
              <tr>
                <th scope="col">Date</th>
                <th scope="col">{negativeLabel}</th>
                <th scope="col">{positiveLabel}</th>
                <th scope="col">Total</th>
              </tr>
            </thead>
            <tbody>
              {data.map((day) => (
                <tr key={day.date}>
                  <td>{formatDay(day.date)}</td>
                  <td>{day.total - day[positiveKey]}</td>
                  <td>{day[positiveKey]}</td>
                  <td>{day.total}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="trend-svg-wrap">
          <svg
            viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
            role="img"
            aria-label={`${title}: stacked daily counts of ${negativeLabel.toLowerCase()} versus ${positiveLabel.toLowerCase()} diary entries over the last ${data.length} days`}
          >
            {yTicks.map((tick) => {
              const y = PADDING_TOP + plotHeight - (tick / maxTotal) * plotHeight;
              return (
                <g key={tick}>
                  <line
                    x1={PADDING_LEFT}
                    x2={WIDTH - 8}
                    y1={y}
                    y2={y}
                    className="gridline"
                  />
                  <text x={PADDING_LEFT - 8} y={y + 4} className="axis-label" textAnchor="end">
                    {tick}
                  </text>
                </g>
              );
            })}

            {data.map((day, index) => {
              const negative = day.total - day[positiveKey];
              const positive = day[positiveKey];
              const slotX = PADDING_LEFT + index * slotWidth + (slotWidth - barWidth) / 2;
              const baseY = PADDING_TOP + plotHeight;

              const negativeHeight = (negative / maxTotal) * plotHeight;
              const positiveHeight = (positive / maxTotal) * plotHeight;
              const hasNegative = negative > 0;
              const hasPositive = positive > 0;

              const negativeY = baseY - negativeHeight;
              const positiveY =
                negativeY - (hasNegative && hasPositive ? SEGMENT_GAP : 0) - positiveHeight;

              const isHovered = hoverIndex === index;

              return (
                <g
                  key={day.date}
                  tabIndex={0}
                  className="trend-bar-group"
                  onMouseEnter={() => setHoverIndex(index)}
                  onMouseLeave={() => setHoverIndex((current) => (current === index ? null : current))}
                  onFocus={() => setHoverIndex(index)}
                  onBlur={() => setHoverIndex((current) => (current === index ? null : current))}
                >
                  <rect
                    x={slotX - 6}
                    y={PADDING_TOP}
                    width={barWidth + 12}
                    height={plotHeight}
                    fill="transparent"
                  />
                  {hasNegative && (
                    <rect
                      x={slotX}
                      y={negativeY}
                      width={barWidth}
                      height={Math.max(negativeHeight, 0)}
                      rx={hasPositive ? 0 : 4}
                      ry={hasPositive ? 0 : 4}
                      className={`bar-good${isHovered ? " bar-hover" : ""}`}
                    />
                  )}
                  {hasPositive && (
                    <rect
                      x={slotX}
                      y={Math.max(positiveY, PADDING_TOP)}
                      width={barWidth}
                      height={Math.max(positiveHeight, 0)}
                      rx={4}
                      ry={4}
                      className={`bar-critical${isHovered ? " bar-hover" : ""}`}
                    />
                  )}
                  {index % 2 === 0 && (
                    <text
                      x={slotX + barWidth / 2}
                      y={HEIGHT - 6}
                      className="axis-label"
                      textAnchor="middle"
                    >
                      {formatDay(day.date)}
                    </text>
                  )}
                </g>
              );
            })}
          </svg>

          {hoverIndex !== null && (
            <div
              className="trend-tooltip"
              style={{ left: `${((hoverIndex + 0.5) / data.length) * 100}%` }}
            >
              <div className="trend-tooltip-date">{formatDay(data[hoverIndex].date)}</div>
              <div className="trend-tooltip-row">
                <span className="legend-swatch legend-critical" />
                <strong>{data[hoverIndex][positiveKey]}</strong> {positiveLabel.toLowerCase()}
              </div>
              <div className="trend-tooltip-row">
                <span className="legend-swatch legend-good" />
                <strong>{data[hoverIndex].total - data[hoverIndex][positiveKey]}</strong>{" "}
                {negativeLabel.toLowerCase()}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
