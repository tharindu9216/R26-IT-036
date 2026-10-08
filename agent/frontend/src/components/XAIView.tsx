import type { CSSProperties } from "react";
import type { XAIAttribution, XAIResponse, XAISection } from "../types";

interface Props {
  result: XAIResponse;
}

interface ExplainedComponent {
  key: string;
  owner: string;
  title: string;
  description: string;
  section: XAISection;
  features?: boolean;
  displayTarget?: string;
}

function scoreColor(score: number): string {
  const alpha = Math.min(0.72, 0.1 + Math.abs(score) * 0.85);
  return score >= 0
    ? `rgba(101, 133, 120, ${alpha})`
    : `rgba(231, 170, 137, ${alpha})`;
}

function friendlyLabel(label: string): string {
  const known: Record<string, string> = {
    arousal: "Energy in your voice",
    dominance: "Sense of control",
    valence: "Emotional tone",
    bvp: "Pulse pattern",
    eda: "Skin response",
    hr: "Heart rate",
  };
  const normalized = label.trim().toLowerCase();
  if (known[normalized]) return known[normalized];
  return label
    .replace(/bvp/gi, "pulse")
    .replace(/eda/gi, "skin response")
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function friendlyTarget(target: string): string {
  return target.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function cleanToken(token: string): string | null {
  const special = new Set([
    "<s>", "</s>", "<pad>", "<unk>", "[cls]", "[sep]", "[pad]", "[unk]",
  ]);
  if (special.has(token.trim().toLowerCase())) return null;
  const cleaned = token.replace(/^(?:Ġ|▁|##)+/, "").trim();
  return cleaned || null;
}

function influenceStrength(score: number): string {
  const magnitude = Math.abs(score);
  const direction = score >= 0 ? "supports" : "reduces";
  if (magnitude >= 0.55) return `strongly ${direction}`;
  if (magnitude >= 0.2) return direction;
  return `slightly ${direction}`;
}

function coverageLabel(status: string): string {
  const labels: Record<string, string> = {
    explained: "XAI explained",
    recorded: "Saved in history",
    not_used: "Not in this path",
    boundary: "No exact XAI",
    unavailable: "No signal available",
    output_only: "Output only",
  };
  return labels[status] ?? status;
}

function TokenAttributions({ items }: { items: XAIAttribution[] }) {
  const visible = items.flatMap((item) => {
    const label = cleanToken(item.label);
    return label ? [{ ...item, label }] : [];
  });
  return (
    <div className="xai-tokens" aria-label="Words that influenced this result">
      {visible.map((item, index) => (
        <span
          className="xai-token"
          key={`${item.label}-${index}`}
          style={{ backgroundColor: scoreColor(item.score) }}
          title={influenceStrength(item.score)}
        >
          {item.label}
        </span>
      ))}
    </div>
  );
}

function FeatureAttributions({ items }: { items: XAIAttribution[] }) {
  const ordered = [...items]
    .sort((left, right) => Math.abs(right.score) - Math.abs(left.score))
    .slice(0, 8);
  const maximum = Math.max(...ordered.map((item) => Math.abs(item.score)), 1e-9);
  return (
    <div className="xai-features">
      {ordered.map((item) => {
        const style = {
          "--bar-width": `${(Math.abs(item.score) / maximum) * 100}%`,
          "--bar-color": scoreColor(item.score),
        } as CSSProperties;
        return (
          <div className="xai-feature" key={item.label} style={style}>
            <span>{friendlyLabel(item.label)}</span>
            <small>{influenceStrength(item.score)}</small>
          </div>
        );
      })}
    </div>
  );
}

function strongestWithin(section: XAISection): string[] {
  return [...section.attributions]
    .filter((item) => item.score > 0)
    .sort((left, right) => right.score - left.score)
    .flatMap((item) => {
      const token = cleanToken(item.label);
      return token ? [friendlyLabel(token)] : [];
    })
    .slice(0, 3);
}

function Section({
  item,
  index,
}: {
  item: ExplainedComponent;
  index: number;
}) {
  const strongest = strongestWithin(item.section);
  const shownTarget = item.displayTarget ?? item.section.target;
  return (
    <section className="xai-section">
      <div className="xai-component-owner">
        <span>{index + 1}</span>
        <strong>{item.owner}</strong>
      </div>
      <div className="xai-section-heading">
        <div>
          <h4>{item.title}</h4>
          <p>{item.description}</p>
        </div>
        <div className="xai-result">
          <small>{item.displayTarget ? "Combined result" : "Model result"}</small>
          <strong>{friendlyTarget(shownTarget)}</strong>
        </div>
      </div>
      {strongest.length > 0 && (
        <p className="xai-plain-reading">
          <strong>Main evidence:</strong> {strongest.join(", ")}
        </p>
      )}
      <span className="xai-attribution-label">
        {item.features ? "Signals and their influence" : "Words and their influence"}
      </span>
      {item.features ? (
        <FeatureAttributions items={item.section.attributions} />
      ) : (
        <TokenAttributions items={item.section.attributions} />
      )}
      <details className="xai-technical">
        <summary>Model and XAI method</summary>
        <p>
          {item.section.method}
          {item.section.member ? ` · ${item.section.member}` : ""}
          {item.displayTarget && item.displayTarget !== item.section.target
            ? ` · wearable-only result: ${friendlyTarget(item.section.target)}`
            : ""}
        </p>
      </details>
    </section>
  );
}

function strongestSignals(result: XAIResponse): string[] {
  const sections = [
    result.sensor,
    result.questionnaire,
    result.combined,
    result.stress,
    result.cbt,
    result.voice,
    result.current_emotion,
    result.next_emotion,
  ].filter((section): section is XAISection => section !== null);
  const scores = new Map<string, number>();
  for (const item of sections.flatMap((section) => section.attributions)) {
    if (item.score <= 0 || item.label.trim().length < 2) continue;
    const cleaned = cleanToken(item.label);
    if (!cleaned) continue;
    const label = friendlyLabel(cleaned);
    scores.set(label, Math.max(scores.get(label) ?? 0, item.score));
  }
  return [...scores]
    .sort((left, right) => right[1] - left[1])
    .slice(0, 5)
    .map(([label]) => label);
}

export function XAIView({ result }: Props) {
  const errors = Object.entries(result.errors);
  const strongest = strongestSignals(result);
  const components: ExplainedComponent[] = [];
  if (result.questionnaire) components.push({
    key: "questionnaire", owner: "Personal check-in", title: "Your check-in answers",
    description: "How each answer influenced the self-reported stress result.",
    section: result.questionnaire, features: true,
  });
  if (result.sensor) components.push({
    key: "sensor", owner: "C1 · Wearable model", title: "Body-signal patterns",
    description: "Pulse and skin-response features that influenced the wearable result.",
    section: result.sensor, features: true,
    displayTarget: result.combined?.target,
  });
  if (result.voice) components.push({
    key: "voice", owner: "C2 · Voice appraisal", title: "Voice-related cues",
    description: "Energy, emotional tone and sense of control found in the recording.",
    section: result.voice, features: true,
  });
  if (result.combined) components.push({
    key: "combined", owner: "Wellbeing fusion", title: "Combined wellbeing result",
    description: "How the questionnaire (60%) and wearable result (40%) were weighted together.",
    section: result.combined, features: true,
  });
  if (result.stress) components.push({
    key: "stress", owner: "C3 · Stress model", title: "Stress-related language",
    description: "Words that moved the stress prediction toward or away from its result.",
    section: result.stress,
  });
  if (result.cbt) components.push({
    key: "cbt", owner: "C3 · Thinking-pattern model", title: "Thinking patterns",
    description: "Words that influenced the unhelpful-thinking-pattern prediction.",
    section: result.cbt,
  });
  if (result.current_emotion) components.push({
    key: "current-emotion", owner: "Emotion chain · Current", title: "Current emotional tone",
    description: "Words that most influenced the detected emotion family.",
    section: result.current_emotion,
  });
  if (result.next_emotion) components.push({
    key: "next-emotion", owner: "Emotion chain · Forecast", title: "Next-turn intensity outlook",
    description: "Words that influenced whether emotional intensity may rise or remain lower.",
    section: result.next_emotion,
  });

  return (
    <div className="xai-panel">
      <div className="xai-overview">
        <span className="xai-overview-icon" aria-hidden="true">✦</span>
        <div>
          <span className="xai-overview-kicker">Explainable AI</span>
          <h4>Why the models reached these results</h4>
          <p>Each card belongs to one component and shows the evidence that influenced its prediction.</p>
        </div>
      </div>

      <div className="xai-component-map" aria-label="Components explained">
        <span className="xai-component-map-title">Components explained</span>
        <div>
          {components.map((item, index) => (
            <span key={item.key}>
              <b>{index + 1}</b>{item.owner}
            </span>
          ))}
        </div>
      </div>

      {result.coverage && result.coverage.length > 0 && (
        <section className="xai-coverage">
          <div className="xai-coverage-heading">
            <div>
              <span>Explanation coverage</span>
              <h5>What is—and is not—explained in this {result.mode ?? "check-in"} path</h5>
            </div>
            <strong>
              {result.coverage.filter((item) => item.status === "explained").length}
              /{result.coverage.length} explained
            </strong>
          </div>
          <div className="xai-coverage-list">
            {result.coverage.map((item) => (
              <details key={item.component}>
                <summary>
                  <span>{item.component}</span>
                  <b className={`coverage-${item.status}`}>{coverageLabel(item.status)}</b>
                </summary>
                <p>{item.detail}</p>
              </details>
            ))}
          </div>
          {result.resource_note && (
            <p className="xai-resource-note">
              <strong>Resource use:</strong> {result.resource_note}
            </p>
          )}
        </section>
      )}

      {strongest.length > 0 && (
        <div className="xai-strongest">
          <span>Strongest supporting evidence across the check-in</span>
          <div className="xai-signal-list" aria-label="Strongest signals">
            {strongest.map((signal) => <span key={signal}>{signal}</span>)}
          </div>
        </div>
      )}

      <div className="xai-legend">
        <strong>How to read the colours</strong>
        <span><i className="legend-positive" /> Green supports that component's result</span>
        <span><i className="legend-negative" /> Peach reduces that component's result</span>
      </div>

      <div className="xai-component-sections">
        {components.map((item, index) => (
          <Section key={item.key} item={item} index={index} />
        ))}
      </div>

      {errors.length > 0 && (
        <details className="xai-partial-errors">
          <summary>Some signals were unavailable</summary>
          {errors.map(([component, detail]) => <p key={component}>{component}: {detail}</p>)}
        </details>
      )}
      <p className="xai-note">
        This explains each wellbeing model's prediction and the signals used for response routing. It does not explain the exact words chosen by Qwen, and these estimates are not clinical diagnoses.
      </p>
    </div>
  );
}
