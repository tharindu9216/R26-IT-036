import { useEffect, useState, type CSSProperties } from "react";
import { api } from "../api";
import type { ChatMessage, TraceStep, TurnTrace } from "../types";

interface Props {
  onClose: () => void;
}

/** One user message paired with the reply it produced, plus that turn's trace. */
interface Turn {
  index: number;
  question: string;
  answer: string;
  trace?: TurnTrace;
}

function toTurns(messages: ChatMessage[]): Turn[] {
  const turns: Turn[] = [];
  for (let i = 0; i < messages.length; i += 1) {
    if (messages[i].role !== "user") continue;
    const reply = messages[i + 1]?.role === "assistant" ? messages[i + 1] : undefined;
    turns.push({
      index: turns.length + 1,
      question: messages[i].text,
      answer: reply?.text ?? "",
      trace: reply?.trace,
    });
  }
  return turns;
}

const FRIENDLY_STAGES: Record<string, { title: string; description: string }> = {
  input: {
    title: "Your message",
    description: "The words or transcript that entered the wellbeing check.",
  },
  safety: {
    title: "Immediate safety check",
    description: "A first check for language that needs an immediate safety response.",
  },
  sensor: {
    title: "Body and check-in signals",
    description: "The saved questionnaire and wearable readings form the stress score used for this message.",
  },
  voice: {
    title: "How the voice sounded",
    description: "Tone-related cues such as arousal, dominance and valence are considered.",
  },
  c3: {
    title: "What the language suggested",
    description: "The message is checked for stress and unhelpful thinking patterns.",
  },
  fusion: {
    title: "Combined text signal",
    description: "The language findings are combined into one signal for response routing.",
  },
  emotion: {
    title: "How you may be feeling now",
    description: "The current-emotion model identifies the strongest emotion family.",
  },
  next: {
    title: "What may happen next",
    description: "The forecaster estimates whether emotional intensity may rise or remain lower.",
  },
  deviation: {
    title: "Change from the previous message",
    description: "The current emotion is compared with the previous turn when one exists.",
  },
  strategy: {
    title: "Response approach selected",
    description: "The available signals decide which supportive conversation style to use.",
  },
  generate: {
    title: "Reply created",
    description: "Qwen creates the response using the selected approach and recent context.",
  },
  escalation: {
    title: "Human-support check",
    description: "The system decides whether a verified support contact should be offered.",
  },
  output: {
    title: "Final response",
    description: "The completed response shown in the conversation.",
  },
};

function friendlyRoute(route: string): string {
  if (route === "supportive_adapter") return "Supportive conversation";
  if (route === "base") return "Everyday conversation";
  if (route === "crisis") return "Immediate safety response";
  return route || "Response selected";
}

function Step({ step, index }: { step: TraceStep; index: number }) {
  const skipped = step.status === "skipped";
  const visibleReasons = step.reasons?.filter(
    (reason) => reason.code !== "multimodal_stress",
  );
  const presentation = FRIENDLY_STAGES[step.key] ?? {
    title: step.component,
    description: step.title,
  };
  return (
    <li
      className={`flow-step${skipped ? " skipped" : ""}`}
      style={{ "--step-order": index } as CSSProperties}
    >
      <div className="flow-marker" aria-hidden="true">{index + 1}</div>
      <div className="flow-body">
        <div className="flow-heading">
          <div>
            <span className="flow-stage-label">Stage {index + 1}</span>
            <h4>{presentation.title}</h4>
          </div>
          <span className={`flow-status ${skipped ? "is-skipped" : "is-complete"}`}>
            {skipped ? "Not used" : "Complete"}
          </span>
        </div>
        <p className="flow-stage-description">{presentation.description}</p>
        {step.summary && <p className="flow-summary">{step.summary}</p>}
        {step.outputs && (
          <dl className="flow-outputs">
            {step.outputs.map((output) => (
              <div key={output.label}>
                <dt>{output.label}</dt>
                <dd>{output.value}</dd>
              </div>
            ))}
          </dl>
        )}
        {visibleReasons && visibleReasons.length > 0 && (
          <div className="flow-reason-box">
            <span>Why this decision was made</span>
            <ul className="flow-reasons">
              {visibleReasons.map((reason) => (
                <li key={reason.code}>{reason.text}</li>
              ))}
            </ul>
          </div>
        )}
        {step.detail && <p className="flow-detail">{step.detail}</p>}
        <details className="flow-technical">
          <summary>Technical record</summary>
          <p><strong>{step.component}</strong> — {step.title}</p>
        </details>
      </div>
    </li>
  );
}

/**
 * Mounted only while open (see App.tsx), so it fetches once per open rather
 * than keeping a copy of the conversation in sync. What it shows is whatever
 * the backend holds — the same turns it will replay into the next prompt.
 */
export function HistoryDrawer({ onClose }: Props) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [expanded, setExpanded] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .chatHistory()
      .then((messages) => {
        if (!cancelled) setTurns(toTurns(messages));
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <>
      <div className="drawer-scrim" onClick={onClose} />
      <aside
        className="drawer"
        role="dialog"
        aria-modal="true"
        aria-label="Conversation history"
      >
        <header className="drawer-header">
          <h2>History</h2>
          <button
            type="button"
            className="drawer-close"
            onClick={onClose}
            aria-label="Close history"
          >
            ✕
          </button>
        </header>

        <div className="drawer-body">
          {loading && <p className="muted">Loading…</p>}
          {error && <p className="drawer-error">{error}</p>}
          {!loading && !error && turns.length === 0 && (
            <p className="muted">No turns in this conversation yet.</p>
          )}

          {turns.map((turn) => {
            const isOpen = expanded === turn.index;
            return (
              <article className="drawer-turn" key={turn.index}>
                <button
                  type="button"
                  className="drawer-turn-head"
                  onClick={() => setExpanded(isOpen ? null : turn.index)}
                  aria-expanded={isOpen}
                >
                  <span className="drawer-turn-index">{turn.index}</span>
                  <span className="drawer-turn-text">{turn.question}</span>
                  {turn.trace && (
                    <span className="drawer-turn-strategy">
                      {friendlyRoute(turn.trace.route)}
                    </span>
                  )}
                  <span className="drawer-turn-caret">{isOpen ? "▾" : "▸"}</span>
                </button>

                {isOpen && (
                  <div className="drawer-turn-body">
                    {turn.trace ? (
                      <>
                        <section className="journey-intro">
                          <span className="journey-eyebrow">Saved decision journey</span>
                          <h3>How this reply was created</h3>
                          <p>
                            Follow the signals in the same order the system used them.
                            This is the saved record of the turn; opening it does not
                            run the models again.
                          </p>
                          <div className="journey-meta">
                            <span>{turn.trace.mode === "voice" ? "Voice check-in" : "Written check-in"}</span>
                            <span>{friendlyRoute(turn.trace.route)}</span>
                            <span>{turn.trace.steps.length} stages</span>
                          </div>
                        </section>

                        {turn.trace.version < 2 && (
                          <p className="journey-legacy-note">
                            This older turn keeps the weighting that was active when
                            its reply was created. New turns use 60% questionnaire and
                            40% wearable, with voice appraisal shown separately.
                          </p>
                        )}

                        <div className="journey-conversation">
                          <section className="journey-message is-user">
                            <span>Your message</span>
                            <p>{turn.question}</p>
                          </section>
                          <section className="journey-message is-reply">
                            <span>SentiVeraAI response</span>
                            <p>{turn.answer || "No response was stored for this turn."}</p>
                          </section>
                        </div>

                        <div className="journey-heading">
                          <div>
                            <span>Step-by-step explanation</span>
                            <h3>From your message to the reply</h3>
                          </div>
                          <small>{turn.trace.steps.length} recorded stages</small>
                        </div>
                        <ol className="flow">
                          {turn.trace.steps.map((step, index) => (
                            <Step key={step.key} step={step} index={index} />
                          ))}
                        </ol>
                        <p className="journey-note">
                          Predictions describe model estimates, not diagnoses. The
                          history records what influenced routing; it does not claim
                          that every estimate is certain.
                        </p>
                      </>
                    ) : (
                      <>
                        <p className="muted">
                          No flow was recorded for this turn.
                        </p>
                        <p className="drawer-answer">{turn.answer}</p>
                      </>
                    )}
                  </div>
                )}
              </article>
            );
          })}
        </div>
      </aside>
    </>
  );
}
