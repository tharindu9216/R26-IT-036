import { useEffect, useState, type CSSProperties } from "react";
import type { ModelLoadState } from "../types";

interface ModelStatusEntry {
  status: ModelLoadState;
  error: string | null;
}

interface PreparationStatus<Key extends string> {
  ready: boolean;
  models: Record<Key, ModelStatusEntry>;
}

export interface PreparationDetail<Key extends string> {
  key: Key;
  name: string;
  description: string;
}

interface Props<Key extends string> {
  details: readonly PreparationDetail<Key>[];
  initialStatus: PreparationStatus<Key>;
  getStatus: () => Promise<PreparationStatus<Key>>;
  prepare: () => Promise<PreparationStatus<Key>>;
  onReady: () => void;
  eyebrow: string;
  title: string;
  description: string;
  note: string;
  variant: "text" | "voice";
  fallbackError: string;
}

function StatusIcon({ state }: { state: ModelLoadState }) {
  if (state === "ready") return <span className="preparation-check" aria-hidden="true">✓</span>;
  if (state === "error") return <span className="preparation-error-icon" aria-hidden="true">!</span>;
  return <span className={`preparation-spinner${state === "pending" ? " is-pending" : ""}`} aria-hidden="true" />;
}

function CoreIcon({ variant }: { variant: "text" | "voice" }) {
  if (variant === "voice") {
    return (
      <svg viewBox="0 0 32 32">
        <rect x="11" y="5" width="10" height="16" rx="5" />
        <path d="M7.5 16.5a8.5 8.5 0 0 0 17 0M16 25v3.5M11.5 28.5h9" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 32 32">
      <path d="M16 26c-6-3.5-10-7.3-10-11.8A5.7 5.7 0 0 1 16 10.5a5.7 5.7 0 0 1 10 3.7C26 18.7 22 22.5 16 26Z" />
      <path d="M16 11c-1.5 2.9-1.5 6 0 9.2m0-4.6c-2.5-1.5-4.4-1.3-5.8.3m5.8 2.3c2.1-1.3 4-1.1 5.4.4" />
    </svg>
  );
}

export function ModelPreparation<Key extends string>({
  details,
  initialStatus,
  getStatus,
  prepare,
  onReady,
  eyebrow,
  title,
  description,
  note,
  variant,
  fallbackError,
}: Props<Key>) {
  const [status, setStatus] = useState(initialStatus);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [visibleStage, setVisibleStage] = useState(0);

  useEffect(() => {
    let cancelled = false;
    let pollTimer: number | undefined;

    async function poll() {
      try {
        const next = await getStatus();
        if (cancelled) return;
        setStatus(next);
        if (next.ready) return;
      } catch {
        // The preparation request below owns the user-facing error message.
      }
      if (!cancelled) pollTimer = window.setTimeout(poll, 500);
    }

    void poll();
    prepare().then((next) => {
      if (!cancelled) setStatus(next);
    }).catch((reason: unknown) => {
      if (!cancelled) setError(reason instanceof Error ? reason.message : fallbackError);
    });

    return () => {
      cancelled = true;
      if (pollTimer !== undefined) window.clearTimeout(pollTimer);
    };
  }, [attempt, fallbackError, getStatus, prepare]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      setVisibleStage((current) => {
        if (current >= details.length) {
          window.clearInterval(timer);
          return current;
        }
        return current + 1;
      });
    }, 800);
    return () => window.clearInterval(timer);
  }, [attempt, details.length]);

  useEffect(() => {
    if (!status.ready || visibleStage < details.length) return;
    const timer = window.setTimeout(onReady, 600);
    return () => window.clearTimeout(timer);
  }, [details.length, onReady, status.ready, visibleStage]);

  const actualReadyCount = details.filter(({ key }) => status.models[key].status === "ready").length;
  const readyCount = status.ready ? visibleStage : Math.min(actualReadyCount, visibleStage);
  const progress = readyCount / details.length;

  return (
    <div className={`card model-preparation-card is-${variant}`} aria-live="polite">
      <div className="preparation-visual" aria-hidden="true">
        <div className="preparation-orbit"><i /><i /><i /><i /></div>
        <div className="preparation-core"><CoreIcon variant={variant} /></div>
      </div>

      <div className="preparation-copy">
        <span className="eyebrow">{eyebrow}</span>
        <h2>{title}</h2>
        <p className="muted">{description}</p>
      </div>

      <div className="preparation-overall-progress" aria-hidden="true">
        <span style={{ width: `${progress * 100}%` }} />
      </div>

      <div className="preparation-models">
        {details.map(({ key, name, description: itemDescription }, index) => {
          const model = status.models[key];
          const displayState: ModelLoadState = model.status === "error"
            ? "error"
            : index < visibleStage && model.status === "ready"
              ? "ready"
              : index === visibleStage || model.status === "loading"
                ? "loading"
                : "pending";
          return (
            <div
              className={`preparation-model is-${displayState}`}
              key={key}
              style={{ "--model-delay": `${index * 90}ms` } as CSSProperties}
            >
              <StatusIcon state={displayState} />
              <span className="preparation-model-copy">
                <strong>{name}</strong>
                <small>{model.error ?? itemDescription}</small>
              </span>
              <span className="preparation-state">
                {displayState === "ready" ? "Ready" : displayState === "loading" ? "Preparing" : displayState === "error" ? "Error" : "Waiting"}
              </span>
            </div>
          );
        })}
      </div>

      {error && (
        <div className="preparation-retry">
          <p className="error">{error}</p>
          <button
            className="primary"
            type="button"
            onClick={() => {
              setStatus(initialStatus);
              setError(null);
              setVisibleStage(0);
              setAttempt((value) => value + 1);
            }}
          >
            Try again
          </button>
        </div>
      )}
      {!error && <p className="preparation-note">{note}</p>}
    </div>
  );
}
