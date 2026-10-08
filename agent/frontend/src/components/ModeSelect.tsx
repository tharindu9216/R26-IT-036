interface Props {
  onSelect: (mode: "text" | "voice") => void;
}

export function ModeSelect({ onSelect }: Props) {
  return (
    <div className="card mode-select">
      <span className="eyebrow">You don’t have to carry it alone</span>
      <h1>A safe space to<br />check in with yourself.</h1>
      <p className="subtitle">
        Take a breath. Share what’s on your mind in whatever way feels
        easiest today.
      </p>
      <div className="mode-grid">
        <button className="mode-card" onClick={() => onSelect("text")}>
          <span className="mode-icon mode-icon-text" aria-hidden="true">
            <svg viewBox="0 0 24 24"><path d="M5 18.5 3.5 21l4.2-1.1c1.3.7 2.7 1.1 4.3 1.1 5 0 9-3.6 9-8s-4-8-9-8-9 3.6-9 8c0 2.1.8 4 2 5.5Z"/><path d="M8 11h8M8 15h5"/></svg>
          </span>
          <span className="mode-copy">
          <span className="mode-kicker">TYPE A MESSAGE</span>
          <span className="mode-title">Write how you feel</span>
          <span className="mode-desc">
            Put your thoughts into words, at your own pace.
          </span>
          </span>
          <span className="mode-arrow" aria-hidden="true">→</span>
        </button>
        <button className="mode-card" onClick={() => onSelect("voice")}>
          <span className="mode-icon mode-icon-voice" aria-hidden="true">
            <svg viewBox="0 0 24 24"><rect x="8" y="3" width="8" height="13" rx="4"/><path d="M5 11.5a7 7 0 0 0 14 0M12 18.5V22M8.5 22h7"/></svg>
          </span>
          <span className="mode-copy">
          <span className="mode-kicker">SPEAK FREELY</span>
          <span className="mode-title">Talk it through</span>
          <span className="mode-desc">
            Say what’s on your mind when typing feels like too much.
          </span>
          </span>
          <span className="mode-arrow" aria-hidden="true">→</span>
        </button>
      </div>
      <p className="privacy-note">
        <span aria-hidden="true">♡</span>
        A calm, judgment-free conversation, whenever you need it.
      </p>
    </div>
  );
}
