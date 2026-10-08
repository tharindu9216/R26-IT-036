interface HudRingProps {
  size?: number;
  /** 0-1. Omit for a purely decorative/idle ring. */
  progress?: number;
  variant?: "idle" | "progress" | "alert";
  children?: React.ReactNode;
}

const STROKE = 5;

export function HudRing({ size = 180, progress, variant = "idle", children }: HudRingProps) {
  const radius = size / 2 - STROKE * 2;
  const circumference = 2 * Math.PI * radius;
  const clamped = progress === undefined ? 0 : Math.min(1, Math.max(0, progress));
  const offset = circumference * (1 - clamped);
  const color = variant === "alert" ? "var(--peach)" : "var(--sage)";

  return (
    <div className={`hud-ring hud-ring--${variant}`} style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle
          className="hud-ring-track"
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          strokeWidth={STROKE}
        />
        <circle
          className="hud-ring-tick-ring"
          cx={size / 2}
          cy={size / 2}
          r={radius + STROKE * 1.8}
          fill="none"
          strokeWidth={1}
        />
        {progress !== undefined && (
          <circle
            className="hud-ring-progress"
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            strokeWidth={STROKE}
            stroke={color}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            transform={`rotate(-90 ${size / 2} ${size / 2})`}
          />
        )}
      </svg>
      <div className="hud-ring-content">{children}</div>
    </div>
  );
}
