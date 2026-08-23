const FALLBACKS = {
  no_signal: {
    state: "no_signal",
    label: "No combined signal",
    summary: "Neither component head detected its corresponding signal.",
  },
  stress_only: {
    state: "stress_only",
    label: "Stress signal only",
    summary: "A stress signal was detected without a thinking-pattern signal.",
  },
  distortion_only: {
    state: "distortion_only",
    label: "Thinking-pattern signal only",
    summary: "A thinking-pattern signal was detected without a stress signal.",
  },
  stress_with_distortion: {
    state: "stress_with_distortion",
    label: "Stress with thinking-pattern signal",
    summary: "Both component heads detected their corresponding signals.",
  },
};

export function fusionForEntry(entry) {
  if (entry?.fusion) return entry.fusion;
  const stressed = Boolean(entry?.stress?.is_stressed);
  const distorted = Boolean(entry?.distortion?.has_distortion);
  if (stressed && distorted) return FALLBACKS.stress_with_distortion;
  if (stressed) return FALLBACKS.stress_only;
  if (distorted) return FALLBACKS.distortion_only;
  return FALLBACKS.no_signal;
}

export function fusionTone(state) {
  if (state === "stress_with_distortion") return "combined";
  if (state === "stress_only") return "stress";
  if (state === "distortion_only") return "pattern";
  return "quiet";
}
