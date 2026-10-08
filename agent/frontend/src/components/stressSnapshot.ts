import type { ChatMessage } from "../types";

export interface StressSnapshot {
  stressed: boolean;
  probability: number;
}

/** Recover the latest saved turn score so a page reload keeps the header and
 * history aligned. Older traces used a different output label, so both are
 * accepted without rewriting what happened in that turn. */
export function latestStressSnapshot(messages: ChatMessage[]): StressSnapshot | null {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const trace = messages[index].trace;
    const step = trace?.steps.find((candidate) => candidate.key === "sensor");
    const output = step?.outputs?.find((candidate) =>
      candidate.label === "combined stress score" ||
      candidate.label === "combined probability"
    );
    if (!step?.summary || !output) continue;
    const percentage = Number.parseFloat(output.value);
    if (!Number.isFinite(percentage)) continue;
    const summary = step.summary.toLowerCase();
    return {
      stressed: summary.startsWith("stress"),
      probability: percentage / 100,
    };
  }
  return null;
}
