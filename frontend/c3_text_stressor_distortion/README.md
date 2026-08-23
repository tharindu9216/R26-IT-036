# c3 — Daily Diary

React + Vite frontend for the daily-diary component. Every entry is scored by two
3-transformer weighted ensembles (BERT + MentalBERT + DeBERTa-v3) served by the
backend:

- **Stress** — Stress header ensemble (equal weights, threshold 0.5).
- **CBT distortion** — CBT header ensemble (OOF-selected weights, threshold 0.37).

Two views:

- **Diary** — write and save entries; each save shows stress + distortion badges.
- **Insights** — entry counts and 14-day trend charts for both signals.

Each entry also receives a transparent four-state decision-level fusion result:

- No combined signal
- Stress signal only
- Thinking-pattern signal only
- Stress with thinking-pattern signal

The timeline detail view can generate an on-demand hierarchical explanation:
representative SHAP + LIME + Integrated Gradients consensus evidence from each
component head plus the plain-language rule trace used to select the combined
state.

## Run

```bash
npm install
npm run dev
```

Set `VITE_API_BASE` if the backend is not running on `http://localhost:8005`.
