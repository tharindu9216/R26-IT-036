# C4 demo launcher

`streamlit_app.py` here is a **thin launcher**, not a second copy of the app.
It adds `ai_components/c4_emotion_support/` to `sys.path` and executes the real
app from there, so the UI exists in exactly one place.

## Run

From the repository root:

```bash
pip install -r ai_components/c4_emotion_support/requirements.txt
streamlit run streamlit_app/c4_emotion_support/streamlit_app.py
```

Equivalently, straight from the component (identical result):

```bash
cd ai_components/c4_emotion_support
streamlit run streamlit_app.py
```

## Where everything lives

```text
R26-IT-036/
├── streamlit_app/c4_emotion_support/
│   ├── streamlit_app.py        <- this launcher
│   └── README.md
└── ai_components/c4_emotion_support/
    ├── streamlit_app.py        <- the actual UI
    ├── smoke_test.py           <- headless end-to-end check
    ├── c4_pipeline/            <- classifier, forecaster, XAI, rules
    ├── config.py
    ├── models/                 <- trained checkpoints (gitignored)
    └── README.md               <- component status, metrics, XAI notes
```

Read `ai_components/c4_emotion_support/README.md` for what the component does,
which models are wired in, their measured scores, and how the explanations work.

## Sidebar controls

- **Forecaster checkpoint** — `textcnn` (default), `bilstm`, or `distilbert`
- **Force rule-based forecaster** — bypass the trained model and use the
  persistence heuristic
- **Compute explanations (XAI)** — Integrated Gradients costs ~64 forward passes
  per model, so it can be switched off for a faster chat
- **Show JSON trace** — the full per-turn trace
- **Clear conversation** — reset history

## Notes

- Academic research prototype; not a replacement for professional mental health
  support.
- If the model files are absent the app still runs on fallbacks, and says so in
  the sidebar.
