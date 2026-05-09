# C4 Emotion Forecasting and Supportive Dialogue Demo

This folder contains a runnable Streamlit prototype for Component C4. It demonstrates:

- Current emotion classification (with trained model or fallback)
- Next emotion forecasting (with trained model or fallback)
- Emotion deviation tracking between turns
- Strategy selection
- Template-based supportive response generation
- Full JSON trace for supervisor review

## Project Structure

```text
c4_emotion_support/
├── streamlit_app.py
├── requirements.txt
├── README.md
├── config.py
├── c4_pipeline/
│   ├── __init__.py
│   ├── emotion_classifier.py
│   ├── emotion_forecaster.py
│   ├── deviation_tracker.py
│   ├── strategy_selector.py
│   ├── response_generator.py
│   ├── safety.py
│   └── pipeline.py
├── models/
│   ├── current_emotion_classifier/
│   │   └── README_PLACE_MODEL_HERE.md
│   └── next_emotion_forecaster/
│       └── README_PLACE_MODEL_HERE.md
└── sample_outputs/
    └── sample_conversation_trace.json
```

## Setup

1. Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

2. Place trained models:

- Current emotion classifier in `models/current_emotion_classifier/`
- Next emotion forecaster in `models/next_emotion_forecaster/`

If models are missing, the app will use rule-based fallbacks and show a warning.

## Run the Demo

```bash
streamlit run streamlit_app.py
```

## Outputs Shown

- Current emotion and confidence
- Previous emotion, deviation score, and deviation level
- Forecasted next emotion and confidence
- Selected strategy and supportive response
- JSON trace for each turn
- Emotion history table and confidence chart

## Model Performance Note

Current emotion classifier results (reported):

- F1 = 0.8085
- Accuracy = 81.5%

## Research Disclaimer

This is an academic prototype, not a clinical or therapy system. For serious mental health concerns, users should seek support from qualified professionals.
