# C4 Emotion Forecasting & Supportive Dialogue Demo

This Streamlit application demonstrates Component C4 of the multimodal stress and emotion analysis system.

## File Structure

After moving, the structure is now:

```
R26-IT-036/
├── streamlit_app/
│   ├── c3_text_stressor_distortion/
│   │   └── stressor head/
│   │       └── app.py
│   └── c4_emotion_support/
│       ├── streamlit_app.py          (this file's folder)
│       └── README.md
└── ai_components/
    └── c4_emotion_support/
        ├── c4_pipeline/              (referenced by streamlit_app.py)
        ├── config.py                 (referenced by streamlit_app.py)
        ├── models/
        ├── requirements.txt
        └── sample_outputs/
```

## How to Run

### 1. Install Dependencies

```bash
cd ai_components/c4_emotion_support
pip install -r requirements.txt
```

### 2. Run the Streamlit App

From the project root directory:

```bash
streamlit run streamlit_app/c4_emotion_support/streamlit_app.py
```

Or directly:

```bash
cd streamlit_app/c4_emotion_support
streamlit run streamlit_app.py
```

## Key Changes Made

- **Import Path Resolution**: The `streamlit_app.py` now uses `sys.path` to dynamically add `ai_components/c4_emotion_support` to Python's module search path.
- **Location**: The app has been moved from `ai_components/c4_emotion_support/streamlit_app.py` to `streamlit_app/c4_emotion_support/streamlit_app.py`
- **Dependencies**: The core C4 pipeline code remains in `ai_components/c4_emotion_support/`

## Features

- **Emotion Classification**: Detects current emotion from user input
- **Emotion Forecasting**: Predicts the user's next emotional state
- **Strategy Selection**: Chooses appropriate support strategy
- **Supportive Response**: Generates empathetic dialogue responses
- **Conversation Tracking**: Maintains emotion history and visualizes emotion trajectories
- **Debug Trace**: Optional detailed pipeline output for development/testing

## Settings (Sidebar)

- **Show debug trace**: Toggle to view detailed pipeline execution data
- **Force fallback forecaster**: Toggle to use fallback models if trained models are unavailable
- **Clear conversation**: Reset the conversation history

## Notes

- This is an academic research prototype
- Not a replacement for professional mental health support
- Fallback models are used if trained models cannot be loaded
