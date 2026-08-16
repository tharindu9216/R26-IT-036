"""Launcher for the C4 demo.

The app itself lives next to the pipeline it drives, in
`ai_components/c4_emotion_support/streamlit_app.py`. This file used to be a full
copy of it, which meant every change had to be made twice and the two drifted.
It is now a thin shim: it puts the component on `sys.path` and executes the real
app, so there is exactly one implementation.

    streamlit run streamlit_app/c4_emotion_support/streamlit_app.py
"""

import runpy
import sys
from pathlib import Path

COMPONENT_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "ai_components"
    / "c4_emotion_support"
)

if str(COMPONENT_DIR) not in sys.path:
    sys.path.insert(0, str(COMPONENT_DIR))

runpy.run_path(str(COMPONENT_DIR / "streamlit_app.py"), run_name="__main__")
