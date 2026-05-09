from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

CURRENT_EMOTION_MODEL_PATH = BASE_DIR / "models" / "current_emotion_classifier"
NEXT_EMOTION_MODEL_PATH = BASE_DIR / "models" / "next_emotion_forecaster"

EMOTION_LABELS = [
    "neutral",
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust",
]

POSITIVE = {"joy"}
NEGATIVE = {"sadness", "anger", "fear", "disgust"}
NEUTRAL = {"neutral"}
OTHER = {"surprise"}

SAME_EMOTION = 0.0
LOW_DEVIATION = 0.25
MODERATE_DEVIATION = 0.60
HIGH_DEVIATION = 0.90
