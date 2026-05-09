from pathlib import Path
from typing import Dict, List, Optional

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

from config import EMOTION_LABELS, NEGATIVE, NEUTRAL, POSITIVE


class EmotionForecaster:
    def __init__(
        self,
        model_path: Path,
        labels: Optional[List[str]] = None,
        force_fallback: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.labels = labels or list(EMOTION_LABELS)
        self.force_fallback = force_fallback
        self.model = None
        self.model_type = "rule"
        self.fallback = True
        self._try_load_model()

    def _try_load_model(self) -> None:
        if self.force_fallback or joblib is None:
            return
        candidates = [
            self.model_path / "forecaster.joblib",
            self.model_path / "forecaster.pkl",
        ]
        for candidate in candidates:
            if candidate.exists():
                try:
                    self.model = joblib.load(candidate)
                    self.model_type = "joblib"
                    self.fallback = False
                    return
                except Exception:
                    self.model = None
                    self.fallback = True

    def _build_features(
        self,
        current_emotion: str,
        previous_emotion: Optional[str],
        deviation_level: str,
    ) -> List[float]:
        deviation_map = {"None": 0.0, "Low": 0.25, "Moderate": 0.6, "High": 0.9}
        features = []
        for label in self.labels:
            features.append(1.0 if label == current_emotion else 0.0)
        for label in self.labels:
            features.append(1.0 if label == previous_emotion else 0.0)
        features.append(deviation_map.get(deviation_level, 0.0))
        return features

    def _rule_based_forecast(
        self, current_emotion: str, history: List[str], deviation_level: str
    ) -> Dict[str, object]:
        if current_emotion in POSITIVE:
            label = "joy"
            confidence = 0.7
        elif current_emotion in NEUTRAL:
            label = "neutral" if not history else history[-1]
            confidence = 0.6
        elif current_emotion in NEGATIVE:
            if deviation_level == "High":
                label = "fear"
                confidence = 0.65
            else:
                label = current_emotion
                confidence = 0.7
        else:
            label = "neutral"
            confidence = 0.55

        probabilities = self._build_probabilities(label, confidence)
        return {"label": label, "confidence": confidence, "probabilities": probabilities}

    def _build_probabilities(self, label: str, confidence: float) -> Dict[str, float]:
        remaining = max(0.0, 1.0 - confidence)
        other_labels = [item for item in self.labels if item != label]
        if not other_labels:
            return {label: 1.0}
        share = remaining / len(other_labels)
        probs = {item: share for item in other_labels}
        probs[label] = confidence
        return probs

    def predict(
        self,
        current_emotion: str,
        history: List[str],
        deviation_level: str,
        previous_emotion: Optional[str] = None,
    ) -> Dict[str, object]:
        if self.fallback or self.model is None:
            return self._rule_based_forecast(current_emotion, history, deviation_level)

        features = self._build_features(current_emotion, previous_emotion, deviation_level)
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba([features])[0]
            label_index = int(probabilities.argmax())
            label = self.labels[label_index]
            confidence = float(probabilities[label_index])
            prob_dict = {
                label_name: float(prob)
                for label_name, prob in zip(self.labels, probabilities)
            }
            return {"label": label, "confidence": confidence, "probabilities": prob_dict}

        if hasattr(self.model, "predict"):
            prediction = self.model.predict([features])[0]
            label = str(prediction)
            return {
                "label": label,
                "confidence": 0.55,
                "probabilities": self._build_probabilities(label, 0.55),
            }

        return self._rule_based_forecast(current_emotion, history, deviation_level)

    def status(self) -> Dict[str, str]:
        return {
            "available": str(not self.fallback),
            "fallback": str(self.fallback),
            "model_path": str(self.model_path),
            "model_type": self.model_type,
        }
