"""
LIME explainability for the Stress Header classifier.

LIME learns a sparse local surrogate around one text by repeatedly masking
words and observing how the stress classifier's probabilities change.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from decision import select_predicted_class
from rationale_extractor import extract_rationale_spans, highlight_rationales


TOKEN_PATTERN = re.compile(r"\b\w+(?:['’]\w+)*\b|[^\w\s]", re.UNICODE)


class StressLIME:
    """Word-level LIME explainer for the binary stress classification head."""

    def __init__(
        self,
        model_or_pipeline,
        tokenizer=None,
        label_names: list[str] | dict | None = None,
        device: str | None = None,
        max_length: int = 192,
        random_state: int = 42,
        prediction_batch_size: int = 8,
        decision_threshold: float | None = None,
    ):
        self.model_or_pipeline = model_or_pipeline
        self.tokenizer = tokenizer or getattr(model_or_pipeline, "tokenizer", None)
        if self.tokenizer is None:
            raise ValueError("A tokenizer is required for LIME perturbations.")

        self.label_names = self._normalise_label_names(label_names)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.prediction_batch_size = max(1, prediction_batch_size)
        self.decision_threshold = decision_threshold
        self.is_pipeline = callable(model_or_pipeline) and hasattr(
            model_or_pipeline, "tokenizer"
        )

        if not self.is_pipeline:
            self.model_or_pipeline = model_or_pipeline.to(self.device).eval()

        # Import lazily so the rest of the XAI panel can still load and report a
        # helpful dependency error when LIME has not been installed yet.
        try:
            from lime.lime_text import LimeTextExplainer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "LIME is not installed. Install the project's requirements.",
                name="lime",
            ) from exc

        self.explainer = LimeTextExplainer(
            class_names=self.label_names,
            bow=True,
            random_state=random_state,
        )

    @staticmethod
    def _normalise_label_names(label_names: list[str] | dict | None) -> list[str]:
        if label_names is None:
            return ["Not Stressed", "Stressed"]
        if isinstance(label_names, dict):
            return [
                label_names[str(index)]
                if str(index) in label_names
                else label_names[index]
                for index in range(len(label_names))
            ]
        return list(label_names)

    def _pipeline_results_to_probs(self, results) -> np.ndarray:
        if results and isinstance(results[0], dict):
            results = [results]

        rows = []
        for row in results:
            by_label = {item["label"]: item["score"] for item in row}
            rows.append(
                [
                    by_label.get(
                        label,
                        by_label.get(str(index), by_label.get(f"LABEL_{index}", 0.0)),
                    )
                    for index, label in enumerate(self.label_names)
                ]
            )
        return np.asarray(rows, dtype=float)

    def _predict(self, texts) -> np.ndarray:
        """Return class probabilities in small batches to avoid GPU OOMs."""
        text_list = [str(text) for text in texts]
        probability_batches = []

        for start in range(0, len(text_list), self.prediction_batch_size):
            batch = text_list[start : start + self.prediction_batch_size]
            if self.is_pipeline:
                results = self.model_or_pipeline(batch, return_all_scores=True)
                probability_batches.append(self._pipeline_results_to_probs(results))
                continue

            encoded = self.tokenizer(
                batch,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            with torch.no_grad():
                stress_logits, _ = self.model_or_pipeline(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                probability_batches.append(
                    torch.softmax(stress_logits, dim=1).detach().cpu().numpy()
                )

        if not probability_batches:
            return np.empty((0, len(self.label_names)), dtype=float)
        return np.concatenate(probability_batches, axis=0)

    def explain(
        self,
        text: str,
        target_class: int | None = None,
        num_samples: int = 1000,
        num_features: int | None = None,
    ) -> dict:
        """Explain one text with a locally fitted, word-level surrogate model."""
        if not text or not text.strip():
            raise ValueError("LIME requires non-empty text.")
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2.")

        probabilities = self._predict([text])[0]
        predicted_class = select_predicted_class(
            probabilities, self.decision_threshold)
        explained_class = predicted_class if target_class is None else target_class
        if explained_class < 0 or explained_class >= len(self.label_names):
            raise ValueError(f"Invalid target class: {explained_class}")

        word_tokens = TOKEN_PATTERN.findall(text)
        unique_words = {token for token in word_tokens if re.search(r"\w", token)}
        if not unique_words:
            raise ValueError("LIME requires text containing at least one word.")
        requested_features = num_features or max(1, len(unique_words))
        explanation = self.explainer.explain_instance(
            text,
            self._predict,
            labels=(explained_class,),
            num_features=requested_features,
            num_samples=num_samples,
        )

        feature_attributions = [
            {"feature": feature, "weight": float(weight)}
            for feature, weight in explanation.as_list(label=explained_class)
        ]
        exact_weights = {
            item["feature"]: item["weight"] for item in feature_attributions
        }
        folded_weights = {
            item["feature"].casefold(): item["weight"]
            for item in feature_attributions
        }
        token_scores = [
            exact_weights.get(token, folded_weights.get(token.casefold(), 0.0))
            if re.search(r"\w", token)
            else 0.0
            for token in word_tokens
        ]
        rationales = extract_rationale_spans(word_tokens, token_scores)

        intercept = explanation.intercept.get(explained_class)
        local_prediction = np.asarray(explanation.local_pred).reshape(-1)
        return {
            "lime_explanation": explanation,
            "tokens": word_tokens,
            "attributions": token_scores,
            "feature_attributions": feature_attributions,
            "rationales": rationales,
            "highlighted_text": highlight_rationales(text, rationales),
            "predicted_class": predicted_class,
            "predicted_label": self.label_names[predicted_class],
            "confidence": float(probabilities[predicted_class]),
            "explained_class": explained_class,
            "explained_label": self.label_names[explained_class],
            "explained_probability": float(probabilities[explained_class]),
            "class_scores": {
                label: float(probabilities[index])
                for index, label in enumerate(self.label_names)
            },
            "local_prediction": (
                float(local_prediction[0]) if local_prediction.size else None
            ),
            "local_intercept": float(intercept) if intercept is not None else None,
            "local_r2": float(explanation.score),
            "num_samples": num_samples,
            "text": text,
        }

    def save_plot(self, result: dict, save_path: str | Path, max_features: int = 20):
        """Save the strongest LIME word weights as a horizontal bar chart."""
        features = result["feature_attributions"][:max_features]
        if not features:
            return

        # Put the strongest item at the top of the horizontal chart.
        features = list(reversed(features))
        names = [item["feature"] for item in features]
        weights = [float(item["weight"]) for item in features]
        colours = ["#1B7A5E" if weight >= 0 else "#9B2335" for weight in weights]

        fig, ax = plt.subplots(figsize=(8, max(4, len(features) * 0.35)))
        ax.barh(range(len(names)), weights, color=colours, edgecolor="white", linewidth=0.4)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.axvline(0, color="#555", linewidth=0.6)
        ax.set_xlabel("LIME weight", fontsize=10)
        ax.set_title(
            f"LIME local explanation — class: '{result['explained_label']}' "
            f"(p={result['explained_probability']:.1%}, local R²={result['local_r2']:.2f})",
            fontsize=11,
            pad=10,
        )
        positive = mpatches.Patch(color="#1B7A5E", label="Supports explained class")
        negative = mpatches.Patch(color="#9B2335", label="Opposes explained class")
        ax.legend(handles=[positive, negative], fontsize=9, framealpha=0.6)

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[LIME] Saved bar plot → {save_path}")

    @staticmethod
    def save_html(result: dict, save_path: str | Path):
        """Save LIME's interactive local-explanation report."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        result["lime_explanation"].save_to_file(
            str(save_path), labels=(result["explained_class"],)
        )
        print(f"[LIME] Saved HTML explanation → {save_path}")
