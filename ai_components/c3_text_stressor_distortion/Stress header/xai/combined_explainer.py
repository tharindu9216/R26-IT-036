"""Consensus explanation combining SHAP, LIME, and Integrated Gradients."""

from __future__ import annotations

import re
from collections import defaultdict, deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from integrated_gradients import StressIG
from lime_explainer import TOKEN_PATTERN, StressLIME
from rationale_extractor import (
    extract_rationale_spans,
    highlight_rationales,
    tokens_to_words,
)
from shap_explainer import StressSHAP


class StressCombinedXAI:
    """Align and average local evidence from three complementary XAI methods."""

    def __init__(
        self,
        model,
        tokenizer,
        label_names: list[str] | dict | None = None,
        device: str | None = None,
        max_length: int = 192,
        decision_threshold: float | None = None,
    ):
        common = {
            "label_names": label_names,
            "device": device,
            "max_length": max_length,
            "decision_threshold": decision_threshold,
        }
        self.lime = StressLIME(model, tokenizer, **common)
        self.shap = StressSHAP(model, tokenizer, **common)
        self.integrated_gradients = StressIG(
            model,
            tokenizer,
            **common,
        )

    @staticmethod
    def _word_key(word: str) -> str:
        return re.sub(r"^\W+|\W+$", "", word.casefold())

    @staticmethod
    def _normalise(scores: list[float]) -> list[float]:
        values = np.asarray(scores, dtype=float)
        if not values.size:
            return []
        scale = float(np.max(np.abs(values)))
        if scale <= 1e-12:
            return [0.0] * len(values)
        return (values / scale).tolist()

    @classmethod
    def _align(
        cls,
        canonical_words: list[str],
        source_words: list[str],
        source_scores: list[float],
    ) -> list[float]:
        """Align method-specific word scores by normalized word occurrence."""
        available: dict[str, deque[float]] = defaultdict(deque)
        for word, score in zip(source_words, source_scores):
            key = cls._word_key(word)
            if key:
                available[key].append(float(score))

        aligned = []
        for word in canonical_words:
            key = cls._word_key(word)
            aligned.append(available[key].popleft() if available[key] else 0.0)
        return cls._normalise(aligned)

    @staticmethod
    def _token_result_words(result: dict) -> tuple[list[str], list[float]]:
        words = tokens_to_words(result["tokens"], result["attributions"])
        return [word.text for word in words], [word.score for word in words]

    @classmethod
    def combine_results(
        cls,
        text: str,
        lime_result: dict,
        shap_result: dict,
        ig_result: dict,
    ) -> dict:
        """Create an equal-weight word consensus from completed method results."""
        canonical_words = [
            token for token in TOKEN_PATTERN.findall(text) if re.search(r"\w", token)
        ]
        if not canonical_words:
            raise ValueError("Combined XAI requires text containing at least one word.")

        lime_words, lime_scores = cls._token_result_words(lime_result)
        shap_words, shap_scores = cls._token_result_words(shap_result)
        aligned = {
            "lime": cls._align(canonical_words, lime_words, lime_scores),
            "shap": cls._align(canonical_words, shap_words, shap_scores),
        }
        ig_words, ig_scores = cls._token_result_words(ig_result)
        aligned["integrated_gradients"] = cls._align(
            canonical_words,
            ig_words,
            ig_scores,
        )
        methods_used = ["SHAP", "LIME", "Integrated Gradients"]

        consensus_scores = []
        feature_rows = []
        for index, word in enumerate(canonical_words):
            ig_score = aligned["integrated_gradients"][index]
            available_scores = [
                aligned["lime"][index],
                aligned["shap"][index],
                ig_score,
            ]
            consensus = float(np.mean(available_scores))
            consensus_scores.append(consensus)

            nonzero = [score for score in available_scores if abs(score) > 1e-9]
            if nonzero and abs(consensus) > 1e-9:
                direction = 1 if consensus > 0 else -1
                agreement = sum(
                    1 for score in nonzero if (1 if score > 0 else -1) == direction
                ) / len(nonzero)
            else:
                agreement = 0.0
            feature_rows.append(
                {
                    "word": word,
                    "lime": aligned["lime"][index],
                    "shap": aligned["shap"][index],
                    "integrated_gradients": ig_score,
                    "consensus": consensus,
                    "agreement": agreement,
                }
            )

        ranked_features = sorted(
            feature_rows,
            key=lambda item: abs(item["consensus"]),
            reverse=True,
        )
        rationales = extract_rationale_spans(canonical_words, consensus_scores)
        return {
            "tokens": canonical_words,
            "attributions": consensus_scores,
            "rationales": rationales,
            "highlighted_text": highlight_rationales(text, rationales),
            "feature_rows": feature_rows,
            "ranked_features": ranked_features,
            "methods_used": methods_used,
            "predicted_class": shap_result["predicted_class"],
            "predicted_label": shap_result["predicted_label"],
            "confidence": shap_result["confidence"],
            "explained_class": shap_result["explained_class"],
            "explained_label": shap_result["explained_label"],
            "explained_probability": shap_result["explained_probability"],
            "text": text,
            "lime_result": lime_result,
            "shap_result": shap_result,
            "integrated_gradients_result": ig_result,
        }

    def explain(
        self,
        text: str,
        target_class: int | None = None,
        lime_samples: int = 1000,
        ig_steps: int = 50,
    ) -> dict:
        lime_result = self.lime.explain(
            text,
            target_class=target_class,
            num_samples=lime_samples,
        )
        shap_result = self.shap.explain(text, target_class=target_class)
        ig_result = self.integrated_gradients.explain(
            text,
            target_class=target_class,
            n_steps=ig_steps,
        )
        return self.combine_results(text, lime_result, shap_result, ig_result)

    @staticmethod
    def save_plot(
        result: dict,
        save_path: str | Path,
        max_features: int = 15,
    ) -> None:
        rows = list(reversed(result["ranked_features"][:max_features]))
        if not rows:
            return

        names = [row["word"] for row in rows]
        positions = np.arange(len(rows), dtype=float)
        series = [
            ("SHAP", "shap", "#61DDAA"),
            ("LIME", "lime", "#5B8FF9"),
            ("Integrated Gradients", "integrated_gradients", "#F6BD16"),
            ("Consensus", "consensus", "#7262FD"),
        ]

        height = min(0.8 / len(series), 0.2)
        offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * height
        fig, ax = plt.subplots(figsize=(10, max(5, len(rows) * 0.42)))
        for offset, (label, key, colour) in zip(offsets, series):
            ax.barh(
                positions + offset,
                [float(row[key]) for row in rows],
                height=height,
                label=label,
                color=colour,
            )
        ax.set_yticks(positions)
        ax.set_yticklabels(names, fontsize=9)
        ax.axvline(0, color="#555", linewidth=0.7)
        ax.set_xlim(-1.05, 1.05)
        ax.set_xlabel("Normalized contribution to explained class")
        ax.set_title(
            "Combined SHAP + LIME + Integrated Gradients — "
            f"{result['explained_label']} ({result['explained_probability']:.1%})"
        )
        ax.legend(fontsize=9)
        fig.tight_layout()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
