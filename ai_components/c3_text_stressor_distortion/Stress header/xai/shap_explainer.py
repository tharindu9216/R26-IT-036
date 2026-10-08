"""
SHAP explainability for the Stress Header classifier.
Uses shap.Explainer (Partition / Text explainer) for token-level attribution.

Usage:
    from xai.shap_explainer import StressSHAP
    sx = StressSHAP(pipeline, label_names)
    result = sx.explain("I'm overwhelmed by my assignments")
    sx.save_plot(result, "reports/.../xai_outputs/shap_plots/sample.png")
"""

import shap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

from decision import select_predicted_class
from rationale_extractor import extract_rationale_spans, highlight_rationales


class StressSHAP:
    """
    SHAP token-level explainer for the binary stress head.
    Works with either this project's DualHeadStressModel or a compatible
    HuggingFace text-classification pipeline.

    The shap.Explainer (PartitionExplainer under the hood for text) masks
    token subsets and measures how predictions change — giving reliable
    attribution without needing model internals.
    """

    def __init__(self, model_or_pipeline, tokenizer=None, label_names: list[str] | dict = None,
                 device: str = None, max_length: int = 192,
                 decision_threshold: float | None = None,
                 prediction_batch_size: int = 8):
        """
        Args:
            model_or_pipeline : DualHeadStressModel or transformers text-classification pipeline
            tokenizer         : required when model_or_pipeline is DualHeadStressModel
            label_names       : ordered list/dict matching Head 1A output positions
            device            : "cuda" | "cpu" | None (auto-detect)
            max_length        : tokenizer max length used during training/inference
        """
        self.model_or_pipeline = model_or_pipeline
        self.tokenizer = tokenizer or getattr(model_or_pipeline, "tokenizer", None)
        if self.tokenizer is None:
            raise ValueError("A tokenizer is required for SHAP text masking.")

        self.label_names = self._normalise_label_names(label_names)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.decision_threshold = decision_threshold
        self.prediction_batch_size = max(1, int(prediction_batch_size))
        self.is_pipeline = callable(model_or_pipeline) and hasattr(model_or_pipeline, "tokenizer")

        if not self.is_pipeline:
            self.model_or_pipeline = model_or_pipeline.to(self.device).eval()

        # Wrap pipeline into a function that returns a 2-D probability array
        def _predict(texts):
            texts = list(texts)
            probability_batches = []
            for start in range(0, len(texts), self.prediction_batch_size):
                batch = texts[start:start + self.prediction_batch_size]
                if self.is_pipeline:
                    results = self.model_or_pipeline(
                        batch, return_all_scores=True)
                    probability_batches.append(
                        self._pipeline_results_to_probs(results))
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
                    logits_1a, _ = self.model_or_pipeline(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    probability_batches.append(
                        torch.softmax(logits_1a, dim=1)
                        .detach().cpu().numpy()
                    )
            if not probability_batches:
                return np.empty((0, len(self.label_names)), dtype=float)
            return np.concatenate(probability_batches, axis=0)

        self._predict = _predict
        self.explainer = shap.Explainer(
            _predict,
            shap.maskers.Text(self.tokenizer),
            output_names=self.label_names,
        )

    @staticmethod
    def _normalise_label_names(label_names: list[str] | dict | None) -> list[str]:
        if label_names is None:
            return ["Not Stressed", "Stressed"]
        if isinstance(label_names, dict):
            return [label_names[str(i)] for i in range(len(label_names))]
        return list(label_names)

    def _pipeline_results_to_probs(self, results) -> np.ndarray:
        """Convert HF pipeline output to probabilities ordered by label_names."""
        if results and isinstance(results[0], dict):
            results = [results]

        rows = []
        for row in results:
            by_label = {item["label"]: item["score"] for item in row}
            values = []
            for idx, label in enumerate(self.label_names):
                values.append(
                    by_label.get(label,
                                 by_label.get(str(idx),
                                              by_label.get(f"LABEL_{idx}", 0.0)))
                )
            rows.append(values)
        return np.array(rows, dtype=float)

    @staticmethod
    def _single_explanation(shap_values):
        """Return one sample explanation from either batched or single SHAP output."""
        if len(getattr(shap_values, "shape", ())) == 3:
            return shap_values[0]
        return shap_values

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def explain(self, text: str, target_class: int | None = None) -> dict:
        """
        Compute SHAP values for a single text.

        Returns:
            dict with keys:
                shap_values  : shap.Explanation object (use for shap plots)
                tokens       : list[str]
                predicted_label : str
                confidence   : float
                text         : str
        """
        shap_values = self.explainer([text])   # shape: (1, n_tokens, n_classes)

        probs = self._predict([text])[0]
        predicted_idx = select_predicted_class(
            probs, self.decision_threshold)
        explained_idx = predicted_idx if target_class is None else target_class
        if not 0 <= explained_idx < len(self.label_names):
            raise ValueError(
                f"target_class must be in [0, {len(self.label_names) - 1}]."
            )
        predicted_label = self.label_names[predicted_idx]
        confidence = float(probs[predicted_idx])
        scores = {label: float(probs[i]) for i, label in enumerate(self.label_names)}

        tokens = shap_values[0].data   # list of token strings
        sv = self._single_explanation(shap_values)
        token_scores = sv.values[:, explained_idx].tolist()
        rationales = extract_rationale_spans(list(tokens), token_scores)

        return {
            "shap_values": shap_values,
            "tokens": list(tokens),
            "attributions": token_scores,
            "rationales": rationales,
            "highlighted_text": highlight_rationales(text, rationales),
            "predicted_class": predicted_idx,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "explained_class": explained_idx,
            "explained_label": self.label_names[explained_idx],
            "explained_probability": float(probs[explained_idx]),
            "text": text,
            "class_scores": scores,
        }

    def batch_explain(self, texts: list[str]) -> list[dict]:
        """Explain a list of texts (more efficient — SHAP batches internally)."""
        shap_values = self.explainer(texts)
        probs = self._predict(texts)
        out = []
        for i, text in enumerate(texts):
            predicted_idx = select_predicted_class(
                probs[i], self.decision_threshold)
            predicted_label = self.label_names[predicted_idx]
            token_scores = shap_values[i].values[:, predicted_idx].tolist()
            rationales = extract_rationale_spans(list(shap_values[i].data), token_scores)
            out.append({
                "shap_values": shap_values[i],
                "tokens": list(shap_values[i].data),
                "attributions": token_scores,
                "rationales": rationales,
                "highlighted_text": highlight_rationales(text, rationales),
                "predicted_label": predicted_label,
                "confidence": float(probs[i][predicted_idx]),
                "text": text,
                "class_scores": {
                    label: float(probs[i][idx])
                    for idx, label in enumerate(self.label_names)
                },
            })
        return out

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def save_text_plot(self, result: dict, save_path: str, class_name: str = None):
        """
        Save SHAP text plot (inline token highlighting).
        If class_name is None, uses the predicted class.

        Args:
            result    : output from .explain()
            save_path : full output path, e.g. ".../shap_plots/sample.html"
        """
        sv = self._single_explanation(result["shap_values"])
        cls = class_name or result.get("explained_label", result["predicted_label"])
        class_idx = self.label_names.index(cls) if cls in self.label_names else 0

        # shap.plots.text returns HTML on recent SHAP versions; older versions
        # may write to stdout, so keep the fallback capture.
        import io, contextlib
        html_buf = io.StringIO()
        with contextlib.redirect_stdout(html_buf):
            html = shap.plots.text(sv[:, class_idx], display=False)
        if html is None:
            html = html_buf.getvalue()
        else:
            html = str(html)

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[SHAP] Saved text plot → {save_path}")

    def save_bar_plot(self, result: dict, save_path: str):
        """
        Save a clean horizontal bar chart of per-token SHAP values
        for the predicted class.

        Args:
            result    : output from .explain()
            save_path : full path including filename (.png)
        """
        sv = self._single_explanation(result["shap_values"])
        predicted_label = result.get("explained_label", result["predicted_label"])
        confidence = result.get("explained_probability", result["confidence"])

        if predicted_label not in self.label_names:
            print(f"[SHAP] Warning: label '{predicted_label}' not in label_names")
            return

        class_idx = self.label_names.index(predicted_label)

        # sv.values shape: (n_tokens, n_classes)
        if hasattr(sv, "values"):
            token_shap = sv.values[:, class_idx]
        else:
            token_shap = np.array(sv)[:, class_idx]

        tokens = result["tokens"]

        # Filter padding / special tokens
        keep = [(t, float(v)) for t, v in zip(tokens, token_shap)
                if t not in ["<s>", "</s>", "<pad>", "[CLS]", "[SEP]"]]
        if not keep:
            return

        toks, vals = zip(*keep)
        colours = ["#1B7A5E" if v >= 0 else "#9B2335" for v in vals]

        fig, ax = plt.subplots(figsize=(max(8, len(toks) * 0.6), 4))
        ax.bar(range(len(toks)), vals, color=colours, edgecolor="white", linewidth=0.4)
        ax.set_xticks(range(len(toks)))
        ax.set_xticklabels(toks, rotation=45, ha="right", fontsize=9)
        ax.axhline(0, color="#555", linewidth=0.6)
        ax.set_ylabel("SHAP value", fontsize=10)
        ax.set_title(
            f"SHAP attributions — class: '{predicted_label}' (conf: {confidence:.1%})",
            fontsize=11, pad=10
        )

        pos_patch = mpatches.Patch(color="#1B7A5E", label="Positive contribution")
        neg_patch = mpatches.Patch(color="#9B2335", label="Negative contribution")
        ax.legend(handles=[pos_patch, neg_patch], fontsize=9, framealpha=0.6)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] Saved bar plot → {save_path}")

    def save_beeswarm(self, batch_results: list[dict], save_path: str, class_name: str = None):
        """
        Save a SHAP beeswarm plot across a batch of samples.
        Useful for dataset-level feature importance analysis.

        Args:
            batch_results : list of dicts from .batch_explain()
            save_path     : .png output path
            class_name    : stressor class to plot; defaults to most common predicted
        """
        # Determine class to plot
        if class_name is None:
            from collections import Counter
            class_name = Counter(r["predicted_label"] for r in batch_results).most_common(1)[0][0]

        class_idx = self.label_names.index(class_name)

        # Build a combined Explanation object — pad shorter sequences
        all_sv = []
        for result in batch_results:
            sv = self._single_explanation(result["shap_values"])
            all_sv.append(sv.values[:, class_idx])
        max_len = max(len(x) for x in all_sv)
        padded = np.array([
            np.pad(v, (0, max_len - len(v))) for v in all_sv
        ])

        all_tokens = [r["tokens"] for r in batch_results]
        feature_names = all_tokens[np.argmax([len(t) for t in all_tokens])]

        exp = shap.Explanation(
            values=padded,
            feature_names=feature_names[:max_len],
        )

        shap.plots.beeswarm(exp, max_display=15, show=False)
        plt.title(f"SHAP beeswarm — class: '{class_name}' (n={len(batch_results)})", fontsize=11)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] Saved beeswarm → {save_path}")
