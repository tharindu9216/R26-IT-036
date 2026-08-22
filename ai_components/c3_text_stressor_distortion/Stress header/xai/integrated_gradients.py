"""
Integrated Gradients explainability for the Stress Header classifier.
Uses Captum to produce token-level attribution scores.

Usage:
    from xai.integrated_gradients import StressIG
    ig = StressIG(model, tokenizer, label_names)
    result = ig.explain("I can't handle my exam deadlines anymore")
    ig.save_plot(result, "reports/.../Stress header/evaluation/xai_outputs/ig_plots/sample.png")
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from captum.attr import LayerIntegratedGradients

from decision import select_predicted_class
from rationale_extractor import extract_rationale_spans, highlight_rationales


class StressIG:
    """
    Token-level Integrated Gradients explainer for the binary stress head.
    Wraps this project's DualHeadStressModel and explains Head 1A.
    """

    def __init__(self, model, tokenizer, label_names: list[str] | dict = None,
                 device: str = None, max_length: int = 192,
                 decision_threshold: float | None = None,
                 internal_batch_size: int = 1):
        """
        Args:
            model       : fine-tuned DualHeadStressModel
            tokenizer   : matching tokenizer
            label_names : ordered list/dict matching Head 1A output positions
                          e.g. ["Not Stressed", "Stressed"] or {"0": "..."}
            device      : "cuda" | "cpu" | None (auto-detect)
            max_length  : tokenizer max length used during training/inference
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.label_names = self._normalise_label_names(label_names)
        self.max_length = max_length
        self.decision_threshold = decision_threshold
        self.internal_batch_size = max(1, int(internal_batch_size))

        # Attach IG to the encoder embedding layer used by DualHeadStressModel.
        self.lig = LayerIntegratedGradients(
            self._forward_func,
            self._get_embedding_layer(),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_label_names(label_names: list[str] | dict | None) -> list[str]:
        if label_names is None:
            return ["Not Stressed", "Stressed"]
        if isinstance(label_names, dict):
            return [label_names[str(i)] for i in range(len(label_names))]
        return list(label_names)

    def _get_embedding_layer(self):
        """Return the token embedding layer for the wrapped encoder."""
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "get_input_embeddings"):
            return self.model.encoder.get_input_embeddings()
        if hasattr(self.model, "get_input_embeddings"):
            return self.model.get_input_embeddings()
        raise AttributeError("Could not find an input embedding layer on the provided model.")

    def _forward_func(self, input_ids, attention_mask=None, target_class=None):
        """Scalar output for Captum: Head 1A logit for the target class."""
        logits_1a, _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return logits_1a[:, target_class]

    def _get_baseline(self, input_ids):
        """Baseline = all PAD tokens, preserving the same sequence shape."""
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        return torch.full_like(input_ids, pad_id)

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def explain(self, text: str, target_class: int = None, n_steps: int = 50) -> dict:
        """
        Run Integrated Gradients on a single text.

        Args:
            text         : raw input string
            target_class : class index to explain. If None, uses the predicted class.
            n_steps      : IG approximation steps (50 is a good default)

        Returns:
            dict with keys:
                tokens       : list[str]   – wordpiece tokens
                attributions : list[float] – signed per-token attribution
                predicted_class : int
                predicted_label : str
                confidence   : float
                text         : str
        """
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            # A transformer does not require padding to its training maximum
            # for one sample. Dynamic length is critical for CBT checkpoints
            # whose configured maximum is 448/512 tokens.
            padding=False,
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Predict class if not specified
        with torch.no_grad():
            logits_1a, _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits_1a, dim=-1).squeeze(0)
            model_prediction = select_predicted_class(
                probs.tolist(), self.decision_threshold)
            explained_class = (
                model_prediction if target_class is None else target_class)
            if not 0 <= explained_class < len(self.label_names):
                raise ValueError(
                    f"target_class must be in [0, {len(self.label_names) - 1}]."
                )
            confidence = float(probs[model_prediction])

        baseline = self._get_baseline(input_ids)

        # Run IG. Convergence delta is intentionally not requested here:
        # captum's LayerIntegratedGradients raises a spurious shape-mismatch
        # assertion in its convergence-delta check when combined with
        # internal_batch_size, and the delta isn't surfaced anywhere downstream.
        attributions = self.lig.attribute(
            inputs=input_ids,
            baselines=baseline,
            additional_forward_args=(attention_mask, explained_class),
            n_steps=n_steps,
            internal_batch_size=self.internal_batch_size,
        )

        # Summarise per token with signed attribution, then normalize to [-1, 1].
        attrs = attributions.squeeze(0)          # (seq_len, embed_dim)
        token_scores = attrs.sum(dim=-1).detach().cpu().numpy()
        max_abs = np.max(np.abs(token_scores))
        if max_abs == 0:
            max_abs = 1.0
        token_scores_norm = (token_scores / max_abs).tolist()

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        rationales = extract_rationale_spans(tokens, token_scores_norm)

        return {
            "tokens": tokens,
            "attributions": token_scores_norm,
            "rationales": rationales,
            "highlighted_text": highlight_rationales(text, rationales),
            "predicted_class": model_prediction,
            "predicted_label": self.label_names[model_prediction],
            "confidence": confidence,
            "explained_class": explained_class,
            "explained_label": self.label_names[explained_class],
            "explained_probability": float(probs[explained_class]),
            "text": text,
        }

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def save_plot(self, result: dict, save_path: str):
        """
        Save a horizontal token-attribution bar chart.
        Positive values support the explained class; negative values oppose it.

        Args:
            result    : output dict from .explain()
            save_path : full path including filename, e.g. ".../ig_plots/sample.png"
        """
        tokens = result["tokens"]
        scores = result["attributions"]
        label = result.get("explained_label", result["predicted_label"])
        conf = result.get("explained_probability", result["confidence"])

        # Drop [CLS] and [SEP] for cleaner plot
        pairs = [(t, s) for t, s in zip(tokens, scores)
                 if t not in ("<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "[PAD]")]

        if not pairs:
            return

        toks, vals = zip(*pairs)
        colours = ["#1B7A5E" if float(v) >= 0 else "#9B2335" for v in vals]

        fig, ax = plt.subplots(figsize=(max(8, len(toks) * 0.55), 4))
        ax.bar(range(len(toks)), vals, color=colours, edgecolor="white", linewidth=0.5)

        ax.set_xticks(range(len(toks)))
        ax.set_xticklabels(toks, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Attribution score (normalised)", fontsize=10)
        ax.set_title(
            f"Integrated Gradients — predicted: '{label}' (conf: {conf:.1%})",
            fontsize=11, pad=10
        )
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_ylim(-1.15, 1.15)

        pos_patch = mpatches.Patch(color="#1B7A5E", label="Supports class")
        neg_patch = mpatches.Patch(color="#9B2335", label="Opposes class")
        ax.legend(handles=[pos_patch, neg_patch], fontsize=9, framealpha=0.6)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[IG] Saved plot → {save_path}")

    def batch_explain(self, texts: list[str], **kwargs) -> list[dict]:
        """Convenience wrapper for multiple samples."""
        return [self.explain(t, **kwargs) for t in texts]
