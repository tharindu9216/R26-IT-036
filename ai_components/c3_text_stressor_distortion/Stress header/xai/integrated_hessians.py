"""
Token-interaction explanations for the Stress Header using Integrated Hessians.

The implementation follows the Integrated Hessians definition from:
Janizek, Sturmfels, and Lee (JMLR, 2021). It operates on scalar gates over
token embeddings, so each matrix entry describes the interaction between two
tokens rather than between individual embedding dimensions.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from rationale_extractor import LOW_INFORMATION_WORDS, tokens_to_words


class StressIntegratedHessians:
    """Embedding-level Integrated Hessians for Head 1A of the stress model."""

    def __init__(
        self,
        model,
        tokenizer,
        label_names: list[str] | dict | None = None,
        device: str | None = None,
        max_length: int = 192,
        max_interaction_tokens: int = 48,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.label_names = self._normalise_label_names(label_names)
        self.max_length = max_length
        self.max_interaction_tokens = max(4, min(max_interaction_tokens, max_length))

        required = ("encoder", "layer_norm", "dropout", "head_1a")
        missing = [name for name in required if not hasattr(self.model, name)]
        if missing:
            raise TypeError(
                "Integrated Hessians requires the DualHeadStressModel interface; "
                f"missing: {', '.join(missing)}"
            )
        if not hasattr(self.model.encoder, "get_input_embeddings"):
            raise TypeError("The transformer encoder does not expose its embedding layer.")

    @staticmethod
    def _normalise_label_names(label_names: list[str] | dict | None) -> list[str]:
        if label_names is None:
            return ["Not Stressed", "Stressed"]
        if isinstance(label_names, dict):
            return [label_names[str(i)] for i in range(len(label_names))]
        return list(label_names)

    def _forward_from_embeddings(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        target_class: int,
    ) -> torch.Tensor:
        output = self.model.encoder(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
        )
        cls = output.last_hidden_state[:, 0, :]
        cls = self.model.dropout(self.model.layer_norm(cls))
        return self.model.head_1a(cls)[:, target_class]

    def _embedding_path(
        self,
        baseline_embeddings: torch.Tensor,
        embedding_delta: torch.Tensor,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        return baseline_embeddings + gates.view(1, -1, 1) * embedding_delta

    @staticmethod
    def _quadrature(n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        if n_steps < 2:
            raise ValueError("n_steps must be at least 2.")
        nodes, weights = np.polynomial.legendre.leggauss(n_steps)
        return (nodes + 1.0) / 2.0, weights / 2.0

    @staticmethod
    def _normalised_error(observed: float, expected: float) -> float:
        return abs(observed - expected) / max(abs(expected), 1e-6)

    @staticmethod
    def _relative_matrix_change(current: torch.Tensor, previous: torch.Tensor) -> float:
        numerator = torch.linalg.vector_norm(current - previous)
        denominator = torch.linalg.vector_norm(current).clamp_min(1e-8)
        return float((numerator / denominator).item())

    @staticmethod
    def _is_content_word(text: str) -> bool:
        key = text.lower().strip(" \t\r\n,.;:!?\"'`()[]{}")
        return (
            bool(key)
            and key not in LOW_INFORMATION_WORDS
            and bool(re.search(r"[a-z0-9]", key))
        )

    @staticmethod
    def _is_punctuation_token(token: str) -> bool:
        cleaned = token
        if cleaned.startswith(("▁", "Ġ", "_")):
            cleaned = cleaned[1:]
        if cleaned.startswith("##"):
            cleaned = cleaned[2:]
        return bool(cleaned) and bool(re.fullmatch(r"[^\w\s]+", cleaned))

    @staticmethod
    def _aggregate_words(
        tokens: list[str],
        token_attributions: np.ndarray,
        pair_matrix: np.ndarray,
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        word_groups = tokens_to_words(tokens, token_attributions.tolist())
        words = [
            word.text.strip(" \t\r\n,.;:!?\"'`()[]{}") or word.text
            for word in word_groups
        ]
        word_attributions = np.zeros(len(word_groups), dtype=float)
        word_matrix = np.zeros((len(word_groups), len(word_groups)), dtype=float)

        ranges = [
            [
                index
                for index in range(word.token_start, word.token_end + 1)
                if not StressIntegratedHessians._is_punctuation_token(tokens[index])
            ]
            for word in word_groups
        ]
        for index, token_range in enumerate(ranges):
            if token_range:
                word_attributions[index] = float(token_attributions[token_range].sum())
        for left, left_range in enumerate(ranges):
            for right, right_range in enumerate(ranges):
                if left == right or not left_range or not right_range:
                    continue
                word_matrix[left, right] = float(
                    pair_matrix[np.ix_(left_range, right_range)].sum()
                )
        return words, word_attributions, word_matrix

    def explain(
        self,
        text: str,
        target_class: int | None = None,
        n_steps: int = 8,
        attribution_steps: int = 32,
        max_attribution_steps: int = 128,
        max_interaction_steps: int = 16,
        convergence_tolerance: float = 0.05,
        interaction_tolerance: float = 0.20,
        top_k: int = 12,
    ) -> dict:
        """
        Explain pairwise token interactions for one text.

        The double Integrated Hessians integral is evaluated as the equivalent
        one-dimensional integral:

            integral_0^1 -t log(t) H(x' + t(x-x')) dt

        Attribution and interaction integration are increased adaptively until
        their numerical checks pass or the configured step limits are reached.
        Diagonal entries are retained for completeness accounting but excluded
        from the pair ranking and visualisation.
        """
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_interaction_tokens,
            padding="max_length",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        active_length = int(attention_mask[0].sum().item())
        input_ids = input_ids[:, :active_length]
        attention_mask = attention_mask[:, :active_length]

        with torch.no_grad():
            logits_1a, _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            probs = torch.softmax(logits_1a, dim=-1).squeeze(0)
            model_prediction = int(probs.argmax().item())
            explained_class = model_prediction if target_class is None else target_class
            if not 0 <= explained_class < len(self.label_names):
                raise ValueError(f"target_class must be in [0, {len(self.label_names) - 1}].")

        embedding_layer = self.model.encoder.get_input_embeddings()
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        baseline_ids = torch.full_like(input_ids, pad_id)
        special_ids = set(getattr(self.tokenizer, "all_special_ids", []))
        # Keep CLS/SEP and other structural tokens fixed. Replacing them with
        # PAD creates a baseline outside the transformer's normal input format.
        for index, token_id in enumerate(input_ids.squeeze(0).tolist()):
            if token_id in special_ids and token_id != pad_id:
                baseline_ids[0, index] = token_id
        input_embeddings = embedding_layer(input_ids).detach()
        baseline_embeddings = embedding_layer(baseline_ids).detach()
        embedding_delta = input_embeddings - baseline_embeddings

        def score_from_gates(gates: torch.Tensor) -> torch.Tensor:
            embeddings = self._embedding_path(
                baseline_embeddings,
                embedding_delta,
                gates,
            )
            return self._forward_from_embeddings(
                embeddings,
                attention_mask,
                explained_class,
            ).squeeze(0)

        with torch.no_grad():
            zero_gates = torch.zeros(
                active_length,
                device=self.device,
                dtype=input_embeddings.dtype,
            )
            one_gates = torch.ones_like(zero_gates)
            baseline_score = float(score_from_gates(zero_gates).item())
            input_score = float(score_from_gates(one_gates).item())
        score_difference = input_score - baseline_score

        def integrate_attributions(steps: int) -> torch.Tensor:
            nodes, weights = self._quadrature(steps)
            values = torch.zeros(
                active_length,
                device=self.device,
                dtype=input_embeddings.dtype,
            )
            for node, weight in zip(nodes, weights):
                gates = torch.full(
                    (active_length,),
                    float(node),
                    device=self.device,
                    dtype=input_embeddings.dtype,
                    requires_grad=True,
                )
                gradient = torch.autograd.grad(score_from_gates(gates), gates)[0]
                values += float(weight) * gradient.detach()
            return values

        used_attribution_steps = max(2, attribution_steps)
        token_attributions = integrate_attributions(used_attribution_steps)
        completeness_error = float(token_attributions.sum().item() - score_difference)
        normalised_completeness_error = self._normalised_error(
            float(token_attributions.sum().item()),
            score_difference,
        )
        while (
            normalised_completeness_error > convergence_tolerance
            and used_attribution_steps < max_attribution_steps
        ):
            used_attribution_steps = min(
                used_attribution_steps * 2,
                max_attribution_steps,
            )
            token_attributions = integrate_attributions(used_attribution_steps)
            completeness_error = float(
                token_attributions.sum().item() - score_difference
            )
            normalised_completeness_error = self._normalised_error(
                float(token_attributions.sum().item()),
                score_difference,
            )

        def integrate_interactions(steps: int) -> torch.Tensor:
            nodes, weights = self._quadrature(steps)
            values = torch.zeros(
                (active_length, active_length),
                device=self.device,
                dtype=input_embeddings.dtype,
            )
            for node, weight in zip(nodes, weights):
                gates = torch.full(
                    (active_length,),
                    float(node),
                    device=self.device,
                    dtype=input_embeddings.dtype,
                    requires_grad=True,
                )
                hessian = torch.autograd.functional.hessian(
                    score_from_gates,
                    gates,
                    create_graph=False,
                    vectorize=False,
                )
                ih_weight = float(weight) * (-float(node) * math.log(float(node)))
                values += ih_weight * hessian.detach()
            values.fill_diagonal_(0.0)
            return (values + values.T) / 2.0

        used_interaction_steps = max(4, n_steps)
        coarse_steps = max(2, used_interaction_steps // 2)
        coarse_interactions = integrate_interactions(coarse_steps)
        off_diagonal = integrate_interactions(used_interaction_steps)
        interaction_change = self._relative_matrix_change(
            off_diagonal,
            coarse_interactions,
        )
        while (
            interaction_change > interaction_tolerance
            and used_interaction_steps < max_interaction_steps
        ):
            coarse_interactions = off_diagonal
            used_interaction_steps = min(
                used_interaction_steps * 2,
                max_interaction_steps,
            )
            off_diagonal = integrate_interactions(used_interaction_steps)
            interaction_change = self._relative_matrix_change(
                off_diagonal,
                coarse_interactions,
            )

        interaction_matrix = off_diagonal.clone()
        diagonal = token_attributions - off_diagonal.sum(dim=1)
        interaction_matrix.diagonal().copy_(diagonal)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
        matrix_np = interaction_matrix.detach().cpu().numpy()
        pair_matrix_np = off_diagonal.detach().cpu().numpy()
        attrs_np = token_attributions.detach().cpu().numpy()
        words, word_attributions, word_matrix = self._aggregate_words(
            tokens,
            attrs_np,
            pair_matrix_np,
        )
        content_indices = [
            index for index, word in enumerate(words) if self._is_content_word(word)
        ]

        pairs = []
        for left_pos, left_index in enumerate(content_indices):
            for right_index in content_indices[left_pos + 1 :]:
                score = float(word_matrix[left_index, right_index])
                pairs.append(
                    {
                        "left_token": words[left_index],
                        "right_token": words[right_index],
                        "left_word": words[left_index],
                        "right_word": words[right_index],
                        "left_index": left_index,
                        "right_index": right_index,
                        "score": score,
                        "absolute_score": abs(score),
                        "effect": "synergy" if score >= 0 else "opposition",
                    }
                )
        pairs.sort(key=lambda item: item["absolute_score"], reverse=True)

        return {
            "tokens": tokens,
            "attributions": attrs_np.tolist(),
            "interaction_matrix": matrix_np.tolist(),
            "pair_interaction_matrix": pair_matrix_np.tolist(),
            "words": words,
            "word_attributions": word_attributions.tolist(),
            "word_interaction_matrix": word_matrix.tolist(),
            "content_word_indices": content_indices,
            "top_interactions": pairs[:top_k],
            "predicted_class": model_prediction,
            "model_predicted_label": self.label_names[model_prediction],
            "explained_class": explained_class,
            "predicted_label": self.label_names[explained_class],
            "confidence": float(probs[explained_class].item()),
            "class_scores": {
                label: float(probs[index].item())
                for index, label in enumerate(self.label_names)
            },
            "baseline_score": baseline_score,
            "input_score": input_score,
            "score_difference": score_difference,
            "completeness_error": completeness_error,
            "normalised_completeness_error": normalised_completeness_error,
            "attribution_converged": (
                normalised_completeness_error <= convergence_tolerance
            ),
            "interaction_change": interaction_change,
            "interaction_converged": interaction_change <= interaction_tolerance,
            "explanation_valid": (
                normalised_completeness_error <= convergence_tolerance
                and interaction_change <= interaction_tolerance
            ),
            "integration_steps": used_interaction_steps,
            "attribution_steps": used_attribution_steps,
            "convergence_tolerance": convergence_tolerance,
            "interaction_tolerance": interaction_tolerance,
            "max_interaction_tokens": self.max_interaction_tokens,
            "text": text,
        }

    def save_plot(self, result: dict, save_path: str | Path) -> None:
        """Save an off-diagonal, content-word interaction heatmap."""
        words = result["words"]
        matrix = np.asarray(result["word_interaction_matrix"], dtype=float)
        keep = result["content_word_indices"]
        if not keep:
            return

        visible_tokens = [words[index] for index in keep]
        visible_matrix = matrix[np.ix_(keep, keep)]
        np.fill_diagonal(visible_matrix, np.nan)
        finite_values = np.abs(visible_matrix[np.isfinite(visible_matrix)])
        max_abs = (
            float(np.percentile(finite_values, 95))
            if finite_values.size
            else 1.0
        )
        max_abs = max(max_abs, 1e-12)

        size = max(6.5, min(14.0, len(visible_tokens) * 0.55))
        fig, ax = plt.subplots(figsize=(size, size))
        colour_map = plt.get_cmap("RdBu_r").copy()
        colour_map.set_bad("#e8e8e8")
        image = ax.imshow(
            visible_matrix,
            cmap=colour_map,
            vmin=-max_abs,
            vmax=max_abs,
            aspect="auto",
        )
        ax.set_xticks(range(len(visible_tokens)))
        ax.set_yticks(range(len(visible_tokens)))
        ax.set_xticklabels(visible_tokens, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(visible_tokens, fontsize=8)
        ax.set_title(
            "Integrated Hessians word-pair interactions — "
            f"{result['predicted_label']} ({result['confidence']:.1%})",
            fontsize=11,
        )
        fig.colorbar(image, ax=ax, label="Off-diagonal interaction score")
        fig.tight_layout()
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
