"""Model-guided counterfactual text explanations for the Stress Header."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

import numpy as np
import torch

from decision import select_predicted_class

class StressCounterfactual:
    """
    Find minimal lexical edits that move a stress prediction to another class.

    The search is model-agnostic: candidates are evaluated in batches and a
    beam search returns the first (fewest-edit) candidates that flip the class.
    Returned text explains model sensitivity and is not psychological advice.
    """

    _NEUTRAL_PAIRS = (
        ("overwhelmed", "calm"),
        ("stressed", "relaxed"),
        ("anxious", "calm"),
        ("worried", "confident"),
        ("worrying", "coping"),
        ("hopeless", "hopeful"),
        ("exhausted", "rested"),
        ("failing", "succeeding"),
        ("fail", "succeed"),
        ("impossible", "manageable"),
        ("panic", "calm"),
        ("terrible", "manageable"),
    )
    _LOW_INFORMATION = {
        "a", "an", "the", "i", "me", "my", "you", "your", "he", "she",
        "it", "we", "they", "is", "am", "are", "was", "were", "be", "been",
        "to", "of", "in", "on", "at", "for", "and", "or", "but", "with",
        "this", "that",
    }

    def __init__(
        self,
        model_or_pipeline,
        tokenizer=None,
        label_names: list[str] | dict | None = None,
        device: str | None = None,
        max_length: int = 192,
        prediction_batch_size: int = 8,
        decision_threshold: float | None = None,
    ):
        self.model_or_pipeline = model_or_pipeline
        self.tokenizer = tokenizer or getattr(model_or_pipeline, "tokenizer", None)
        if self.tokenizer is None:
            raise ValueError("A tokenizer is required for counterfactual search.")
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

    @staticmethod
    def _normalise_label_names(label_names: list[str] | dict | None) -> list[str]:
        if label_names is None:
            return ["Not Stressed", "Stressed"]
        if isinstance(label_names, dict):
            return [label_names[str(i)] for i in range(len(label_names))]
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

    def _predict(self, texts: list[str]) -> np.ndarray:
        probability_batches = []
        for start in range(0, len(texts), self.prediction_batch_size):
            batch = texts[start : start + self.prediction_batch_size]
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
                logits_1a, _ = self.model_or_pipeline(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                probability_batches.append(
                    torch.softmax(logits_1a, dim=1).detach().cpu().numpy()
                )
        if not probability_batches:
            return np.empty((0, len(self.label_names)), dtype=float)
        return np.concatenate(probability_batches, axis=0)

    @staticmethod
    def _replace_span(text: str, start: int, end: int, replacement: str) -> str:
        updated = text[:start] + replacement + text[end:]
        updated = re.sub(r"\s+([,.;!?])", r"\1", updated)
        updated = re.sub(r"\s{2,}", " ", updated)
        return updated.strip()

    @staticmethod
    def _preserve_case(original: str, replacement: str) -> str:
        if original.isupper():
            return replacement.upper()
        if original[:1].isupper():
            return replacement.capitalize()
        return replacement

    def _candidate_edits(self, text: str, target_class: int) -> list[dict]:
        target_is_stressed = target_class == 1
        pairs = (
            [(neutral, stressed) for stressed, neutral in self._NEUTRAL_PAIRS]
            if target_is_stressed
            else list(self._NEUTRAL_PAIRS)
        )
        phrase_pairs = (
            [("can manage", "can not manage"), ("able to", "unable to")]
            if target_is_stressed
            else [("can not", "can"), ("unable to", "able to"), ("no hope", "hope")]
        )

        candidates = []
        seen = {text}
        for source, replacement in phrase_pairs:
            for match in re.finditer(rf"\b{re.escape(source)}\b", text, re.IGNORECASE):
                rendered = self._preserve_case(match.group(0), replacement)
                candidate = self._replace_span(text, match.start(), match.end(), rendered)
                if candidate and candidate not in seen:
                    seen.add(candidate)
                    candidates.append(
                        {
                            "text": candidate,
                            "operation": "replace",
                            "original": match.group(0),
                            "replacement": rendered,
                        }
                    )

        replacement_map = dict(pairs)
        for match in re.finditer(r"\b[A-Za-z][A-Za-z'-]*\b", text):
            original = match.group(0)
            replacement = replacement_map.get(original.lower())
            if replacement:
                rendered = self._preserve_case(original, replacement)
                candidate = self._replace_span(text, match.start(), match.end(), rendered)
                if candidate and candidate not in seen:
                    seen.add(candidate)
                    candidates.append(
                        {
                            "text": candidate,
                            "operation": "replace",
                            "original": original,
                            "replacement": rendered,
                        }
                    )

        # Deletion provides coverage when a word is not in the conservative
        # replacement lexicon. It is ranked below replacement candidates.
        for match in re.finditer(r"\b[A-Za-z][A-Za-z'-]*\b", text):
            original = match.group(0)
            if original.lower() in self._LOW_INFORMATION:
                continue
            candidate = self._replace_span(text, match.start(), match.end(), "")
            if candidate and candidate not in seen:
                seen.add(candidate)
                candidates.append(
                    {
                        "text": candidate,
                        "operation": "delete",
                        "original": original,
                        "replacement": "",
                    }
                )
        return candidates

    @staticmethod
    def _similarity(original: str, candidate: str) -> float:
        return SequenceMatcher(None, original.lower(), candidate.lower()).ratio()

    def explain(
        self,
        text: str,
        target_class: int | None = None,
        max_edits: int = 3,
        beam_width: int = 12,
        num_counterfactuals: int = 3,
        max_candidates_per_step: int = 256,
    ) -> dict:
        if max_edits < 1:
            raise ValueError("max_edits must be at least 1.")
        original_probs = self._predict([text])[0]
        original_class = select_predicted_class(
            original_probs, self.decision_threshold)
        if target_class is None:
            if len(self.label_names) != 2:
                raise ValueError("target_class is required for non-binary classifiers.")
            target_class = 1 - original_class
        if not 0 <= target_class < len(self.label_names):
            raise ValueError(f"target_class must be in [0, {len(self.label_names) - 1}].")
        if target_class == original_class:
            raise ValueError("target_class must differ from the model's original prediction.")

        beam = [{"text": text, "edits": []}]
        visited = {text}
        best_attempts = []
        counterfactuals = []

        for depth in range(1, max_edits + 1):
            generated = []
            for state in beam:
                for edit in self._candidate_edits(state["text"], target_class):
                    if edit["text"] in visited:
                        continue
                    visited.add(edit["text"])
                    generated.append(
                        {
                            "text": edit["text"],
                            "edits": state["edits"]
                            + [
                                {
                                    "operation": edit["operation"],
                                    "original": edit["original"],
                                    "replacement": edit["replacement"],
                                }
                            ],
                        }
                    )
                    if len(generated) >= max_candidates_per_step:
                        break
                if len(generated) >= max_candidates_per_step:
                    break
            if not generated:
                break

            probs = self._predict([item["text"] for item in generated])
            scored = []
            for state, row in zip(generated, probs):
                predicted_class = select_predicted_class(
                    row, self.decision_threshold)
                similarity = self._similarity(text, state["text"])
                deletion_count = sum(
                    edit["operation"] == "delete" for edit in state["edits"]
                )
                rank_score = (
                    float(row[target_class])
                    + 0.15 * similarity
                    - 0.03 * deletion_count
                )
                scored.append(
                    {
                        **state,
                        "num_edits": depth,
                        "predicted_class": predicted_class,
                        "predicted_label": self.label_names[predicted_class],
                        "confidence": float(row[predicted_class]),
                        "target_probability": float(row[target_class]),
                        "similarity": similarity,
                        "flipped": predicted_class == target_class,
                        "_rank_score": rank_score,
                    }
                )
            scored.sort(key=lambda item: item["_rank_score"], reverse=True)
            best_attempts = scored[:num_counterfactuals]
            counterfactuals = [
                item for item in scored if item["flipped"]
            ][:num_counterfactuals]
            if counterfactuals:
                break
            beam = scored[:beam_width]

        selected = counterfactuals or best_attempts
        for item in selected:
            item.pop("_rank_score", None)

        return {
            "text": text,
            "predicted_class": original_class,
            "predicted_label": self.label_names[original_class],
            "confidence": float(original_probs[original_class]),
            "class_scores": {
                label: float(original_probs[index])
                for index, label in enumerate(self.label_names)
            },
            "target_class": target_class,
            "target_label": self.label_names[target_class],
            "counterfactual_found": bool(counterfactuals),
            "counterfactuals": selected,
            "best_counterfactual": selected[0] if selected else None,
            "max_edits": max_edits,
            "disclaimer": (
                "Counterfactual text describes model sensitivity only; it is not "
                "a clinical conclusion or advice to change how distress is expressed."
            ),
        }
