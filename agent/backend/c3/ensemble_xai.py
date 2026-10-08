"""Align and weight per-member token attributions at word level."""

from __future__ import annotations

import re
from collections import defaultdict, deque

import numpy as np

from xai import TokenAttribution


SPECIAL_TOKENS = {
    "[CLS]", "[SEP]", "[PAD]", "[UNK]", "<s>", "</s>", "<pad>", "<unk>",
}


def _member_words(items: list[TokenAttribution]) -> list[tuple[str, float]]:
    words: list[tuple[str, float]] = []
    current = ""
    score = 0.0
    for item in items:
        token = item.token
        if token in SPECIAL_TOKENS:
            continue
        continuation = token.startswith("##")
        sentencepiece_start = token.startswith("▁")
        cleaned = token.removeprefix("##").removeprefix("▁").strip()
        if not cleaned:
            continue
        if continuation and current:
            current += cleaned
            score += float(item.score)
            continue
        if sentencepiece_start or not current:
            if current:
                words.append((current, score))
            current = cleaned
            score = float(item.score)
        else:
            # Punctuation/non-prefixed tokens are kept as separate readable
            # units unless they are a BERT continuation.
            if current:
                words.append((current, score))
            current = cleaned
            score = float(item.score)
    if current:
        words.append((current, score))
    return words


def _key(word: str) -> str:
    return re.sub(r"^\W+|\W+$", "", word.casefold())


def combine_member_attributions(
    member_items: list[tuple[str, float, list[TokenAttribution]]],
) -> list[TokenAttribution]:
    """Normalize each member then combine using its deployment weight."""
    if not member_items:
        return []
    member_words = [
        (name, weight, _member_words(items))
        for name, weight, items in member_items
    ]
    canonical = [word for word, _ in member_words[0][2] if _key(word)]
    total_weight = sum(weight for _, weight, _ in member_words)
    combined = np.zeros(len(canonical), dtype=float)
    for _name, weight, rows in member_words:
        values = np.asarray([score for _, score in rows], dtype=float)
        scale = float(np.max(np.abs(values))) if values.size else 0.0
        if scale > 1e-12:
            values = values / scale
        available: dict[str, deque[float]] = defaultdict(deque)
        for (word, _), value in zip(rows, values, strict=True):
            if _key(word):
                available[_key(word)].append(float(value))
        aligned = [
            available[_key(word)].popleft() if available[_key(word)] else 0.0
            for word in canonical
        ]
        combined += (weight / total_weight) * np.asarray(aligned)
    return [
        TokenAttribution(token=word, score=float(score))
        for word, score in zip(canonical, combined, strict=True)
    ]
