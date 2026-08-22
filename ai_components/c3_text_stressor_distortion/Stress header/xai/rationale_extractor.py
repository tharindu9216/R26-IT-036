from __future__ import annotations

import html
import re
from dataclasses import dataclass


SPECIAL_TOKENS = {
    "<s>",
    "</s>",
    "<pad>",
    "[CLS]",
    "[SEP]",
    "[PAD]",
    "[UNK]",
    "<unk>",
}

LOW_INFORMATION_WORDS = {
    "i",
    "me",
    "my",
    "mine",
    "you",
    "your",
    "he",
    "she",
    "it",
    "we",
    "they",
    "a",
    "an",
    "the",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "and",
    "or",
    "but",
    "with",
    "that",
    "this",
    "everything",
    "lately",
    "is",
    "am",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "can",
    "could",
    "should",
}

PHRASE_STARTERS = {
    "feel",
    "feeling",
    "felt",
    "keep",
    "keeps",
    "kept",
    "worry",
    "worried",
    "worrying",
}


@dataclass
class WordAttribution:
    text: str
    score: float
    token_start: int
    token_end: int


def _is_special_token(token: str) -> bool:
    return token in SPECIAL_TOKENS or token.strip() == ""


def _is_punctuation(text: str) -> bool:
    return bool(re.fullmatch(r"[^\w\s]+", text))


def _word_key(text: str) -> str:
    return text.lower().strip(" ,.!?;:")


def _clean_token(token: str) -> tuple[str, bool]:
    """
    Return cleaned text and whether the token starts a new word.

    Handles common HF tokenizer markers:
    - SentencePiece/DeBERTa: "_word" or "▁word"
    - RoBERTa/GPT-2: "Ġword"
    - BERT WordPiece: "##piece"
    - SHAP text masker segments: " word"
    """
    if token and token[0].isspace():
        return token.strip(), True
    if token.startswith(("▁", "Ġ", "_")):
        return token[1:], True
    if token.startswith("##"):
        return token[2:], False
    return token, False


def tokens_to_words(tokens: list[str], scores: list[float]) -> list[WordAttribution]:
    words: list[WordAttribution] = []
    has_explicit_boundaries = any(
        token.startswith(("▁", "Ġ", "_")) or (token and token[0].isspace())
        for token in tokens
        if not _is_special_token(token)
    )

    for index, (token, score) in enumerate(zip(tokens, scores)):
        if _is_special_token(token):
            continue

        text, starts_new = _clean_token(token)
        if not text:
            continue
        if not starts_new and not token.startswith("##") and not has_explicit_boundaries:
            starts_new = True

        if _is_punctuation(text):
            if words:
                words[-1].text += text
                words[-1].score = max(words[-1].score, float(score))
                words[-1].token_end = index
            continue

        if starts_new or not words:
            words.append(
                WordAttribution(
                    text=text,
                    score=float(score),
                    token_start=index,
                    token_end=index,
                )
            )
            continue

        words[-1].text += text
        words[-1].score = max(words[-1].score, float(score))
        words[-1].token_end = index

    return words


def extract_rationale_spans(
    tokens: list[str],
    scores: list[float],
    relative_threshold: float = 0.35,
    min_threshold: float = 0.0,
    max_gap: int = 0,
    max_spans: int = 5,
    min_words: int = 1,
    max_words_per_span: int = 4,
) -> list[dict]:
    """
    Convert token-level attributions into readable rationale spans.

    Positive scores are treated as supporting the explained/predicted class.
    The threshold is relative to the strongest positive word score, so this
    works for normalized IG scores and raw SHAP values.
    """
    words = tokens_to_words(tokens, scores)
    positive_scores = [word.score for word in words if word.score > 0]
    if not positive_scores:
        return []

    threshold = max(max(positive_scores) * relative_threshold, min_threshold)
    selected = [
        index
        for index, word in enumerate(words)
        if word.score >= threshold and _word_key(word.text) not in LOW_INFORMATION_WORDS
    ]
    if not selected:
        return []

    groups: list[list[int]] = []
    current = [selected[0]]
    for index in selected[1:]:
        crosses_sentence = words[current[-1]].text.endswith((".", "!", "?"))
        if not crosses_sentence and index - current[-1] <= max_gap + 1:
            current.append(index)
        else:
            groups.append(current)
            current = [index]
    groups.append(current)

    spans = []
    for group in groups:
        start = group[0]
        end = group[-1]
        start, end = _expand_to_readable_phrase(words, start, end)
        included = list(range(start, end + 1)) if max_gap > 0 else group
        if max_gap == 0:
            included = list(range(start, end + 1))

        span_words = [words[index] for index in included]
        span_words = _trim_low_information_edges(span_words)
        if len(span_words) > max_words_per_span:
            span_words = _keep_strongest_window(span_words, max_words_per_span)
            span_words = _trim_low_information_edges(span_words)
        if len(span_words) < min_words:
            continue
        if all(_word_key(word.text) in LOW_INFORMATION_WORDS for word in span_words):
            continue

        text = " ".join(word.text for word in span_words)
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        text = text.strip(" ,.!?;:")
        if not text:
            continue
        score = sum(max(0.0, word.score) for word in span_words) / len(span_words)
        spans.append(
            {
                "text": text,
                "score": score,
                "word_start": start,
                "word_end": end,
                "token_start": span_words[0].token_start,
                "token_end": span_words[-1].token_end,
            }
        )

    spans = _deduplicate_spans(spans)
    spans.sort(key=lambda item: item["score"], reverse=True)
    return spans[:max_spans]


def _deduplicate_spans(spans: list[dict]) -> list[dict]:
    best_by_text: dict[str, dict] = {}
    for span in spans:
        key = re.sub(r"\s+", " ", span["text"].lower()).strip()
        current = best_by_text.get(key)
        if current is None or span["score"] > current["score"]:
            best_by_text[key] = span
    return list(best_by_text.values())


def _expand_to_readable_phrase(
    words: list[WordAttribution],
    start: int,
    end: int,
) -> tuple[int, int]:
    """
    Expand terse single-token rationales into short readable phrases.

    Example: if "keep" is selected and "worrying" has positive attribution,
    return "keep worrying" instead of the less useful standalone "keep".
    """
    if not words:
        return start, end

    while start > 0:
        previous = words[start - 1]
        if previous.score <= 0 or previous.text.lower() in LOW_INFORMATION_WORDS:
            break
        start -= 1

    while end + 1 < len(words):
        current = _word_key(words[end].text)
        following = words[end + 1]
        if following.score <= 0 or _word_key(following.text) in LOW_INFORMATION_WORDS:
            break
        if current in PHRASE_STARTERS or end == start:
            end += 1
            continue
        break

    return start, end


def _trim_low_information_edges(words: list[WordAttribution]) -> list[WordAttribution]:
    start = 0
    end = len(words)

    while start < end and _word_key(words[start].text) in LOW_INFORMATION_WORDS:
        start += 1
    while end > start and _word_key(words[end - 1].text) in LOW_INFORMATION_WORDS:
        end -= 1

    return words[start:end]


def _keep_strongest_window(
    words: list[WordAttribution],
    max_words: int,
) -> list[WordAttribution]:
    if len(words) <= max_words:
        return words

    best_start = 0
    best_score = float("-inf")
    for start in range(0, len(words) - max_words + 1):
        window = words[start:start + max_words]
        score = sum(max(0.0, word.score) for word in window)
        if score > best_score:
            best_score = score
            best_start = start
    return words[best_start:best_start + max_words]


def highlight_rationales(text: str, rationales: list[dict]) -> str:
    """
    Build simple HTML with extracted rationale spans highlighted.

    This operates on the preprocessed text used by the model. If an exact span
    cannot be found, it is skipped instead of risking a wrong highlight.
    """
    replacements = []

    for rationale in rationales:
        span = re.sub(r"\s+", " ", rationale["text"]).strip()
        if not span:
            continue
        pattern = r"\s+".join(re.escape(part) for part in span.split())
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is None:
            continue
        replacements.append((match.start(), match.end(), span, rationale["score"]))

    replacements.sort(key=lambda item: item[0])
    merged = []
    last_end = -1
    for item in replacements:
        if item[0] >= last_end:
            merged.append(item)
            last_end = item[1]

    if not merged:
        return html.escape(text)

    output = []
    cursor = 0
    for start, end, span, score in merged:
        output.append(html.escape(text[cursor:start]))
        output.append(
            '<mark title="rationale score: '
            f'{score:.3f}" style="background:#ffe08a;color:#111827;'
            'padding:0.1rem 0.18rem;border-radius:4px;">'
            f"{html.escape(text[start:end])}</mark>"
        )
        cursor = end
    output.append(html.escape(text[cursor:]))
    return "".join(output)
