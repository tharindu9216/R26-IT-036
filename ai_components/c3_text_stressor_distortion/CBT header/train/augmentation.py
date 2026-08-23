"""Text augmentation utilities for CDT (cognitive distortion) classification training.

This module randomly replaces selected emotionally-loaded words with simple
predefined synonyms during training. This creates small variations of the
training text without changing the original dataset.

The purpose is to help the model learn different word choices with similar
meanings, improving generalization on unseen text. Reused unchanged from the
Stress header — the vocabulary (anxious, overwhelmed, hopeless, etc.) is
generic emotional-distress language that applies equally to patient posts
describing cognitive distortions, not specific to Reddit/Dreaddit.
"""

import random
import re


# Mapping of target words to interchangeable synonyms used for augmentation.
SYNONYM_MAP = {
    'afraid': ['scared', 'fearful'],
    'angry': ['upset', 'frustrated'],
    'anxious': ['worried', 'uneasy'],
    'awful': ['terrible', 'dreadful'],
    'bad': ['difficult', 'rough'],
    'confused': ['uncertain', 'unsure'],
    'depressed': ['down', 'low'],
    'difficult': ['hard', 'challenging'],
    'exhausted': ['drained', 'tired'],
    'fear': ['worry', 'anxiety'],
    'hard': ['difficult', 'challenging'],
    'help': ['support', 'assist'],
    'hopeless': ['discouraged', 'desperate'],
    'issue': ['problem', 'concern'],
    'lonely': ['isolated', 'alone'],
    'nervous': ['anxious', 'uneasy'],
    'overwhelmed': ['stressed', 'burdened'],
    'panic': ['fear', 'alarm'],
    'problem': ['issue', 'concern'],
    'sad': ['unhappy', 'down'],
    'scared': ['afraid', 'fearful'],
    'stress': ['pressure', 'strain'],
    'stressed': ['overwhelmed', 'strained'],
    'struggle': ['difficulty', 'challenge'],
    'terrible': ['awful', 'dreadful'],
    'tired': ['exhausted', 'drained'],
    'worried': ['anxious', 'concerned'],
}


def _match_case(source, replacement):
    """Match replacement casing to the original token.

    Rules:
    - If the source token is fully uppercase, return uppercase replacement.
    - If the source token is title-cased, capitalize replacement.
    - Otherwise, keep replacement lowercase as provided.

    Args:
        source (str): Original token from input text.
        replacement (str): Candidate synonym to insert.

    Returns:
        str: Replacement token with casing aligned to source.
    """
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement.capitalize()
    return replacement


def synonym_replace(text, probability=0.15, max_replacements=1):
    """Randomly replace eligible words with synonyms.

    The function tokenizes text into words, punctuation, and whitespace so the
    original spacing/punctuation structure is preserved after augmentation.
    Replacement is applied only with the given probability and only to words
    present in `SYNONYM_MAP`.

    Case is preserved via `_match_case`, which matters for Longformer's
    cased pipeline (text_for_longformer keeps original capitalization) just
    as much as it does for the lowercased BERT/MentalBERT/DeBERTa-v3 columns.

    Args:
        text (str): Input sentence/document.
        probability (float, optional): Chance to apply augmentation for the
            given text. Defaults to 0.15.
        max_replacements (int, optional): Maximum number of word substitutions
            in a single text. Defaults to 1.

    Returns:
        str: Augmented text, or original text when augmentation is skipped or
        no eligible words are found.
    """
    if not text or random.random() > probability:
        return text

    tokens = re.findall(r"\w+|[^\w\s]|\s+", text)
    candidate_indices = [
        i for i, token in enumerate(tokens)
        if token.lower() in SYNONYM_MAP
    ]
    if not candidate_indices:
        return text

    random.shuffle(candidate_indices)
    replacements = 0
    for idx in candidate_indices:
        token = tokens[idx]
        replacement = random.choice(SYNONYM_MAP[token.lower()])
        tokens[idx] = _match_case(token, replacement)
        replacements += 1
        if replacements >= max_replacements:
            break

    return ''.join(tokens)
