
import random
import re


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
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement.capitalize()
    return replacement


def synonym_replace(text, probability=0.15, max_replacements=1):
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
