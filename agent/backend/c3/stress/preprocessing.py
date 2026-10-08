"""Inference preprocessing reproduced from the Dreaddit training notebook."""

from __future__ import annotations

import re


CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "don't": "do not",
    "doesn't": "does not", "didn't": "did not", "wouldn't": "would not",
    "couldn't": "could not", "shouldn't": "should not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "mustn't": "must not", "needn't": "need not",
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "you're": "you are", "you've": "you have", "you'll": "you will",
    "he's": "he is", "she's": "she is", "it's": "it is",
    "we're": "we are", "we've": "we have", "we'll": "we will",
    "they're": "they are", "they've": "they have", "they'll": "they will",
    "that's": "that is", "there's": "there is", "what's": "what is",
    "who's": "who is", "let's": "let us", "how's": "how is",
    "rn": "right now", "tbh": "to be honest", "idk": "i do not know",
    "imo": "in my opinion", "imho": "in my humble opinion",
    "omg": "oh my god", "bc": "because", "b/c": "because",
    "tho": "though", "gonna": "going to", "wanna": "want to",
    "gotta": "got to", "kinda": "kind of", "sorta": "sort of",
    "ngl": "not going to lie", "irl": "in real life",
    "smh": "shaking my head", "nvm": "never mind",
    "afaik": "as far as i know", "fwiw": "for what it is worth",
    "tldr": "too long did not read", "asap": "as soon as possible",
    "lol": "", "lmao": "", "lmfao": "", "rofl": "",
    "wtf": "", "omfg": "", "smfh": "",
}


def expand_contractions(text: str) -> str:
    for source, replacement in CONTRACTIONS.items():
        text = re.sub(rf"\b{re.escape(source)}\b", replacement, text)
    return text


def remove_noise(text: str) -> str:
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"/?r/[A-Za-z0-9_]+|/?u/[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    return re.sub(r"&[a-z]+;|&#?[a-z0-9]+;|<[^>]+>", " ", text)


def base_clean(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = expand_contractions(text.lower())
    text = remove_noise(text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"([!?.,;:]){2,}", r"\1", text)
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,2}(.*?)_{1,2}", r"\1", text)
    text = re.sub(r"~~(.*?)~~", r"\1", text)
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^>+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^a-zA-Z0-9\s.!?,'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_for_model(text: str, model_name: str) -> str:
    cleaned = base_clean(text)
    if model_name == "DeBERTa-v3":
        cleaned = re.sub(r"\b[a-z]\b", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
    elif model_name != "BERT":
        raise ValueError(f"Unsupported stress transformer model: {model_name}")
    if not cleaned:
        raise ValueError("Text is empty after stress preprocessing")
    return cleaned
