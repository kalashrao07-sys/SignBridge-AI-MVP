"""
SignBridge AI — Sign Vocabulary (reverse lookup)
=================================================
Given recognised speech text, returns the ordered list of words that
have a real animated sign available (i.e. exist as a key in
static/data/sign_animations.json — real recorded landmark sequences
from the Kaggle Google-ISLR dataset, not fabricated poses).

This list MUST match the keys in sign_animations.json exactly. It's
hardcoded here (rather than read from the 2.8MB JSON at import time)
because the vocabulary is fixed once extraction is done — if you
re-run extraction with a different SELECTED_SIGNS list, update this
file to match.
"""

from __future__ import annotations

import re
from typing import List

# The 80 words actually present in sign_animations.json.
VOCABULARY = {
    "airplane", "alligator", "aunt", "awake", "balloon", "because", "bee",
    "bird", "blow", "brother", "brown", "bye", "cat", "closet", "cow",
    "cry", "doll", "donkey", "drink", "dry", "duck", "ear", "eye", "farm",
    "find", "fireman", "first", "flower", "food", "frog", "gift",
    "glasswindow", "goose", "gum", "hear", "hello", "home", "horse",
    "icecream", "kiss", "kitty", "lion", "lips", "listen", "look", "loud",
    "mad", "make", "man", "mom", "mouse", "mouth", "nap", "napkin", "nuts",
    "old", "orange", "owl", "pajamas", "pen", "pencil", "penny", "pizza",
    "pretend", "pretty", "sad", "shhh", "sleepy", "sun", "talk", "taste",
    "think", "tiger", "tooth", "toothbrush", "uncle", "up", "wake", "who",
    "yesterday",
}

# Common synonyms mapped to a word that's actually IN the vocabulary above.
# Every value here must be a real key or matching will silently fail.
SYNONYMS = {
    "hi": "hello", "goodbye": "bye", "hungry": "food", "eat": "food",
    "thirsty": "drink", "sleep": "sleepy", "tired": "sleepy",
    "angry": "mad", "mother": "mom", "watch": "look", "see": "look",
    "speak": "talk", "kitten": "kitty",
}

_WORD_RE = re.compile(r"[a-zA-Z']+")


def text_to_sign_sequence(text: str) -> List[str]:
    """
    Tokenize `text` and return an ordered list of words that have a
    real animated sign. Words with no match are skipped (the full
    text still reaches the user via the existing Smart Display).
    """
    if not text:
        return []

    tokens = _WORD_RE.findall(text.lower())
    sequence = []
    for word in tokens:
        canonical = word if word in VOCABULARY else SYNONYMS.get(word)
        if canonical:
            sequence.append(canonical)
    return sequence


def sign_vocabulary_size() -> int:
    return len(VOCABULARY)