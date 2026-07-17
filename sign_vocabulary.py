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
from typing import List, Optional

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

# Very small, deliberately conservative suffix-stripping so simple
# grammatical variants (plurals, -ing/-ed forms) still match a base word
# that IS in the vocabulary — e.g. "cats" -> "cat", "looking" -> "look".
# This is NOT semantic matching: it never maps one concept to a different
# one. A word like "doctor" or "headache" has no animation and correctly
# stays unmatched — there's no honest way to stem or synonym our way to
# a sign that was never recorded. Closing that gap means adding more
# words to sign_animations.json (see roadmap: WLASL / INCLUDE / AI4Bharat),
# not cleverer text matching here.
_SUFFIXES = ["ing", "ed", "es", "s"]


def _stem_candidates(word: str) -> List[str]:
    candidates = [word]
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            candidates.append(word[: -len(suffix)])
    return candidates


def _resolve(word: str) -> Optional[str]:
    for candidate in _stem_candidates(word):
        if candidate in VOCABULARY:
            return candidate
        if candidate in SYNONYMS:
            return SYNONYMS[candidate]
    return None


def text_to_sign_sequence(text: str) -> List[str]:
    """
    Tokenize `text` and return an ordered list of words that have a
    real animated sign. Words with no match (including no match after
    simple stemming) are skipped — the full text still reaches the user
    via the existing Smart Display. Use text_to_sign_sequence_detailed()
    if you also need to know which words were skipped.
    """
    return text_to_sign_sequence_detailed(text)[0]


def text_to_sign_sequence_detailed(text: str) -> "tuple[List[str], List[str]]":
    """
    Same as text_to_sign_sequence(), but also returns the list of
    tokens that had no animation available, in case a caller wants to
    surface that (e.g. "no sign for: doctor, headache") rather than
    silently dropping them.
    """
    if not text:
        return [], []

    tokens = _WORD_RE.findall(text.lower())
    matched, unmatched = [], []
    for word in tokens:
        canonical = _resolve(word)
        if canonical:
            matched.append(canonical)
        else:
            unmatched.append(word)
    return matched, unmatched


def sign_vocabulary_size() -> int:
    return len(VOCABULARY)