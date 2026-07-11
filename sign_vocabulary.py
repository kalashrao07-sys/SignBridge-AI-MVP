"""
SignBridge AI — Sign Vocabulary (reverse lookup)
=================================================
Mirrors the sign vocabulary defined in static/js/gesture.js (SIGN_MAP /
SIGN_TO_WORD) so the backend can go the OTHER direction: given a piece of
recognised speech text, find which of it maps onto known signs and return
an ordered sequence the frontend can render as a visual "signed" strip.

Keep this list in sync with SIGN_MAP in gesture.js. If you add a new sign
on the frontend, add the matching entry here too.
"""

from __future__ import annotations

import re
from typing import List, Dict

# word (lowercase) -> sign info. Includes a couple of common synonyms so a
# spoken sentence has a reasonable chance of matching the small vocabulary.
SIGN_VOCABULARY = {
    "hello":  {"sign": "HELLO",  "emoji": "👋", "desc": "Open hand — Hello"},
    "hi":     {"sign": "HELLO",  "emoji": "👋", "desc": "Open hand — Hello"},
    "yes":    {"sign": "YES",    "emoji": "✊", "desc": "Fist — Yes"},
    "good":   {"sign": "GOOD",   "emoji": "👍", "desc": "Thumbs up — Good"},
    "fine":   {"sign": "GOOD",   "emoji": "👍", "desc": "Thumbs up — Good"},
    "i":      {"sign": "I",      "emoji": "👆", "desc": "Point up — I / Me"},
    "me":     {"sign": "I",      "emoji": "👆", "desc": "Point up — I / Me"},
    "please": {"sign": "PLEASE", "emoji": "🙏", "desc": "Four fingers — Please"},
    "water":  {"sign": "WATER",  "emoji": "💧", "desc": "W shape — Water"},
    "thirsty":{"sign": "WATER",  "emoji": "💧", "desc": "W shape — Water"},
    "help":   {"sign": "HELP",   "emoji": "🆘", "desc": "L shape — Help"},
    "call":   {"sign": "CALL",   "emoji": "🤙", "desc": "Y shape — Call"},
    "phone":  {"sign": "CALL",   "emoji": "🤙", "desc": "Y shape — Call"},
    "food":   {"sign": "FOOD",   "emoji": "🍽️", "desc": "Eat — Food"},
    "hungry": {"sign": "FOOD",   "emoji": "🍽️", "desc": "Eat — Food"},
    "eat":    {"sign": "FOOD",   "emoji": "🍽️", "desc": "Eat — Food"},
    "no":     {"sign": "NO",     "emoji": "🚫", "desc": "Refuse — No"},
    "pain":   {"sign": "PAIN",   "emoji": "😣", "desc": "Hurt — Pain"},
    "hurt":   {"sign": "PAIN",   "emoji": "😣", "desc": "Hurt — Pain"},
    "doctor": {"sign": "DOCTOR", "emoji": "👨‍⚕️", "desc": "Doctor"},
}

_WORD_RE = re.compile(r"[a-zA-Z']+")


def text_to_sign_sequence(text: str) -> List[Dict]:
    """
    Tokenize `text` and return an ordered list of matched signs.
    Words with no known sign are skipped (the caller still has the full
    text available via the existing Smart Display for anything unmatched).
    """
    if not text:
        return []

    tokens = _WORD_RE.findall(text.lower())
    sequence = []
    for word in tokens:
        entry = SIGN_VOCABULARY.get(word)
        if entry:
            sequence.append({"word": word, **entry})
    return sequence


def sign_vocabulary_size() -> int:
    return len({v["sign"] for v in SIGN_VOCABULARY.values()})
