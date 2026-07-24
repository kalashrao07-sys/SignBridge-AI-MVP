r"""
SignBridge AI — Sentence Builder
=================================
Turns a raw sign-buffer phrase (space-joined words from gesture.js's
SIGN_TO_WORD, e.g. "I WATER" or "HELP DOCTOR") into a grammatical
English sentence.

WHY THIS REPLACES THE OLD GRAMMAR_RULES IN app.py
---------------------------------------------------
The old GRAMMAR_RULES were a flat list of ~22 regex patterns, several of
which require the literal word "need" to appear in the phrase (e.g.
r"\bI\s+water\s+need\b"). No sign in gesture.js's entire 60-word
vocabulary produces the word "NEED" — there is no such gesture. Those
patterns were unreachable dead code: real sign buffers almost never
matched anything, fell through to the generic "capitalize + add period"
fallback, and got handed to Google Translate as ungrammatical fragments
like "Hello Food Water." — which is why translations came out awkward.
That was a translation-quality bug, but the actual defect was upstream.

This module fixes the actual defect: it's built directly against the
real, closed 60-word vocabulary (see VOCABULARY below, extracted from
gesture.js's SIGN_MAP + TWO_HAND_MAP), classifies each word into a small
number of grammatical roles, and assembles a sentence from a handful of
general templates — so it actually fires on realistic sign buffers
instead of requiring an exact, hand-written pattern per phrase.

If gesture.js's vocabulary changes, update WORD_CLASSES below to match —
there's a self-check at the bottom (see verify_coverage()) that flags
any word in VOCABULARY with no assigned class.
"""

from __future__ import annotations

import re
from typing import List, Optional

# The complete set of words gesture.js can actually produce (32 single-hand
# + 28 two-hand signs). Keep in sync with static/js/gesture.js.
VOCABULARY = {
    "ALLERGY", "ANGRY", "BANDAGE", "BETTER", "CALL", "CHILD", "COME",
    "CONFUSED", "DOCTOR", "DOWN", "EMERGENCY", "FAMILY", "FATHER", "FOOD",
    "FOUR", "FRIEND", "GO", "GOOD", "HAPPY", "HELLO", "HELP", "HOME",
    "HOSPITAL", "I", "INJURY", "LEFT", "LESS", "LOVE", "MEDICINE", "MORE",
    "MOTHER", "NO", "NURSE", "PAIN", "PEACE", "PINKY", "PLEASE",
    "PRESSURE", "RIGHT", "SAD", "SCARED", "SCHOOL", "SICK", "SORRY",
    "STOP", "TEMPERATURE", "THANK_YOU", "THREE", "TIRED", "TODAY",
    "TOGETHER", "TOILET", "TOMORROW", "UP", "WAIT", "WATER", "WE",
    "WORSE", "YES", "YESTERDAY",
}

# ── Grammatical roles for every word above ─────────────────────────────
PRONOUNS      = {"I", "WE"}
# Things you can grammatically "need" (a resource / place / person)
NEED_NOUNS    = {"WATER", "FOOD", "DOCTOR", "NURSE", "HOSPITAL",
                  "MEDICINE", "BANDAGE", "TOILET", "HOME", "HELP"}
PEOPLE_NOUNS  = {"DOCTOR", "NURSE", "FAMILY", "FRIEND", "MOTHER",
                  "FATHER", "CHILD"}
PLACE_NOUNS   = {"HOSPITAL", "HOME", "SCHOOL"}
# States/feelings expressed as "I am/feel ___"
STATE_WORDS   = {"PAIN": "in pain", "SICK": "sick", "TIRED": "tired",
                  "BETTER": "better", "WORSE": "worse", "SCARED": "scared",
                  "ANGRY": "angry", "HAPPY": "happy", "SAD": "sad",
                  "CONFUSED": "confused", "GOOD": "good"}
ACTION_VERBS  = {"CALL": "call", "STOP": "stop", "GO": "go", "COME": "come",
                  "WAIT": "wait"}
MEDICAL_NOUNS = {"INJURY": "an injury", "ALLERGY": "an allergy",
                  "TEMPERATURE": "a fever", "PRESSURE": "high blood pressure"}
GREETINGS     = {"HELLO": "Hello.", "PLEASE": "Please.", "SORRY": "I am sorry.",
                  "YES": "Yes.", "NO": "No.", "THANK_YOU": "Thank you.",
                  "PEACE": "Peace.", "LOVE": "I love you.",
                  "TOGETHER": "Let's go together."}
TIME_WORDS    = {"YESTERDAY": "yesterday", "TODAY": "today", "TOMORROW": "tomorrow"}
DIRECTION_WORDS = {"UP": "up", "DOWN": "down", "LEFT": "left", "RIGHT": "right"}
QUANTIFIERS   = {"MORE": "more", "LESS": "less"}
NUMBERS       = {"THREE": "three", "FOUR": "four", "PINKY": "five"}
# Handled as a special case at the top of build_sentence(), but still
# needs to be registered here so verify_coverage() doesn't flag it.
EMERGENCY_WORDS = {"EMERGENCY"}


def _article(word_lower: str) -> str:
    return "an" if word_lower[0] in "aeiou" else "a"


def build_sentence(words: List[str]) -> str:
    """
    Given an ordered list of sign words (e.g. ["I", "PAIN"] or
    ["HELP", "DOCTOR"]), return a grammatical English sentence.

    Falls back to a plain capitalized join if no template matches —
    same safety net the old rule_correct() had, so this never crashes
    or produces empty output on an unrecognized combination.
    """
    if not words:
        return ""

    tokens = [w.upper() for w in words]
    has = lambda s: s in tokens
    token_set = set(tokens)

    # ── EMERGENCY takes priority over everything else ──
    if has("EMERGENCY"):
        rest = [t for t in tokens if t != "EMERGENCY"]
        if has("DOCTOR") or has("HOSPITAL"):
            return "Emergency! I need a doctor immediately."
        if has("CALL"):
            return "Emergency! Please call for help immediately."
        return "Emergency! I need help immediately."

    # ── PRONOUN + NEED_NOUN(s), possibly ALSO combined with a state or
    #    medical condition (e.g. ["DOCTOR","I","PAIN"]) — combine both
    #    rather than letting one branch silently drop the other's info,
    #    since dropping a symptom is a real problem for this app. ──
    pronoun_present = has("I") or has("WE")
    need_nouns_present = [t for t in tokens if t in NEED_NOUNS]
    state_present = [t for t in tokens if t in STATE_WORDS]
    medical_present = [t for t in tokens if t in MEDICAL_NOUNS]

    if pronoun_present and need_nouns_present:
        subj_mid = "I" if not has("WE") else "we"  # "I" is always capitalized; "we" only at sentence start
        verb_be = "are" if has("WE") else "am"
        nouns_en = [_needy_noun_phrase(n) for n in need_nouns_present]
        need_clause = f"{subj_mid} need {' and '.join(nouns_en)}"

        extra_clauses = []
        if state_present:
            phrase = STATE_WORDS[state_present[0]]
            extra_clauses.append(
                f"{subj_mid} {verb_be} in pain" if phrase == "in pain"
                else f"{subj_mid} {verb_be} feeling {phrase}"
            )
        if medical_present:
            extra_clauses.append(f"{subj_mid} have {MEDICAL_NOUNS[medical_present[0]]}")

        if extra_clauses:
            sentence = f"{extra_clauses[0]} and {need_clause}."  # e.g. "I am in pain and I need a doctor."
        else:
            sentence = f"{need_clause}."
        return sentence[0].upper() + sentence[1:]

    # ── PRONOUN + STATE (feeling/condition), no need-noun: "I am in pain." ──
    if pronoun_present and state_present:
        subject = "We" if has("WE") else "I"
        verb = "are" if has("WE") else "am"
        phrase = STATE_WORDS[state_present[0]]
        connector = "in" if phrase == "in pain" else "feeling"
        if phrase == "in pain":
            return f"{subject} {verb} in pain."
        return f"{subject} {verb} feeling {phrase}."

    # ── PRONOUN + MEDICAL_NOUN, no need-noun: "I have an injury." ──
    if pronoun_present and medical_present:
        subject = "We" if has("WE") else "I"
        verb = "have"
        return f"{subject} {verb} {MEDICAL_NOUNS[medical_present[0]]}."

    # ── HELP + a person/place noun, no pronoun required (signed toward someone) ──
    if has("HELP"):
        people = [t for t in tokens if t in PEOPLE_NOUNS]
        if people:
            noun = _needy_noun_phrase(people[0])
            return f"Please help — I need {noun}."
        return "Please help me."

    # ── Standalone need-noun with no pronoun: treat as a request ──
    if need_nouns_present and not pronoun_present:
        nouns_en = [_needy_noun_phrase(n) for n in need_nouns_present]
        return f"I need {' and '.join(nouns_en)}." if len(nouns_en) > 1 else f"I need {nouns_en[0]}."

    # ── Standalone state word, no pronoun ──
    if state_present and not pronoun_present:
        phrase = STATE_WORDS[state_present[0]]
        return "I am in pain." if phrase == "in pain" else f"I am feeling {phrase}."

    # ── ACTION_VERBS (imperative-style): "Please call." / "Go." ──
    verbs_present = [t for t in tokens if t in ACTION_VERBS]
    if verbs_present and len(tokens) == 1:
        verb = ACTION_VERBS[verbs_present[0]]
        return f"Please {verb}." if verb in ("call", "wait", "stop") else f"{verb.capitalize()}."

    # ── Single-word greetings/social words ──
    if len(tokens) == 1 and tokens[0] in GREETINGS:
        return GREETINGS[tokens[0]]

    # ── Fallback: no template matched. Same safety net as the old
    #    rule_correct() — capitalize + add punctuation rather than fail. ──
    text = " ".join(_display_word(t) for t in tokens)
    text = re.sub(r"\s{2,}", " ", text).strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text[0].upper() + text[1:] if text else text


def _needy_noun_phrase(word: str) -> str:
    display = _display_word(word)
    if word == "HELP":
        return "help"
    if word in PEOPLE_NOUNS or word in {"MEDICINE", "BANDAGE"}:
        return f"{_article(display)} {display}"
    return display  # WATER, FOOD, HOSPITAL, TOILET, HOME read fine bare


def _display_word(word: str) -> str:
    return word.replace("_", " ").lower()


def verify_coverage() -> List[str]:
    """
    Returns any vocabulary word with no assigned grammatical role — run
    this after editing gesture.js's vocabulary to catch drift. An
    uncovered word still won't crash build_sentence() (it hits the
    fallback), it just won't get special-cased phrasing.
    """
    classified = (
        PRONOUNS | NEED_NOUNS | PEOPLE_NOUNS | PLACE_NOUNS
        | set(STATE_WORDS) | set(ACTION_VERBS) | set(MEDICAL_NOUNS)
        | set(GREETINGS) | set(TIME_WORDS) | set(DIRECTION_WORDS)
        | set(QUANTIFIERS) | set(NUMBERS) | EMERGENCY_WORDS
    )
    return sorted(VOCABULARY - classified)
