"""
SignBridge AI — Flask Backend
Self-contained Knowledge Engine + TTS + Translation + Emergency Detection
+ session-based authentication (signup/login/logout).

No external knowledge API. No API key. No network call for the knowledge
layer. Fully owned, fully auditable, fully offline-capable core logic.
"""

import os, re, io, base64
from functools import lru_cache
from typing import Optional

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_login import login_required
from gtts import gTTS
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

from knowledge_base import lookup_knowledge, knowledge_base_size
from sign_vocabulary import text_to_sign_sequence_detailed, sign_vocabulary_size
from sentence_builder import build_sentence
from auth import auth_bp, init_auth

load_dotenv()

app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "signbridge-secret-2024")

init_auth(app)
app.register_blueprint(auth_bp)

# Requests longer than this are rejected before hitting gTTS / translate,
# which have their own undocumented limits and get slow/unreliable on
# very long input.
MAX_INPUT_LEN = 500

# ─── Emergency keyword set ────────────────────────────────────────────
EMERGENCY_KEYWORDS = {
    "help", "emergency", "pain", "chest", "heart", "hospital",
    "doctor", "ambulance", "fire", "accident", "blood", "dying",
    "unconscious", "breathe", "breathing", "attack", "hurt",
    "fall", "fainted", "seizure", "stroke", "cancer", "poison",
    "urgent", "critical", "severe", "immediately",
}

# NOTE: the old GRAMMAR_RULES regex list lived here. It was replaced by
# sentence_builder.py because most of those patterns required the
# literal word "need" to appear in the phrase, but no sign in
# gesture.js's vocabulary produces the word "NEED" — they were
# unreachable dead code. sentence_builder.py is built directly against
# the real 60-word sign vocabulary instead.

FILLER_WORDS = [
    r"\bif\s+you\s+don'?t\s+mind\b", r"\bcould\s+you\s+possibly\b",
    r"\bi\s+was\s+wondering\s+if\b",  r"\bum+\b", r"\buh+\b",
    r"\byou\s+know\b", r"\bi\s+mean\b", r"\bactually\b",
    r"\bbasically\b",  r"\bjust\b",    r"\bkind\s+of\b",
    r"\blike\b",       r"\bso\b",
]


# ═══════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def is_emergency(text: str) -> bool:
    words = set(re.sub(r"[^\w\s]", "", text.lower()).split())
    if words & EMERGENCY_KEYWORDS:
        return True
    for phrase in ["chest pain", "can't breathe", "heart attack",
                   "need help", "call ambulance", "call 108", "call 112"]:
        if phrase in text.lower():
            return True
    return False


def rule_correct(text: str) -> str:
    """
    Turn a raw sign-buffer phrase (space-joined words from gesture.js,
    e.g. "I WATER" or "DOCTOR I PAIN") into a grammatical English
    sentence, using sentence_builder.py's vocabulary-aware templates.
    """
    words = text.strip().split()
    return build_sentence(words) if words else text


def rule_simplify(text: str) -> str:
    result = text
    for filler in FILLER_WORDS:
        result = re.sub(filler, "", result, flags=re.IGNORECASE)
    result = re.sub(r"\s{2,}", " ", result).strip()
    if result and result[-1] not in ".!?":
        result += "."
    return result[0].upper() + result[1:] if result else text


def text_to_speech_b64(text: str, lang: str = "en") -> Optional[str]:
    LANG_MAP = {"en": "en", "hi": "hi", "kn": "kn"}
    try:
        tts = gTTS(text=text, lang=LANG_MAP.get(lang, "en"), slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception:
        return None


@lru_cache(maxsize=256)
def translate_text(text: str, target: str) -> str:
    LANG_MAP = {"en": "en", "hi": "hi", "kn": "kn"}
    code = LANG_MAP.get(target, "en")
    if code == "en":
        return text
    try:
        return GoogleTranslator(source="auto", target=code).translate(text)
    except Exception:
        return text


@lru_cache(maxsize=256)
def translate_to_english(text: str, source: str) -> str:
    """
    Reverse of translate_text(): translates recognized speech FROM the
    selected language INTO English, so downstream logic that only
    understands English (is_emergency's keyword set, the Knowledge
    Engine, text_to_sign_sequence's 60-word vocabulary) can actually
    work when the user spoke Hindi/Kannada. Without this, non-English
    speech was being matched against English-only word lists and
    silently produced nothing.
    """
    LANG_MAP = {"en": "en", "hi": "hi", "kn": "kn"}
    code = LANG_MAP.get(source, "en")
    if code == "en":
        return text
    try:
        return GoogleTranslator(source=code, target="en").translate(text)
    except Exception:
        return text  # fail safe: fall back to the raw (untranslated) text


# ═══════════════════════════════════════════════════════════════════
#  PAGE ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.route("/")
@login_required
def home():
    return render_template("home.html")


@app.route("/sign-to-speech")
@login_required
def sign_to_speech_page():
    return render_template("sign_to_speech.html", kb_active=True)


@app.route("/speech-to-sign")
@login_required
def speech_to_sign_page():
    return render_template("speech_to_sign.html", kb_active=True)


@app.route("/calibrate")
@login_required
def calibration_page():
    return render_template("calibration.html")


# ═══════════════════════════════════════════════════════════════════
#  API ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.route("/api/sign/process", methods=["POST"])
@login_required
def process_sign():
    """
    Direction 1: Sign phrase → grammar correction → Knowledge Engine → TTS

    Body: { "phrase": "HELP CHEST PAIN", "lang": "en" }
    """
    data   = request.get_json(silent=True) or {}
    phrase = data.get("phrase", "").strip()
    lang   = data.get("lang", "en")

    if not phrase:
        return jsonify({"error": "No phrase provided"}), 400
    if len(phrase) > MAX_INPUT_LEN:
        return jsonify({"error": f"Phrase too long (max {MAX_INPUT_LEN} characters)"}), 400

    corrected = rule_correct(phrase)
    emergency = is_emergency(phrase)
    insight   = lookup_knowledge(phrase)

    translated = None
    if lang != "en":
        translated = translate_text(corrected, lang)

    speak_text = translated if translated else corrected
    audio_b64  = text_to_speech_b64(speak_text, lang)

    return jsonify({
        "original":    phrase,
        "corrected":   corrected,
        "translated":  translated,
        "emergency":   emergency,
        "kb_topic":    insight.get("topic"),
        "kb_response": insight.get("response"),
        "kb_category": insight.get("category"),
        "kb_success":  insight.get("success", False),
        "method":      "knowledge_base" if insight.get("success") else "rules",
        "audio_b64":   audio_b64,
        "lang":        lang,
    })


@app.route("/api/speech/process", methods=["POST"])
@login_required
def process_speech():
    """
    Direction 2: Speech transcript → simplification → Knowledge Engine → display
                 AND transcript → sign sequence (voice → visual sign language)

    Body: { "text": "Please go to the hospital immediately", "lang": "en" }
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    lang = data.get("lang", "en")

    if not text:
        return jsonify({"error": "No text provided"}), 400
    if len(text) > MAX_INPUT_LEN:
        return jsonify({"error": f"Text too long (max {MAX_INPUT_LEN} characters)"}), 400

    # Everything downstream (emergency keyword matching, the Knowledge
    # Engine, sign-sequence matching) only understands English. If the
    # user spoke Hindi/Kannada, `text` is in that language — translate
    # to English FIRST so all of that logic actually has something it
    # can match against, instead of silently finding nothing.
    text_en = translate_to_english(text, lang) if lang != "en" else text

    emergency = is_emergency(text_en)
    insight   = lookup_knowledge(text_en)
    simplified = rule_simplify(text_en)

    if emergency:
        words     = text_en.upper().split()
        key_words = [w for w in words
                     if re.sub(r"[^\w]", "", w.lower()) in EMERGENCY_KEYWORDS]
        display = " ".join(key_words[:3]) + " — CALL 108 🚨" if key_words else "EMERGENCY 🚨"
    else:
        display = simplified

    # Translate the (now properly English) simplified sentence back to
    # the selected language for display — a single sentence-level call,
    # same principle as the Sign→Speech fix: translate whole sentences,
    # not fragments.
    translated = None
    if lang != "en":
        translated = translate_text(simplified, lang)

    sign_sequence, sign_unmatched = text_to_sign_sequence_detailed(text_en)

    return jsonify({
        "original":      text,           # raw recognized speech, in the spoken language
        "original_en":   text_en if lang != "en" else None,  # translated-to-English text actually used for matching, for transparency/debugging
        "simplified":    simplified,
        "display":       display,
        "emergency":     emergency,
        "translated":    translated,
        "kb_topic":      insight.get("topic"),
        "kb_response":   insight.get("response"),
        "kb_category":   insight.get("category"),
        "kb_success":    insight.get("success", False),
        "method":        "knowledge_base" if insight.get("success") else "rules",
        "sign_sequence": sign_sequence,
        "sign_unmatched": sign_unmatched,
        "lang":          lang,
    })


@app.route("/api/tts", methods=["POST"])
@login_required
def tts():
    data  = request.get_json(silent=True) or {}
    text  = data.get("text", "")
    lang  = data.get("lang", "en")
    if len(text) > MAX_INPUT_LEN:
        return jsonify({"audio_b64": None, "error": f"Text too long (max {MAX_INPUT_LEN} characters)"}), 400
    audio = text_to_speech_b64(text, lang)
    return jsonify({"audio_b64": audio, "error": None if audio else "TTS failed"})


@app.route("/api/health")
def health():
    # Deliberately left unauthenticated — health checks (uptime monitors,
    # Render's own probe) shouldn't need a logged-in session.
    return jsonify({
        "status":          "ok",
        "project":         "SignBridge AI",
        "version":         "3.2",
        "knowledge_base":  "active",
        "kb_entries":      knowledge_base_size(),
        "signs_supported": sign_vocabulary_size(),
        "languages":       ["en", "hi", "kn"],
        "external_deps":   ["gTTS (TTS audio)", "Google Translate (translation)"],
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"\n🌉 SignBridge AI v3.2  →  http://localhost:{port}")
    print(f"   Knowledge Engine : ✅ {knowledge_base_size()} entries, fully local, no API key")
    print(f"   Sign vocabulary  : {sign_vocabulary_size()} signs (bidirectional)")
    print(f"   Languages        : English, Hindi, Kannada")
    print(f"   Auth             : session-based (Flask-Login + SQLite)\n")
    app.run(host="0.0.0.0", port=port,
            debug=os.getenv("FLASK_DEBUG", "true") == "true")
