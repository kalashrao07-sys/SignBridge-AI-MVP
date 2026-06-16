"""
SignBridge AI — Flask Backend
Wolfram Alpha knowledge layer + TTS + Translation + Emergency Detection
Python 3.8+ compatible (MediaPipe runs in the browser — no Python CV needed)

OSC AI BUILD 1.0 — AI for Social Impact
"""

import os, re, io, base64, requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from gtts import gTTS
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "signbridge-secret-2024")

WOLFRAM_APP_ID = os.getenv("WOLFRAM_APP_ID", "")

# ─── Emergency keyword set ────────────────────────────────────────────
EMERGENCY_KEYWORDS = {
    "help", "emergency", "pain", "chest", "heart", "hospital",
    "doctor", "ambulance", "fire", "accident", "blood", "dying",
    "unconscious", "breathe", "breathing", "attack", "hurt",
    "fall", "fainted", "seizure", "stroke", "cancer", "poison",
    "urgent", "critical", "severe", "immediately",
}

# ─── ISL/ASL grammar correction rules ────────────────────────────────
# ISL uses SOV order (Subject-Object-Verb); these rules convert to SVO English
GRAMMAR_RULES = [
    (r"\bI\s+water\s+need\b",         "I need water."),
    (r"\bI\s+food\s+need\b",          "I need food."),
    (r"\bI\s+doctor\s+need\b",        "I need a doctor."),
    (r"\bI\s+help\s+need\b",          "I need help."),
    (r"\bI\s+toilet\s+need\b",        "I need to use the restroom."),
    (r"\bI\s+hospital\s+need\b",      "I need to go to the hospital."),
    (r"\bdoctor\s+need\s+pain\b",     "I need a doctor. I am in pain."),
    (r"\bhelp\s+chest\s+pain\b",      "Help! I have chest pain."),
    (r"\bpain\s+chest\b",             "I have chest pain."),
    (r"\bhelp\s+please\b",            "Please help me."),
    (r"\bI\s+deaf\b",                 "I am deaf."),
    (r"\bI\s+mute\b",                 "I cannot speak."),
    (r"\bneed\s+doctor\b",            "need a doctor"),
    (r"\bneed\s+ambulance\b",         "need an ambulance"),
    (r"\bwater\s+please\b",           "I would like water, please."),
    (r"\bfood\s+please\b",            "I would like food, please."),
    (r"\bthank\s+you\b",              "Thank you."),
    (r"\bsorry\b",                    "I am sorry."),
    (r"\bI\s+understand\s+not\b",     "I do not understand."),
    (r"\brepeat\s+please\b",          "Please repeat that."),
    (r"\bI\s+dizzy\b",                "I feel dizzy."),
    (r"\bI\s+fever\b",                "I have a fever."),
    (r"\bI\s+pain\b",                 "I am in pain."),
]

# Filler words removed from hearing person's speech for display clarity
FILLER_WORDS = [
    r"\bif\s+you\s+don'?t\s+mind\b", r"\bcould\s+you\s+possibly\b",
    r"\bi\s+was\s+wondering\s+if\b",  r"\bum+\b", r"\buh+\b",
    r"\byou\s+know\b", r"\bi\s+mean\b", r"\bactually\b",
    r"\bbasically\b",  r"\bjust\b",    r"\bkind\s+of\b",
    r"\blike\b",       r"\bso\b",
]

# ─── Wolfram topic map ────────────────────────────────────────────────
# Maps sign keywords → (wolfram_query, category)
# Queries use single medical terms / short factual phrases — what Wolfram excels at
TOPIC_MAP = {
    # Emergency category
    "help":        ("medical emergency",        "emergency"),
    "chest":       ("chest pain",               "emergency"),
    "pain":        ("acute pain",               "emergency"),
    "heart":       ("heart attack",             "emergency"),
    "breathe":     ("dyspnea",                  "emergency"),
    "blood":       ("hemorrhage",               "emergency"),
    "seizure":     ("seizure",                  "emergency"),
    "stroke":      ("stroke",                   "emergency"),
    "unconscious": ("syncope",                  "emergency"),
    "poison":      ("poisoning",                "emergency"),
    "ambulance":   ("emergency services India", "emergency"),
    "attack":      ("cardiac arrest",           "emergency"),
    # Medical category
    "dizzy":       ("vertigo",                  "medical"),
    "fever":       ("fever",                    "medical"),
    "fall":        ("trauma injury",            "medical"),
    "hospital":    ("emergency room",           "medical"),
    "doctor":      ("general physician",        "medical"),
    "fainted":     ("fainting causes",          "medical"),
    "cancer":      ("cancer",                   "medical"),
    # Health category
    "water":       ("daily water intake",       "health"),
    "food":        ("nutrition",                "health"),
    "toilet":      ("dehydration",              "health"),
    # Info category
    "deaf":        ("deafness",                 "info"),
    "call":        ("emergency number India",   "info"),
    "hello":       ("sign language",            "info"),
    "good":        ("health",                   "info"),
    "please":      ("communication",            "info"),
    "fire":        ("fire safety",              "emergency"),
}


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
    """Apply ISL/ASL SOV → SVO grammar rules."""
    result = text.strip()
    for pattern, replacement in GRAMMAR_RULES:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    result = re.sub(r"\s{2,}", " ", result).strip()
    if result and result[-1] not in ".!?":
        result += "."
    return result[0].upper() + result[1:] if result else result


def rule_simplify(text: str) -> str:
    """Remove filler words from speech for deaf display clarity."""
    result = text
    for filler in FILLER_WORDS:
        result = re.sub(filler, "", result, flags=re.IGNORECASE)
    result = re.sub(r"\s{2,}", " ", result).strip()
    if result and result[-1] not in ".!?":
        result += "."
    return result[0].upper() + result[1:] if result else text


def wolfram_query(query: str) -> dict:
    """
    Core Wolfram Alpha API call.
    Tries v1/result first, then v1/spoken as fallback.
    Returns { result, success, query, error }
    """
    if not WOLFRAM_APP_ID:
        return {"result": None, "success": False, "query": query,
                "error": "No Wolfram App ID"}

    # Primary: Short Answers API (best for single medical terms)
    try:
        resp = requests.get(
            "https://api.wolframalpha.com/v1/result",
            params={"i": query, "appid": WOLFRAM_APP_ID},
            timeout=6
        )
        if resp.status_code == 200 and resp.text.strip():
            result = resp.text.strip()
            # Filter out Wolfram's "does not understand" responses
            if "not understand" not in result.lower() and len(result) <= 400:
                return {"result": result, "success": True, "query": query}
    except Exception:
        pass

    # Fallback: Spoken API (natural language output)
    try:
        resp = requests.get(
            "https://api.wolframalpha.com/v1/spoken",
            params={"i": query, "appid": WOLFRAM_APP_ID},
            timeout=6
        )
        if resp.status_code == 200 and resp.text.strip():
            return {"result": resp.text.strip(), "success": True, "query": query}
    except Exception:
        pass

    return {"result": None, "success": False, "query": query,
            "error": f"HTTP {resp.status_code if 'resp' in dir() else 'timeout'}"}


def wolfram_insight(phrase: str) -> dict:
    """
    Extract a keyword from the sign phrase and query Wolfram
    for authoritative factual/medical context.

    Returns { insight, query, category, success }
    """
    phrase_lower = phrase.lower()
    matched_query = None
    category = "general"

    for keyword, (query, cat) in TOPIC_MAP.items():
        if keyword in phrase_lower:
            matched_query = query
            category = cat
            break

    if not matched_query:
        return {"insight": None, "query": None,
                "category": None, "success": False}

    wf = wolfram_query(matched_query)
    return {
        "insight":  wf.get("result"),
        "query":    matched_query,
        "category": category,
        "success":  wf["success"],
        "error":    wf.get("error"),
    }


def text_to_speech_b64(text: str, lang: str = "en") -> str | None:
    LANG_MAP = {"en": "en", "hi": "hi", "kn": "kn"}
    try:
        tts = gTTS(text=text, lang=LANG_MAP.get(lang, "en"), slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception:
        return None


def translate_text(text: str, target: str) -> str:
    LANG_MAP = {"en": "en", "hi": "hi", "kn": "kn"}
    code = LANG_MAP.get(target, "en")
    if code == "en":
        return text
    try:
        return GoogleTranslator(source="auto", target=code).translate(text)
    except Exception:
        return text


# ═══════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html", wolfram_active=bool(WOLFRAM_APP_ID))


@app.route("/api/sign/process", methods=["POST"])
def process_sign():
    """
    Direction 1: Sign phrase → grammar correction → Wolfram knowledge → TTS

    Flow:
      1. rule_correct()    — ISL SOV→SVO grammar fix (fast, reliable)
      2. wolfram_insight() — Wolfram medical/factual knowledge layer
      3. is_emergency()    — keyword-based emergency flag
      4. text_to_speech()  — gTTS audio generation
      5. translate_text()  — Hindi/Kannada if requested

    Body: { "phrase": "HELP CHEST PAIN", "lang": "en" }
    """
    data   = request.get_json(silent=True) or {}
    phrase = data.get("phrase", "").strip()
    lang   = data.get("lang", "en")

    if not phrase:
        return jsonify({"error": "No phrase provided"}), 400

    # 1. Grammar correction (always rule-based — fast and reliable)
    corrected = rule_correct(phrase)

    # 2. Emergency detection
    emergency = is_emergency(phrase)

    # 3. Wolfram knowledge layer
    insight    = wolfram_insight(phrase)
    wf_context = insight.get("insight")

    # 4. Translation
    translated = None
    if lang != "en":
        translated = translate_text(corrected, lang)

    # 5. TTS
    speak_text = translated if translated else corrected
    audio_b64  = text_to_speech_b64(speak_text, lang)

    return jsonify({
        "original":         phrase,
        "corrected":        corrected,
        "translated":       translated,
        "emergency":        emergency,
        "wf_context":       wf_context,
        "wolfram_query":    insight.get("query"),
        "wolfram_response": insight.get("insight") or insight.get("error") or "No result",
        "wolfram_method":   "wolfram" if insight.get("success") else "rules",
        "wolfram_category": insight.get("category"),
        "wolfram_success":  insight.get("success", False),
        "audio_b64":        audio_b64,
        "lang":             lang,
    })


@app.route("/api/speech/process", methods=["POST"])
def process_speech():
    """
    Direction 2: Speech transcript → simplification → Wolfram context → display

    Flow:
      1. is_emergency()    — detect urgent phrases
      2. wolfram_insight() — Wolfram knowledge context
      3. rule_simplify()   — remove filler words for clean display
      4. emergency display — extract key words + "CALL 108 🚨"
      5. translate_text()  — Hindi/Kannada if requested

    Body: { "text": "Please go to the hospital immediately", "lang": "en" }
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    lang = data.get("lang", "en")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # 1. Emergency detection
    emergency = is_emergency(text)

    # 2. Wolfram knowledge layer
    insight = wolfram_insight(text)

    # 3. Text simplification
    simplified = rule_simplify(text)
    method     = "rules"

    # 4. Emergency display override
    if emergency:
        words     = text.upper().split()
        key_words = [w for w in words
                     if re.sub(r"[^\w]", "", w.lower()) in EMERGENCY_KEYWORDS]
        display = " ".join(key_words[:3]) + " — CALL 108 🚨" if key_words else "EMERGENCY 🚨"
    else:
        display = simplified

    # 5. Translation
    translated = None
    if lang != "en":
        translated = translate_text(simplified, lang)

    return jsonify({
        "original":         text,
        "simplified":       simplified,
        "display":          display,
        "emergency":        emergency,
        "translated":       translated,
        "wolfram_query":    insight.get("query"),
        "wolfram_response": insight.get("insight") or insight.get("error") or "No result",
        "wolfram_method":   "wolfram" if insight.get("success") else "rules",
        "wolfram_category": insight.get("category"),
        "wolfram_success":  insight.get("success", False),
        "lang":             lang,
    })


@app.route("/api/tts", methods=["POST"])
def tts():
    data  = request.get_json(silent=True) or {}
    text  = data.get("text", "")
    lang  = data.get("lang", "en")
    audio = text_to_speech_b64(text, lang)
    return jsonify({"audio_b64": audio, "error": None if audio else "TTS failed"})


@app.route("/api/health")
def health():
    return jsonify({
        "status":  "ok",
        "wolfram": "active" if WOLFRAM_APP_ID else "no key — rule-based fallback active",
        "project": "SignBridge AI",
        "version": "2.0",
        "signs_supported": 12,
        "languages": ["en", "hi", "kn"],
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"\n🌉 SignBridge AI  →  http://localhost:{port}")
    print(f"   Wolfram : {'✅ Active' if WOLFRAM_APP_ID else '⚠️  No key — set WOLFRAM_APP_ID in .env'}")
    print(f"   Signs   : 12 gestures supported")
    print(f"   Langs   : English, Hindi, Kannada\n")
    app.run(host="0.0.0.0", port=port,
            debug=os.getenv("FLASK_DEBUG", "true") == "true")