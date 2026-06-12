"""
SignBridge AI — Flask Backend
Wolfram Alpha semantic intelligence + TTS + Translation
Python 3.14 compatible (no mediapipe — CV runs in the browser)
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
app.secret_key = os.getenv("FLASK_SECRET_KEY", "signbridge-secret")

WOLFRAM_APP_ID = os.getenv("WOLFRAM_APP_ID", "")

# ─── Emergency keywords ───────────────────────────────────────────────
EMERGENCY_KEYWORDS = {
    "help", "emergency", "pain", "chest", "heart", "hospital",
    "doctor", "ambulance", "fire", "accident", "blood", "dying",
    "unconscious", "breathe", "breathing", "attack", "hurt",
    "fall", "fainted", "seizure", "stroke", "cancer", "poison",
}

# ─── ISL/ASL grammar correction rules ────────────────────────────────
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
    (r"\bI\s+deaf\b",                 "I am deaf."),
    (r"\bI\s+mute\b",                 "I cannot speak."),
    (r"\bneed\s+doctor\b",            "need a doctor"),
    (r"\bneed\s+ambulance\b",         "need an ambulance"),
    (r"\bwater\s+please\b",           "I would like water, please."),
    (r"\bfood\s+please\b",            "I would like food, please."),
    (r"\bhelp\s+please\b",            "Please help me."),
    (r"\bthank\s+you\b",              "Thank you."),
    (r"\bsorry\b",                    "I am sorry."),
    (r"\bI\s+understand\s+not\b",     "I do not understand."),
    (r"\brepeat\s+please\b",          "Please repeat that."),
]

FILLER_WORDS = [
    r"\bif\s+you\s+don'?t\s+mind\b", r"\bcould\s+you\s+possibly\b",
    r"\bi\s+was\s+wondering\s+if\b",  r"\bum+\b", r"\buh+\b",
    r"\byou\s+know\b", r"\bi\s+mean\b", r"\bactually\b",
    r"\bbasically\b",  r"\bjust\b",    r"\bkind\s+of\b",
]

# ─── Helpers ──────────────────────────────────────────────────────────

def is_emergency(text: str) -> bool:
    words = set(re.sub(r"[^\w\s]", "", text.lower()).split())
    if words & EMERGENCY_KEYWORDS:
        return True
    for phrase in ["chest pain", "can't breathe", "heart attack",
                   "need help", "call ambulance", "call 108"]:
        if phrase in text.lower():
            return True
    return False


def rule_correct(text: str) -> str:
    result = text.strip()
    for pattern, replacement in GRAMMAR_RULES:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    result = re.sub(r"\s{2,}", " ", result).strip()
    if result and result[-1] not in ".!?":
        result += "."
    return result[0].upper() + result[1:] if result else result


def rule_simplify(text: str) -> str:
    result = text
    for filler in FILLER_WORDS:
        result = re.sub(filler, "", result, flags=re.IGNORECASE)
    result = re.sub(r"\s{2,}", " ", result).strip()
    if result and result[-1] not in ".!?":
        result += "."
    return result[0].upper() + result[1:] if result else text


def wolfram_query(query: str) -> dict:
    """
    Call Wolfram Alpha Simple API.
    Returns {"result": str, "success": bool, "query": str}
    """
    if not WOLFRAM_APP_ID:
        return {"result": None, "success": False, "query": query,
                "error": "No Wolfram App ID configured"}
    try:
        url = "https://api.wolframalpha.com/v1/result"
        resp = requests.get(url, params={"i": query, "appid": WOLFRAM_APP_ID},
                            timeout=6)
        if resp.status_code == 200 and resp.text.strip():
            result = resp.text.strip()
            if "not understand" in result.lower() or len(result) > 300:
                return {"result": None, "success": False,
                        "query": query, "error": "No usable result"}
            return {"result": result, "success": True, "query": query}
        return {"result": None, "success": False,
                "query": query, "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"result": None, "success": False, "query": query, "error": str(e)}


TOPIC_MAP = {
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
    "dizzy":       ("vertigo",                  "medical"),
    "fever":       ("fever",                    "medical"),
    "fall":        ("trauma injury",            "medical"),
    "hospital":    ("emergency room",           "medical"),
    "doctor":      ("general physician",        "medical"),
    "water":       ("daily water intake",       "health"),
    "food":        ("nutrition",                "health"),
    "toilet":      ("dehydration",              "health"),
    "deaf":        ("deafness",                 "info"),
    "call":        ("emergency number India",   "info"),
    "hello":       ("sign language",            "info"),
    "good":        ("health",                   "info"),
    "please":      ("communication",            "info"),
}

def wolfram_insight(phrase: str) -> dict:
    phrase_lower = phrase.lower()
    matched_query = None
    category = "general"

    for keyword, (query, cat) in TOPIC_MAP.items():
        if keyword in phrase_lower:
            matched_query = query
            category = cat
            break

    if not matched_query:
        return {"insight": None, "query": None, "category": None, "success": False}

    if not WOLFRAM_APP_ID:
        return {"insight": None, "query": matched_query,
                "category": category, "success": False, "error": "No App ID"}

    # Try result API first (more permissive for short terms)
    try:
        resp = requests.get(
            "https://api.wolframalpha.com/v1/result",
            params={"i": matched_query, "appid": WOLFRAM_APP_ID},
            timeout=6
        )
        if resp.status_code == 200 and resp.text.strip():
            return {"insight": resp.text.strip(), "query": matched_query,
                    "category": category, "success": True}
    except Exception:
        pass

    # Fallback to spoken API
    try:
        resp = requests.get(
            "https://api.wolframalpha.com/v1/spoken",
            params={"i": matched_query, "appid": WOLFRAM_APP_ID},
            timeout=6
        )
        if resp.status_code == 200 and resp.text.strip():
            return {"insight": resp.text.strip(), "query": matched_query,
                    "category": category, "success": True}
    except Exception:
        pass

    return {"insight": None, "query": matched_query,
            "category": category, "success": False, "error": "HTTP 501"}


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
    return render_template("index.html",
                           wolfram_active=bool(WOLFRAM_APP_ID))


@app.route("/api/sign/process", methods=["POST"])
def process_sign():
    data   = request.get_json(silent=True) or {}
    phrase = data.get("phrase", "").strip()
    lang   = data.get("lang", "en")

    if not phrase:
        return jsonify({"error": "No phrase"}), 400

    emergency = is_emergency(phrase)
    corrected = rule_correct(phrase)          # grammar always via rules

    insight   = wolfram_insight(phrase)       # wolfram for knowledge only
    wf_context = insight.get("insight")

    translated = None
    if lang != "en":
        translated = translate_text(corrected, lang)

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
    Direction 2: Transcribed speech → Wolfram simplify → display
    Body: { "text": "Please go to hospital", "lang": "en" }
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    lang = data.get("lang", "en")

    if not text:
        return jsonify({"error": "No text"}), 400

    emergency = is_emergency(text)
    insight = wolfram_insight(text)

    # Wolfram simplification for visual display
    if emergency:
        query = f'What is the simplified emergency message for: "{text}"'
    else:
        query = f'Simplify this sentence for a deaf person: "{text}"'

    wf = wolfram_query(query)
    if wf["success"] and wf["result"]:
        simplified = wf["result"].strip().strip('"')
        method = "wolfram"
    else:
        simplified = rule_simplify(text)
        method = "rules"

    # Emergency override display
    if emergency:
        words = text.upper().split()
        key_words = [w for w in words if w.lower() in EMERGENCY_KEYWORDS]
        display = " ".join(key_words[:3]) + " — CALL 108 🚨" if key_words else "EMERGENCY 🚨"
    else:
        display = simplified

    # Translation
    translated = None
    if lang != "en":
        translated = translate_text(simplified, lang)

    return jsonify({
        "original":         text,
        "simplified":       simplified,
        "display":          display,
        "emergency":        emergency,
        "translated":       translated,
        "wolfram_query":    wf.get("query") or insight.get("query"),
        "wolfram_response": wf.get("result") or insight.get("insight") or wf.get("error") or "No result",
        "wolfram_method":   method,
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
        "status": "ok",
        "wolfram": "active" if WOLFRAM_APP_ID else "no key — using rule-based fallback",
        "project": "SignBridge AI"
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"\n🌉 SignBridge AI  →  http://localhost:{port}")
    print(f"   Wolfram : {'✅ Active' if WOLFRAM_APP_ID else '⚠️  No key — add WOLFRAM_APP_ID to .env'}\n")
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT", 5000)))
