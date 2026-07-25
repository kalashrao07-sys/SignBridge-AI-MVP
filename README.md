# SignBridge AI
### Two-Way Communication System for Deaf & Speech-Impaired Communities

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7?style=for-the-badge)](https://signbridge-ai-9rzh.onrender.com/)
[![GitHub](https://img.shields.io/badge/GitHub-Public-181717?style=for-the-badge)](https://github.com/kalashrao07-sys/SignBridge-AI-MVP)

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [The Knowledge Engine](#the-knowledge-engine)
- [Two Recognition Pipelines: Rule-Based + DL Model](#two-recognition-pipelines-rule-based--dl-model)
- [Setup Instructions](#setup-instructions)
- [API Reference](#api-reference)
- [Known Limitations](#known-limitations)
- [Roadmap (v2)](#roadmap-v2)
- [Project Structure](#project-structure)

---

## Problem Statement

466 million people worldwide have disabling hearing loss (WHO, 2023); over 63 million in India alone. Communication barriers show up acutely in hospitals, schools, emergencies, and public services, where professional interpreters are expensive, scarce, and rarely available on demand.

SignBridge AI is a browser-based, two-way communication bridge that needs no app install and no interpreter on standby.

---

## Solution Overview

```
DIRECTION 1 (Sign -> Speech)
Hand signs (camera) -> MediaPipe Hands -> Rule-based gesture classifier
   -> Sentence Builder -> Knowledge Engine (offline) -> TTS (gTTS) + optional translation

DIRECTION 2 (Speech -> Sign)
Speech (mic) -> Web Speech API -> [translate to English if needed]
   -> Knowledge Engine + emergency detection -> Keyword extraction
   -> Animated hand/arm skeleton (recorded landmark sequences)
```

Both directions run through the same **offline Knowledge Engine** — there is no external AI API, no API key, and no dependency on an internet-connected third-party service for the core logic. The only network calls in production are gTTS (text-to-speech audio) and Google Translate (for Hindi/Kannada), both of which degrade gracefully to English/no-audio if unavailable.

> **Note on project history:** earlier versions of this project used the Wolfram Alpha API as a knowledge layer. That dependency has been fully removed — Wolfram Alpha requires an API key, a network call per query, and is a black box that can't be audited or reviewed by a domain expert. It's been replaced by `knowledge_base.py`, a self-contained, auditable knowledge layer (see below). No Wolfram references remain anywhere in the codebase or deployment config.

---

## Architecture

```
+----------------------------- BROWSER --------------------------------+
|                                                                       |
|  MediaPipe Hands --> gesture.js (rule-based classifier, 60 signs)    |
|       |                     |                                        |
|       |                     +--> Phrase Buffer --> POST /api/sign/   |
|       |                     |                        process         |
|       |                                                               |
|       +--> sequence-model.js (DL model, DEBUG-ONLY)                  |
|                 |                                                     |
|                 +--> console.log only -- never touches the UI,       |
|                        the phrase buffer, or the Knowledge Engine    |
|                                                                       |
|  Web Speech API --> app.js --> POST /api/speech/process              |
|                                                                       |
|  sign-animation.js <-- sign_sequence (from /api/speech/process)      |
|     SignAnimationPlayer: Play / Pause / Replay / Stop                |
+-------------------------------+---------------------------------------+
                                 |
+----------------------------- FLASK BACKEND ---------------------------+
|                                                                        |
|  /api/sign/process                                                    |
|    sentence_builder.py  -> grammatical English from sign buffer       |
|    knowledge_base.py    -> offline Knowledge Engine lookup            |
|    gTTS                 -> audio                                      |
|    deep-translator      -> EN -> Hindi/Kannada (whole-sentence)       |
|                                                                        |
|  /api/speech/process                                                  |
|    deep-translator       -> non-English speech -> English (if needed) |
|    knowledge_base.py     -> offline Knowledge Engine lookup           |
|    sign_vocabulary.py    -> keyword extraction + sign matching        |
|                                                                        |
|  Flask-Login + Flask-SQLAlchemy (SQLite) -> session-based auth        |
+-------------------------------------------------------------------------+
```

---

## Key Features

### Sign -> Speech
| Feature | Details |
|---|---|
| Hand tracking | MediaPipe Hands, 21 landmarks/hand, up to 2 hands |
| Sign vocabulary | **60 signs** — 32 single-hand (all possible 5-finger-bitmask combinations) + 28 two-hand combinations |
| Sentence building | `sentence_builder.py` — rule-based, built directly against the real 60-word vocabulary, handles multi-concept buffers (e.g. `["DOCTOR","I","PAIN"]` -> *"I am in pain and I need a doctor."*) |
| Knowledge Engine | Offline lookup, 47 entries — medical/emergency guidance, social/family context |
| Voice output | gTTS — English, Hindi, Kannada |
| Translation | Whole-sentence (not word-by-word) English -> Hindi/Kannada |
| Emergency detection | Keyword + phrase-based, auto-flags urgent phrases |
| Calibration | Dedicated `/calibrate` page to record personal samples per sign (accounts for individual hand/camera/lighting variation) |

### Speech -> Sign
| Feature | Details |
|---|---|
| Speech recognition | Web Speech API, English/Hindi/Kannada input |
| Language handling | Non-English speech is translated to English *before* matching — the animation vocabulary is English-only, so this step is required for Hindi/Kannada input to produce any animation at all |
| Keyword extraction | Tokenizes the full sentence, keeps only words with a real recorded animation (with light stemming for plurals/verb forms), silently drops filler words — **not** whole-sentence exact matching |
| Sign animation | Real recorded hand/arm landmark sequences (not emoji, not fabricated poses) — **80-word vocabulary** (Kaggle Google-ISLR children's ASL subset, referred to internally as `ASL_80`) |
| Playback controls | Play/Resume, Pause, Stop — a played sequence can be rewatched without repeating the speech input |
| Transparency | If a sentence has words with no available animation, the UI says so explicitly (e.g. *"no animation for: doctor, urgently"*) rather than silently showing nothing |

### Knowledge Engine (both directions)
- Fully offline — no API key, no network call, no rate limit
- 47 hand-written, auditable entries covering emergency, medical, health, and social/informational topics
- Specificity-based keyword matching (a multi-word keyword like "blood pressure" always wins over a shorter, unrelated one like "blood")

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Frontend | HTML5, CSS3, vanilla JS (classic `<script>` tags, no bundler) | Zero build step, matches the rest of the codebase |
| Hand tracking | MediaPipe Hands (CDN) | Browser-native, no server-side ML needed for the live demo |
| Speech input | Web Speech API | Built into Chrome, no API key |
| Backend | Python 3.8+, Flask | Lightweight, Render-compatible |
| Knowledge layer | `knowledge_base.py` (self-contained) | Offline, auditable, no external dependency |
| TTS | gTTS | English/Hindi/Kannada |
| Translation | `deep-translator` (Google Translate) | Whole-sentence, bidirectional (EN <-> local language) |
| Auth | Flask-Login + Flask-SQLAlchemy + SQLite | Real accounts, intentionally simple for this stage (see `auth.py`'s scope note — this is **not** the JWT + PostgreSQL setup planned for v2) |
| Sign animation data | Recorded MediaPipe landmark sequences (`sign_animations.json`) | Real motion data, not fabricated |
| Deployment | Render.com | Free tier, HTTPS, auto-deploy from GitHub |

---

## The Knowledge Engine

Both `/api/sign/process` and `/api/speech/process` route recognized text through `knowledge_base.py` — a flat, hand-curated list of `{keywords, topic, response, category}` entries. No API key, no network call, no rate limit, fully readable and auditable in one file.

```python
{
    "keywords": ["chest", "heart attack"],
    "topic": "Chest Pain",
    "response": "Chest pain can be a sign of a serious heart-related emergency...",
    "category": "emergency",
}
```

**Production note (kept from the original design intent):** this content is compiled from general public-health and first-aid guidance for demonstration purposes. Before any real deployment (hospitals, schools, public use), a licensed medical professional should review and sign off on every entry — that review is exactly what owning this file, instead of calling a black-box third-party API, makes possible.

---

## Two Recognition Pipelines: Rule-Based + DL Model

Sign -> Speech actually runs **two independent pipelines simultaneously**, and this separation is intentional:

- **`gesture.js` (rule-based, 60-sign bitmask classifier) drives 100% of the user-facing output** — the detected-sign display, the phrase buffer, the Knowledge Engine call, the final text, and TTS. This is what makes the demo reliable.
- **A trained Bidirectional-GRU sequence model runs in parallel, entirely in the background.** It logs its own predictions (label + confidence) to the browser console only (`[DL model][debug] ...`) — it never writes to the phrase buffer, never touches the DOM, and has no path back into the user-facing flow.

This lets the DL model keep being monitored and improved without any risk of it destabilizing the live demo. See [Roadmap](#roadmap-v2) for what closing that gap looks like.

The DL model itself required real engineering to get running in the browser at all — TensorFlow.js's built-in `GRU` layer doesn't support the `reset_after=True` gate formulation the model was trained with (a genuine tfjs-layers limitation, not a bug in this app), so inference runs through a hand-written forward pass (`gru_sequence_model.js`) implementing the correct cuDNN-compatible equations directly on `tfjs-core` ops, loading the original trained weights unmodified.

---

## Setup Instructions

### Prerequisites
- Python 3.8+
- Chrome (for the Web Speech API)

### Local Setup
```bash
git clone https://github.com/kalashrao07-sys/SignBridge-AI-MVP.git
cd SignBridge-AI-MVP

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env          # set FLASK_SECRET_KEY

python app.py
```
Open `http://localhost:5000` in Chrome. No API keys required for the core app (Knowledge Engine, gesture recognition, sign animation are fully offline/local).

### Deploy to Render (Free)
1. Push to GitHub (public repo)
2. Render -> New -> Web Service -> connect the repo
3. Render reads `render.yaml` automatically (build/start commands + `FLASK_SECRET_KEY` auto-generated)
4. Live in a few minutes

---

## API Reference

### `POST /api/sign/process`
```json
// Request
{ "phrase": "HELP DOCTOR", "lang": "en" }

// Response
{
  "original": "HELP DOCTOR",
  "corrected": "Please help - I need a doctor.",
  "translated": null,
  "emergency": false,
  "kb_topic": "Seeing a Doctor",
  "kb_response": "...",
  "kb_category": "medical",
  "kb_success": true,
  "method": "knowledge_base",
  "audio_b64": "<base64 mp3>",
  "lang": "en"
}
```

### `POST /api/speech/process`
```json
// Request
{ "text": "I need a doctor", "lang": "en" }

// Response
{
  "original": "I need a doctor",
  "original_en": null,
  "simplified": "I need a doctor.",
  "display": "I need a doctor.",
  "emergency": false,
  "translated": null,
  "kb_topic": "Seeing a Doctor",
  "kb_response": "...",
  "kb_category": "medical",
  "kb_success": true,
  "method": "knowledge_base",
  "sign_sequence": ["doctor"],
  "sign_unmatched": ["i", "need", "a"],
  "lang": "en"
}
```
`sign_unmatched` lists every word with no available animation — the frontend surfaces this directly rather than silently showing nothing.

### `GET /api/health`
```json
{ "status": "ok", "project": "SignBridge AI", "knowledge_base": "active", "kb_entries": 47 }
```

---

## Known Limitations

Being upfront about these rather than papering over them:

- **The two sign vocabularies don't overlap.** The 60-word gesture-recognition set (medical/emergency-focused) and the 80-word Speech->Sign animation set (`ASL_80`, a children's-vocabulary dataset — animals, objects, everyday words) were built from different sources for different reasons. A sentence like *"I have a headache"* will correctly extract "headache" as a keyword but find no animation for it, because that word was never recorded. This is a dataset-coverage gap, not a bug — the extraction logic is already working correctly.
- **The trained DL sequence model is not demo-ready.** It loads and runs correctly, but its real-time predictions are unstable — this is very likely a distribution mismatch between clean, pre-segmented training clips and live, variable-speed camera input, which needs real instrumentation to diagnose properly. It's kept running in debug-only mode for exactly this reason.
- **Auth is intentionally minimal** (session-based, SQLite) — appropriate for this stage, not for a real multi-tenant production deployment.

---

## Roadmap (v2)

Deliberately out of scope for this version, with a production-grade architecture already designed and scaffolded separately for when it's picked up:

- LLM integration (dynamic Knowledge Engine explanations, better multilingual translation quality, natural-language rewriting of sign sequences)
- Diagnosing and fixing the DL sequence model's real-time instability, then promoting it from debug-only to user-facing
- Full WLASL / INCLUDE / AI4Bharat dataset expansion — this is what actually closes the vocabulary-overlap gap described above
- FastAPI + Clean Architecture backend, Next.js frontend, PyTorch/ST-GCN/Whisper/NLLB/Piper pipeline, Kubernetes/AWS deployment
- JWT + PostgreSQL auth, replacing the current session/SQLite setup

---

## Project Structure

```
SignBridge-AI-MVP/
|-- app.py                        Flask backend, routes, translation orchestration
|-- auth.py                       Session-based auth (Flask-Login + SQLite)
|-- knowledge_base.py             Offline Knowledge Engine (47 entries)
|-- sentence_builder.py           Sign-buffer -> grammatical English sentence
|-- sign_vocabulary.py            Speech text -> sign-sequence keyword matching
|-- requirements.txt
|-- render.yaml / Procfile        Deployment config
|-- templates/
|   |-- base.html
|   |-- home.html
|   |-- sign_to_speech.html
|   |-- speech_to_sign.html
|   `-- calibration.html
`-- static/
    |-- css/style.css
    |-- data/sign_animations.json      Recorded landmark sequences, 80 signs
    |-- model/                         Trained DL sequence model (debug-only)
    `-- js/
        |-- gesture.js                 Rule-based classifier -- drives the UI
        |-- sequence-model.js          DL model -- console-log only, no UI access
        |-- gru_sequence_model.js      Custom cuDNN-compatible GRU forward pass
        |-- sign-animation.js          SignAnimationPlayer (Play/Pause/Replay/Stop)
        |-- calibration.js
        `-- app.js                     Camera, speech, API orchestration
```

---

## Team

| Name | Role |
|---|---|
| Kalash | Full Stack Developer & AI Integration |

---

## License
MIT License — open source for maximum accessibility impact.

---

*Built for the deaf and speech-impaired community.*
