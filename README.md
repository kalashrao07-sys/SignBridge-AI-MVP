# 🌉 SignBridge AI

> **Breaking Communication Barriers with AI**

An AI-powered two-way communication system for Deaf and Speech-Impaired communities, built for OSC AI Build 1.0.

---

## 🎯 Problem Statement

466M+ people worldwide have disabling hearing loss. Most cannot communicate in hospitals, schools, public transport, or emergencies because others don't understand sign language. Existing solutions are expensive, one-directional, and not multilingual.

## 💡 Solution

SignBridge AI is a real-time two-way communication bridge:
- **Direction 1** — Sign Language → Wolfram-corrected sentence → Voice Output
- **Direction 2** — Speech → Wolfram-simplified text → Smart Visual Display

## 🧠 AI & Wolfram Usage

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Hand Detection | MediaPipe Hands (browser JS) | Real-time landmark detection |
| Gesture Classification | Custom JS classifier | Sign → word mapping |
| Semantic Correction | **Wolfram Alpha API** | Grammar correction ("I water need" → "I need water.") |
| Speech Simplification | **Wolfram Alpha API** | Convert speech to short visual display |
| Emergency Detection | **Wolfram Alpha API** + rules | Flag urgent phrases |
| Text-to-Speech | gTTS (Python) | Convert corrected sentence to audio |
| Translation | Google Translate API | English ↔ Hindi ↔ Kannada |

Wolfram Alpha is the **semantic intelligence engine** — every sign phrase and speech input passes through Wolfram for correction, simplification, and context extraction before being shown or spoken.

## 🛠️ Tech Stack

- **Frontend**: HTML5, CSS3, JavaScript, MediaPipe Hands.js (CDN)
- **Backend**: Python, Flask, gTTS, deep-translator
- **AI/Semantic**: Wolfram Alpha API
- **Speech Input**: Web Speech API (browser-native)
- **Deployment**: Render.com

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8+ (any version, no mediapipe needed)
- A Wolfram Alpha App ID (free at developer.wolframalpha.com)

### Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/signbridge-ai
cd signbridge-ai

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env from example
copy .env.example .env        # Windows
cp .env.example .env          # Mac/Linux

# 5. Add your Wolfram App ID to .env
# Edit .env → set WOLFRAM_APP_ID=your_key_here

# 6. Run
python app.py
```

Open **http://localhost:5000**

### Deploy to Render (free)

1. Push this repo to GitHub
2. Go to render.com → New → Web Service → connect your repo
3. Add environment variable: `WOLFRAM_APP_ID` = your key
4. Deploy — get a live URL in ~3 minutes

## 📁 Project Structure

```
signbridge-ai/
├── app.py                  ← Flask backend + Wolfram integration
├── requirements.txt
├── Procfile                ← Deployment
├── render.yaml
├── .env.example
├── templates/
│   └── index.html          ← Full two-panel UI
└── static/
    ├── css/style.css
    └── js/
        ├── gesture.js      ← MediaPipe gesture classifier
        └── app.js          ← Camera, speech, API calls
```

## 🌍 Languages Supported
- English (en)
- Hindi (hi) — हिंदी
- Kannada (kn) — ಕನ್ನಡ

## 👥 Team
- [Your Name] — [Your Role]

## 📄 License
MIT
