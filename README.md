# 🌉 SignBridge AI

> **Breaking Communication Barriers with Artificial Intelligence**

*A real-time bidirectional communication system that translates Sign Language ↔ Speech using Computer Vision, Deep Learning, Speech Recognition, and Natural Language Processing.*

---

# 📖 Table of Contents

* Overview
* Motivation
* Features
* Demo
* System Architecture
* Technology Stack
* Model Development Journey
* Dataset
* Project Pipeline
* Installation
* Project Structure
* Future Scope
* Contributors
* License

---

# 📖 Overview

SignBridge AI is an AI-powered web application designed to reduce the communication gap between people who communicate using **Sign Language** and those who communicate using **spoken language**.

The application supports communication in **both directions**:

* ✋ **Sign Language → Speech**
* 🎙 **Speech → Sign Language**

Unlike many existing systems that perform only one task, SignBridge AI combines both capabilities into a single platform while also providing multilingual support and educational information through an integrated knowledge base.

---

# 💡 Motivation

Millions of people around the world rely on sign language for everyday communication.

However,

* most people cannot understand sign language,
* interpreters are not always available,
* existing systems usually support only one direction of communication,
* and many solutions require specialized hardware.

Our goal was to build an affordable, AI-powered, browser-based solution that requires only a webcam and a microphone while making communication more accessible and inclusive.

---

# ✨ Features

### ✋ Sign → Speech

* Real-time hand gesture detection
* MediaPipe Hand Landmark Detection
* Rule-Based Gesture Recognition
* Deep Learning Sequence Recognition
* Phrase Buffering
* Multilingual Translation
* Text-to-Speech
* Knowledge Base

---

### 🎙 Speech → Sign

* Speech Recognition
* Keyword Extraction
* Sign Animation Generation
* Sequential Animation Playback
* Multilingual Speech Input
* Educational Sign Demonstration

---

# 🧠 Model Development Journey

The project has evolved through multiple iterations, gradually improving both accuracy and usability.

## Phase 1 — Rule-Based Prototype

Initially, the system was developed using handcrafted gesture rules.

MediaPipe landmarks were analyzed to determine finger positions, allowing the application to recognize supported signs with high stability and low latency.

Advantages:

* Fast
* Lightweight
* Reliable
* Excellent for real-time demonstrations

---

## Phase 2 — Deep Learning Integration

To improve scalability and enable recognition of more complex gestures, a sequence-based Deep Learning model was introduced.

Instead of analyzing a single image, the model learns the movement of a sign over time.

### Workflow

```
Video

↓

MediaPipe Hands

↓

21 Hand Landmarks

↓

Sequence Generation

↓

Bidirectional GRU

↓

Gesture Prediction
```

The trained model currently runs alongside the rule-based recognizer for evaluation and future improvements.

---

## Phase 3 — Bidirectional Communication

The project was expanded from a gesture recognizer into a complete communication platform by introducing:

* Speech Recognition
* Sign Animation Generation
* Knowledge Base
* Translation
* Text-to-Speech

making the system capable of communication in both directions.

---

# 📊 Dataset

## ASL_80 Dataset

The Deep Learning model is trained using the **ASL_80 (American Sign Language – 80 Classes)** dataset.

This dataset contains isolated sign language videos belonging to **80 different gesture classes**.

Instead of directly training on images, the project converts every video into **MediaPipe hand landmark sequences**, allowing the model to focus on hand movements rather than raw pixels.

---

# 📚 Dataset Source

The Deep Learning model used in SignBridge AI is trained using the **ASL_80 (American Sign Language - 80 Classes)** dataset.

- **Dataset:** ASL_80
- **Description:** An isolated American Sign Language dataset containing 80 sign classes for gesture recognition.
- **Source:** [ASL_80_Kaggle](https://www.kaggle.com/datasets/najaf456ali/asl-100-google-slr)

If you use this dataset in your research or projects, please acknowledge the original dataset authors.

---

## Dataset Statistics

| Property        | Value                        |
| --------------- | ---------------------------- |
| Dataset         | ASL_80                       |
| Language        | American Sign Language (ASL) |
| Classes         | 80                           |
| Input Type      | Video Sequences              |
| Processed Input | MediaPipe Landmark Sequences |
| Sequence Length | 32 Frames                    |

---

## Supported Classes

```
airplane
alligator
aunt
awake
balloon
because
bee
bird
blow
brother
brown
bye
cat
closet
cow
cry
doll
donkey
drink
dry
duck
ear
eye
farm
find
fireman
first
flower
food
frog
gift
glasswindow
goose
gum
hear
hello
home
horse
icecream
kiss
kitty
lion
lips
listen
look
loud
mad
make
man
mom
mouse
mouth
nap
napkin
nuts
old
orange
owl
pajamas
pen
pencil
penny
pizza
pretend
pretty
sad
shhh
sleepy
sun
talk
taste
think
tiger
tooth
toothbrush
uncle
up
wake
who
yesterday
```

---

# ⚙ Data Preprocessing Pipeline

Every training video undergoes several preprocessing steps before being used by the Deep Learning model.

```
Raw Video

↓

Frame Extraction

↓

MediaPipe Hands

↓

21 Hand Landmarks

↓

Coordinate Normalization

↓

Missing Landmark Interpolation

↓

Temporal Resampling

↓

32-Frame Sequence

↓

Model Training
```

This significantly reduces the input size while preserving the important hand motion information.

---

# 🤖 Deep Learning Model

The project uses a **Bidirectional GRU (BiGRU)** architecture for temporal sequence learning.

### Architecture

```
Input Sequence (32 Frames)

↓

Bidirectional GRU (48)

↓

Bidirectional GRU (32)

↓

Dense (64)

↓

Dense (80 Classes)
```

### Current Performance

| Metric         | Value |
| -------------- | ----: |
| Top-1 Accuracy |  ~72% |
| Top-3 Accuracy |  ~88% |

For the web application, the **Rule-Based Gesture Recognition** module is currently used as the primary inference engine because it provides more stable real-time predictions. The Deep Learning model remains integrated for experimentation and future improvements.

---

# 🏗 Complete Project Pipeline

## Sign → Speech

```
Webcam

↓

MediaPipe Hands

↓

Gesture Recognition

↓

Phrase Buffer

↓

Knowledge Base

↓

Translation

↓

Text-to-Speech
```

---

## Speech → Sign

```
Microphone

↓

Speech Recognition

↓

Keyword Extraction

↓

Animation Mapping

↓

Sign Animation
```

---

# 🛠 Technology Stack

## Frontend

* HTML5
* CSS3
* JavaScript
* TensorFlow.js

---

## Backend

* Python
* Flask

---

## Artificial Intelligence

* TensorFlow
* Bidirectional GRU
* MediaPipe Hands

---

## Browser APIs

* Web Speech API
* Speech Synthesis API

---

## Deployment

* Render
* GitHub

---

# 📁 Project Structure

```
SignBridgeAI/

├── backend/
│   ├── app.py
│   ├── models/
│   ├── routes/
│   ├── utils/
│   └── knowledge_base/
│
├── frontend/
│   ├── assets/
│   ├── css/
│   ├── js/
│   ├── animations/
│   └── index.html
│
├── docker-compose.yml
├── README.md
└── .gitignore
```

---

# 🚀 Future Scope

The project is continuously evolving. Planned improvements include:

* Expanding beyond the current 80-sign vocabulary.
* Training on larger datasets such as the **Google Isolated Sign Language Recognition (GISLR)** dataset to improve generalization.
* Improving continuous sign language recognition.
* Replacing rule-based recognition with a more robust sequence-based Deep Learning model for real-time inference.
* Enhancing multilingual support by translating recognized speech into English internally before mapping it to sign animations.
* Smarter keyword extraction and context-aware sentence formation using Large Language Models (LLMs).
* Mobile application support with optimized camera and microphone handling.
* Offline inference for improved accessibility in low-connectivity environments.

---

# 🤝 Contributors

**Kalash Rao**
Project Lead & AI/ML Developer

---

# 🙏 Acknowledgements

This project was built using several outstanding open-source technologies and frameworks:

* Google MediaPipe
* TensorFlow
* TensorFlow.js
* Flask
* Render
* Open Source Community

---

# 📖 References

- Google MediaPipe Hands
- TensorFlow
- TensorFlow.js
- Flask
- Render
- ASL_80 Dataset
- Web Speech API

---

# ⭐ Support

If you found this project interesting or useful, consider giving it a **⭐ Star** on GitHub. It helps others discover the project and motivates future development.

---

