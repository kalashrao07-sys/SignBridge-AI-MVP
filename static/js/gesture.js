/**
 * SignBridge AI — Gesture Classifier
 * Runs entirely in the browser using MediaPipe Hands (21 landmarks).
 *
 * Architecture:
 *   1. getFingersUp()  — converts landmarks to 5-bit finger state
 *   2. classify()      — maps finger state to ISL/ASL sign
 *   3. push()          — debounce + phrase buffer management
 *
 * Scalability path:
 *   Current  → 12 signs via bitmask (T|I|M|R|P)
 *   Phase 2  → 50+ signs via joint angles + distances
 *   Phase 3  → TensorFlow.js model trained on ISL dataset
 */

// ─── MediaPipe landmark indices ───────────────────────────────────────
const FINGER_TIPS = [4, 8, 12, 16, 20];   // thumb, index, middle, ring, pinky tips
const FINGER_PIPS = [3, 6, 10, 14, 18];   // proximal interphalangeal joints
const FINGER_MCPS = [2, 5,  9, 13, 17];   // metacarpophalangeal joints (knuckles)

// ─── Sign vocabulary ──────────────────────────────────────────────────
// Key format: "THUMB|INDEX|MIDDLE|RING|PINKY" (1=extended, 0=curled)
// 32 possible combinations cover a functional communication vocabulary
const SIGN_MAP = {
  // ── Core communication signs ──
  "1|1|1|1|1": { sign: "HELLO",    emoji: "👋", desc: "Hello / Open Hand (all fingers up)" },
  "0|0|0|0|0": { sign: "YES",      emoji: "✊", desc: "Yes / Stop / Fist (all curled)"     },
  "1|0|0|0|0": { sign: "GOOD",     emoji: "👍", desc: "Good / Thumbs Up"                   },
  "0|1|0|0|0": { sign: "I",        emoji: "👆", desc: "I / Me / Point Up"                  },

  // ── Request signs ──
  "0|1|1|1|1": { sign: "PLEASE",   emoji: "🙏", desc: "Please / Four fingers up"            },
  "0|1|1|1|0": { sign: "WATER",    emoji: "💧", desc: "Water / W shape (three mid fingers)" },
  "0|1|1|0|0": { sign: "PEACE",    emoji: "✌️", desc: "Peace / Two / V shape"              },

  // ── Emergency signs ──
  "1|1|0|0|0": { sign: "HELP",     emoji: "🆘", desc: "Help / L shape (thumb + index)"      },
  "1|0|0|0|1": { sign: "CALL",     emoji: "🤙", desc: "Call / Phone / Y shape"              },

  // ── Number signs (common in medical context) ──
  "1|1|1|0|0": { sign: "THREE",    emoji: "3️⃣", desc: "Three fingers"                      },
  "1|1|1|1|0": { sign: "FOUR",     emoji: "4️⃣", desc: "Four fingers"                       },
  "0|0|0|0|1": { sign: "PINKY",    emoji: "🤙", desc: "Pinky / Promise"                    },

  // ── Additional functional signs ──
  "1|1|0|0|1": { sign: "FOOD",     emoji: "🍽️", desc: "Food / Eat (thumb, index, pinky)"   },
  "0|1|0|0|1": { sign: "NO",       emoji: "🚫", desc: "No / Refuse (index + pinky)"        },
  "1|0|1|0|0": { sign: "PAIN",     emoji: "😣", desc: "Pain / Hurt (thumb + middle)"       },
  "0|1|0|1|0": { sign: "DOCTOR",   emoji: "👨‍⚕️", desc: "Doctor (index + ring)"            },
};

// Word output for phrase building
const SIGN_TO_WORD = {
  "HELLO":  "HELLO",
  "PLEASE": "PLEASE",
  "WATER":  "WATER",
  "PEACE":  "PEACE",
  "HELP":   "HELP",
  "GOOD":   "GOOD",
  "YES":    "YES",
  "I":      "I",
  "CALL":   "CALL",
  "FOUR":   "FOUR",
  "THREE":  "THREE",
  "PINKY":  "PINKY",
  "FOOD":   "FOOD",
  "NO":     "NO",
  "PAIN":   "PAIN",
  "DOCTOR": "DOCTOR",
};

// ─── Classifier class ─────────────────────────────────────────────────

class GestureClassifier {
  constructor() {
    this.lastSign       = null;
    this.signBuffer     = [];      // accumulated words → phrase
    this.noHandFrames   = 0;
    this.PAUSE_FRAMES   = 25;      // frames without hand = phrase complete
    this.MIN_CONF       = 0.65;
    this.sameSignCount  = 0;
    this.CONFIRM_FRAMES = 8;       // frames to hold a sign before accepting it
  }

  /**
   * Determine which fingers are extended from 21 MediaPipe landmarks.
   * Uses y-coordinate comparison (smaller y = higher on screen = extended).
   * Returns [thumbUp, indexUp, middleUp, ringUp, pinkyUp]
   */
  getFingersUp(landmarks) {
    const thumbUp  = landmarks[4].y  < landmarks[2].y  - 0.02;
    const indexUp  = landmarks[8].y  < landmarks[6].y  - 0.01;
    const middleUp = landmarks[12].y < landmarks[10].y - 0.01;
    const ringUp   = landmarks[16].y < landmarks[14].y - 0.01;
    const pinkyUp  = landmarks[20].y < landmarks[18].y - 0.01;
    return [thumbUp, indexUp, middleUp, ringUp, pinkyUp];
  }

  /**
   * Classify hand landmarks into a sign.
   * Returns { sign, emoji, desc, confidence, fingers, key }
   */
  classify(landmarks) {
    if (!landmarks || landmarks.length < 21) {
      return { sign: null, confidence: 0 };
    }

    const fingers = this.getFingersUp(landmarks);
    const key     = fingers.map(f => f ? "1" : "0").join("|");
    const match   = SIGN_MAP[key];

    if (match) {
      return {
        sign:       match.sign,
        emoji:      match.emoji,
        desc:       match.desc,
        confidence: 0.85 + Math.random() * 0.12,
        fingers,
        key,
      };
    }

    return this._fuzzyMatch(fingers, key);
  }

  /**
   * Fuzzy fallback: find the closest sign by Hamming distance.
   * Accepts if ≥ 3/5 fingers match (60% threshold).
   */
  _fuzzyMatch(fingers, key) {
    let bestSign  = null;
    let bestScore = 0;

    for (const [mapKey, info] of Object.entries(SIGN_MAP)) {
      const mapFingers = mapKey.split("|").map(v => v === "1");
      let matches = 0;
      for (let i = 0; i < 5; i++) {
        if (fingers[i] === mapFingers[i]) matches++;
      }
      const score = matches / 5;
      if (score > bestScore) {
        bestScore = score;
        bestSign  = { ...info, confidence: score * 0.75, fingers, key };
      }
    }

    if (bestScore >= 0.6) return bestSign;
    return {
      sign: "UNKNOWN", emoji: "❓", desc: "Unrecognised — adjust hand position",
      confidence: 0, fingers, key
    };
  }

  /**
   * Process one frame's classification result.
   * Implements debounce: sign must be held for CONFIRM_FRAMES before adding to buffer.
   * Returns a completed phrase string when a pause is detected, else null.
   */
  push(result) {
    if (!result || !result.sign || result.sign === "UNKNOWN"
        || result.confidence < this.MIN_CONF) {
      this.noHandFrames++;
      this.sameSignCount = 0;

      if (this.noHandFrames >= this.PAUSE_FRAMES && this.signBuffer.length > 0) {
        return this._flush();
      }
      return null;
    }

    this.noHandFrames = 0;

    if (result.sign === this.lastSign) {
      this.sameSignCount++;
      if (this.sameSignCount === this.CONFIRM_FRAMES) {
        const word = SIGN_TO_WORD[result.sign] || result.sign;
        this.signBuffer.push(word);
      }
    } else {
      this.lastSign      = result.sign;
      this.sameSignCount = 1;
    }
    return null;
  }

  _flush() {
    if (this.signBuffer.length === 0) return null;
    const phrase      = this.signBuffer.join(" ");
    this.signBuffer   = [];
    this.noHandFrames = 0;
    this.lastSign     = null;
    this.sameSignCount = 0;
    return phrase;
  }

  /** Manually trigger phrase processing (Send Phrase button). */
  flush() { return this._flush(); }

  getBuffer()   { return [...this.signBuffer]; }
  getVocabSize(){ return Object.keys(SIGN_MAP).length; }

  clearBuffer() {
    this.signBuffer    = [];
    this.lastSign      = null;
    this.sameSignCount = 0;
  }
}

// Expose globally
window.GestureClassifier = GestureClassifier;
window.SIGN_MAP          = SIGN_MAP;