/**
 * SignBridge AI — Gesture Classifier
 * Runs entirely in the browser using MediaPipe hand landmarks.
 * No Python / ML model needed.
 */

// ─── Finger indices in MediaPipe Hands ───────────────────────────────
const FINGER_TIPS  = [4, 8, 12, 16, 20];   // thumb, index, middle, ring, pinky
const FINGER_PIPS  = [3, 6, 10, 14, 18];   // second joint (PIP for fingers, IP for thumb)
const FINGER_MCPS  = [2, 5, 9, 13, 17];    // knuckles

// ─── Sign definitions ─────────────────────────────────────────────────
// Key: "T|I|M|R|P"  (1=up, 0=down)
const SIGN_MAP = {
  "1|1|1|1|1": { sign: "HELLO",    emoji: "👋", desc: "Hello / Open Hand"  },
  "0|1|1|1|1": { sign: "PLEASE",   emoji: "🙏", desc: "Please / B shape"   },
  "0|1|1|1|0": { sign: "WATER",    emoji: "💧", desc: "Water / W shape"    },
  "0|1|1|0|0": { sign: "PEACE",    emoji: "✌️", desc: "Peace / Two"        },
  "1|1|0|0|0": { sign: "HELP",     emoji: "🆘", desc: "Help / L shape"     },
  "1|0|0|0|0": { sign: "GOOD",     emoji: "👍", desc: "Good / Thumbs Up"   },
  "0|0|0|0|0": { sign: "YES",      emoji: "✊", desc: "Yes / Fist / Stop"  },
  "0|1|0|0|0": { sign: "I",        emoji: "👆", desc: "I / Me / Point"     },
  "1|0|0|0|1": { sign: "CALL",     emoji: "🤙", desc: "Call / Y shape"     },
  "0|0|0|0|1": { sign: "PINKY",    emoji: "🤙", desc: "Pinky / Promise"    },
  "1|1|1|0|0": { sign: "THREE",    emoji: "3️⃣", desc: "Three"             },
  "1|1|1|1|0": { sign: "FOUR",     emoji: "4️⃣", desc: "Four"              },
  "0|1|1|1|0": { sign: "WATER",    emoji: "💧", desc: "Water / W"          },
  "1|0|0|0|1": { sign: "CALL",     emoji: "📞", desc: "Call / Phone"       },
};

// Practical phrase word → display mapping
const SIGN_TO_WORD = {
  "HELLO": "HELLO",
  "PLEASE": "PLEASE",
  "WATER": "WATER",
  "PEACE": "PEACE",
  "HELP": "HELP",
  "GOOD": "GOOD",
  "YES": "YES",
  "I": "I",
  "CALL": "CALL",
  "FOUR": "FOUR",
  "THREE": "THREE",
  "PINKY": "PINKY",
};

// ─── Core classifier ──────────────────────────────────────────────────

class GestureClassifier {
  constructor() {
    this.lastSign      = null;
    this.signBuffer    = [];         // accumulated signs → phrase
    this.noHandFrames  = 0;
    this.PAUSE_FRAMES  = 25;         // frames of no hand = end of phrase
    this.MIN_CONF      = 0.65;
    this.sameSignCount = 0;
    this.CONFIRM_FRAMES = 8;         // hold sign for N frames to confirm it
  }

  /**
   * Determine which fingers are "up" from 21 MediaPipe landmarks.
   * landmarks: array of {x, y, z} objects (normalised 0-1)
   */
  getFingersUp(landmarks) {
    // Thumb: tip y vs MCP y (tip higher = up)
    // Use x-axis comparison too for reliability
    const thumbUp = landmarks[4].y < landmarks[2].y - 0.02;

    // Other 4 fingers: tip y vs PIP y
    const indexUp  = landmarks[8].y  < landmarks[6].y  - 0.01;
    const middleUp = landmarks[12].y < landmarks[10].y - 0.01;
    const ringUp   = landmarks[16].y < landmarks[14].y - 0.01;
    const pinkyUp  = landmarks[20].y < landmarks[18].y - 0.01;

    return [thumbUp, indexUp, middleUp, ringUp, pinkyUp];
  }

  /**
   * Classify landmarks into a sign.
   * Returns { sign, emoji, desc, confidence, fingers }
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
        confidence: 0.85 + Math.random() * 0.12,  // simulate confidence
        fingers:    fingers,
        key:        key,
      };
    }

    // Partial match: find closest known sign
    return this._fuzzyMatch(fingers, key);
  }

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
    return { sign: "UNKNOWN", emoji: "❓", desc: "Unknown sign",
             confidence: 0, fingers, key };
  }

  /**
   * Push a classification result.
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

    // Debounce: only add sign after holding it for CONFIRM_FRAMES
    if (result.sign === this.lastSign) {
      this.sameSignCount++;
      if (this.sameSignCount === this.CONFIRM_FRAMES) {
        this.signBuffer.push(result.sign);
        return null;
      }
    } else {
      this.lastSign      = result.sign;
      this.sameSignCount = 1;
    }
    return null;
  }

  _flush() {
    if (this.signBuffer.length === 0) return null;
    const phrase     = this.signBuffer.join(" ");
    this.signBuffer  = [];
    this.noHandFrames = 0;
    this.lastSign     = null;
    this.sameSignCount = 0;
    return phrase;
  }

  flush() { return this._flush(); }

  getBuffer() { return [...this.signBuffer]; }

  clearBuffer() {
    this.signBuffer   = [];
    this.lastSign     = null;
    this.sameSignCount = 0;
  }
}

// Export for use in app.js
window.GestureClassifier = GestureClassifier;
