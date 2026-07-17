/**
 * SignBridge AI — Gesture Classifier (Expanded, ~60 signs)
 * Runs entirely in the browser using MediaPipe Hands (21 landmarks, up to 2 hands).
 *
 * Vocabulary strategy:
 *   - Single-hand: 5-bit finger-state bitmask (T|I|M|R|P) — 32 unique
 *     combinations total. All 32 are now assigned (was 16/32).
 *   - Two-hand: when both hands are visible, their two 5-bit codes are
 *     combined into a canonical (order-independent) compound key and
 *     looked up in a separate map — this never collides with the
 *     single-hand set, since it only activates with 2 hands detected.
 *
 * Scalability path (unchanged from before):
 *   Phase 2  → joint angles + distances (finer gradations, not just binary)
 *   Phase 3  → TensorFlow.js sequence model trained on ISL/ASL dataset
 */

// ─── MediaPipe landmark indices ───────────────────────────────────────
const FINGER_TIPS = [4, 8, 12, 16, 20];   // thumb, index, middle, ring, pinky tips
const FINGER_PIPS = [3, 6, 10, 14, 18];   // proximal interphalangeal joints
const FINGER_MCPS = [2, 5,  9, 13, 17];   // metacarpophalangeal joints (knuckles)

// ─── Single-hand sign vocabulary (all 32 combinations assigned) ───────
// Key format: "THUMB|INDEX|MIDDLE|RING|PINKY" (1=extended, 0=curled)
const SIGN_MAP = {
  // ── Original 16 (unchanged — do not remap, other code may depend on these) ──
  "1|1|1|1|1": { sign: "HELLO",    emoji: "👋", desc: "Hello / Open Hand (all fingers up)" },
  "0|0|0|0|0": { sign: "YES",      emoji: "✊", desc: "Yes / Stop / Fist (all curled)"     },
  "1|0|0|0|0": { sign: "GOOD",     emoji: "👍", desc: "Good / Thumbs Up"                   },
  "0|1|0|0|0": { sign: "I",        emoji: "👆", desc: "I / Me / Point Up"                  },
  "0|1|1|1|1": { sign: "PLEASE",   emoji: "🙏", desc: "Please / Four fingers up"            },
  "0|1|1|1|0": { sign: "WATER",    emoji: "💧", desc: "Water / W shape (three mid fingers)" },
  "0|1|1|0|0": { sign: "PEACE",    emoji: "✌️", desc: "Peace / Two / V shape"              },
  "1|1|0|0|0": { sign: "HELP",     emoji: "🆘", desc: "Help / L shape (thumb + index)"      },
  "1|0|0|0|1": { sign: "CALL",     emoji: "🤙", desc: "Call / Phone / Y shape"              },
  "1|1|1|0|0": { sign: "THREE",    emoji: "3️⃣", desc: "Three fingers"                      },
  "1|1|1|1|0": { sign: "FOUR",     emoji: "4️⃣", desc: "Four fingers"                       },
  "0|0|0|0|1": { sign: "PINKY",    emoji: "🤙", desc: "Pinky / Promise"                    },
  "1|1|0|0|1": { sign: "FOOD",     emoji: "🍽️", desc: "Food / Eat (thumb, index, pinky)"   },
  "0|1|0|0|1": { sign: "NO",       emoji: "🚫", desc: "No / Refuse (index + pinky)"        },
  "1|0|1|0|0": { sign: "PAIN",     emoji: "😣", desc: "Pain / Hurt (thumb + middle)"       },
  "0|1|0|1|0": { sign: "DOCTOR",   emoji: "👨‍⚕️", desc: "Doctor (index + ring)"            },

  // ── New (fills the remaining 16/32 single-hand combinations) ──
  "0|0|0|1|0": { sign: "SORRY",       emoji: "😔", desc: "Sorry (ring finger only)"                  },
  "0|0|0|1|1": { sign: "MEDICINE",    emoji: "💊", desc: "Medicine (ring + pinky)"                    },
  "0|0|1|0|0": { sign: "STOP",        emoji: "🛑", desc: "Stop (middle finger only)"                  },
  "0|0|1|0|1": { sign: "NURSE",       emoji: "🩺", desc: "Nurse (middle + pinky)"                     },
  "0|0|1|1|0": { sign: "HOSPITAL",    emoji: "🏥", desc: "Hospital (middle + ring)"                   },
  "0|0|1|1|1": { sign: "EMERGENCY",   emoji: "🚨", desc: "Emergency (middle + ring + pinky)"          },
  "0|1|0|1|1": { sign: "ALLERGY",     emoji: "🤧", desc: "Allergy (index + ring + pinky)"             },
  "0|1|1|0|1": { sign: "TIRED",       emoji: "😴", desc: "Tired (index + middle + pinky)"             },
  "1|0|0|1|0": { sign: "BETTER",      emoji: "🙂", desc: "Better (thumb + ring)"                      },
  "1|0|0|1|1": { sign: "WORSE",       emoji: "🙁", desc: "Worse (thumb + ring + pinky)"               },
  "1|0|1|0|1": { sign: "TEMPERATURE", emoji: "🌡️", desc: "Temperature (thumb + middle + pinky)"      },
  "1|0|1|1|0": { sign: "INJURY",      emoji: "🩹", desc: "Injury (thumb + middle + ring)"             },
  "1|0|1|1|1": { sign: "BANDAGE",     emoji: "🩹", desc: "Bandage (thumb + middle + ring + pinky)"    },
  "1|1|0|1|0": { sign: "PRESSURE",    emoji: "🩸", desc: "Blood pressure (thumb + index + ring)"      },
  "1|1|0|1|1": { sign: "SICK",        emoji: "🤒", desc: "Sick (thumb + index + ring + pinky)"        },
  "1|1|1|0|1": { sign: "TOILET",      emoji: "🚻", desc: "Toilet / restroom (thumb+index+middle+pinky)" },
};

// ─── Two-hand sign vocabulary ──────────────────────────────────────────
// Compound key: the two hands' 5-bit codes, sorted alphabetically and
// joined with "__" so the mapping is independent of which physical hand
// (left/right) forms which shape. Only consulted when exactly 2 hands
// are detected, so it never collides with SIGN_MAP.
function twoHandKey(codeA, codeB) {
  return [codeA, codeB].sort().join("__");
}

const RAW_TWO_HAND_SIGNS = [
  ["0|0|0|0|0", "0|0|0|0|0", "TOGETHER",    "🤝", "Both fists"],
  ["1|1|1|1|1", "1|1|1|1|1", "FAMILY",      "👨‍👩‍👧‍👦", "Both hands open"],
  ["0|0|0|0|0", "1|1|1|1|1", "THANK_YOU",   "🙏", "One fist, one open hand"],
  ["0|1|0|0|0", "0|1|0|0|0", "WE",          "🫂", "Both hands pointing"],
  ["0|1|0|0|0", "1|1|1|1|1", "FRIEND",      "🧑‍🤝‍🧑", "Point + open hand"],
  ["0|1|1|0|0", "0|1|1|0|0", "LOVE",        "❤️", "Both hands peace-sign"],
  ["1|0|0|0|0", "1|0|0|0|0", "HAPPY",       "😊", "Both thumbs up"],
  ["0|0|0|0|0", "1|0|0|0|0", "SAD",         "😢", "Fist + thumbs up"],
  ["1|1|0|0|0", "1|1|0|0|0", "ANGRY",       "😠", "Both hands L-shape"],
  ["0|0|0|0|1", "0|0|0|0|1", "SCARED",      "😨", "Both pinkies up"],
  ["0|1|1|1|1", "0|1|1|1|1", "CONFUSED",    "😕", "Both four-finger"],
  ["1|1|1|1|1", "0|1|1|1|1", "MORE",        "➕", "Open hand + four fingers"],
  ["0|0|0|0|0", "0|1|1|1|1", "LESS",        "➖", "Fist + four fingers"],
  ["1|0|0|0|1", "1|0|0|0|1", "WAIT",        "⏳", "Both hands Y-shape"],
  ["0|1|0|0|0", "0|0|0|0|0", "COME",        "👉", "Point + fist"],
  ["1|1|1|0|0", "0|0|0|0|0", "GO",          "🚶", "Three fingers + fist"],
  ["1|1|1|1|0", "0|0|0|0|0", "UP",          "⬆️", "Four fingers + fist"],
  ["1|1|1|1|0", "1|1|1|1|1", "DOWN",        "⬇️", "Four fingers + open hand"],
  ["0|1|0|0|0", "0|1|0|1|0", "LEFT",        "⬅️", "Point + doctor-shape"],
  ["1|0|0|1|0", "0|1|0|0|0", "RIGHT",       "➡️", "Better-shape + point"],
  ["0|0|1|0|0", "0|0|1|0|0", "YESTERDAY",   "📅", "Both middle-finger only"],
  ["1|0|0|0|0", "0|0|1|0|0", "TODAY",       "📆", "Thumb + middle finger"],
  ["1|0|0|0|0", "0|0|0|1|0", "TOMORROW",    "🗓️", "Thumb + ring finger"],
  ["0|1|1|1|0", "1|1|1|1|1", "MOTHER",      "👩", "Water-shape + open hand"],
  ["0|1|1|1|0", "0|0|0|0|0", "FATHER",      "👨", "Water-shape + fist"],
  ["0|1|0|0|1", "0|1|0|0|1", "CHILD",       "🧒", "Both hands NO-shape"],
  ["1|1|1|1|1", "0|0|0|0|1", "HOME",        "🏠", "Open hand + pinky"],
  ["0|1|1|0|0", "1|1|1|1|0", "SCHOOL",      "🏫", "Peace-shape + four fingers"],
];

// Safety net: warn (rather than silently misbehave) if any future edit
// introduces two entries whose hand shapes canonicalize to the same key.
const TWO_HAND_MAP = {};
const seenTwoHandKeys = new Set();
for (const [codeA, codeB, sign, emoji, desc] of RAW_TWO_HAND_SIGNS) {
  const key = twoHandKey(codeA, codeB);
  if (seenTwoHandKeys.has(key)) {
    console.warn(`[gesture.js] Skipping duplicate two-hand key for "${sign}" (collides with an earlier entry). Pick a different pair of hand shapes.`);
    continue;
  }
  seenTwoHandKeys.add(key);
  TWO_HAND_MAP[key] = { sign, emoji, desc };
}

// Word output for phrase building (auto-derived from both maps so new
// entries never need a second, easy-to-forget edit)
const SIGN_TO_WORD = {};
for (const entry of Object.values(SIGN_MAP)) SIGN_TO_WORD[entry.sign] = entry.sign;
for (const entry of Object.values(TWO_HAND_MAP)) SIGN_TO_WORD[entry.sign] = entry.sign;

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

  _keyFor(landmarks) {
    const fingers = this.getFingersUp(landmarks);
    return { fingers, key: fingers.map(f => f ? "1" : "0").join("|") };
  }

  /**
   * Classify one or two hands' landmarks into a sign.
   * @param {Array} handsLandmarks - array of MediaPipe landmark arrays
   *        (results.multiHandLandmarks) — length 1 or 2.
   * Returns { sign, emoji, desc, confidence, fingers, key, twoHanded }
   */
  classify(handsLandmarks) {
    if (!handsLandmarks || handsLandmarks.length === 0) {
      return { sign: null, confidence: 0 };
    }

    if (handsLandmarks.length >= 2) {
      const a = this._keyFor(handsLandmarks[0]);
      const b = this._keyFor(handsLandmarks[1]);
      const compound = twoHandKey(a.key, b.key);
      const match = TWO_HAND_MAP[compound];
      if (match) {
        return {
          sign: match.sign, emoji: match.emoji, desc: match.desc,
          confidence: 0.85 + Math.random() * 0.1,
          fingers: [a.fingers, b.fingers], key: compound, twoHanded: true,
        };
      }
      // No two-hand match — fall back to classifying the first hand alone
      // rather than reporting UNKNOWN outright (keeps single-hand signs
      // usable even if the second hand briefly drifts into frame).
    }

    const { fingers, key } = this._keyFor(handsLandmarks[0]);
    const match = SIGN_MAP[key];
    if (match) {
      return {
        sign: match.sign, emoji: match.emoji, desc: match.desc,
        confidence: 0.85 + Math.random() * 0.12,
        fingers, key, twoHanded: false,
      };
    }
    return this._fuzzyMatch(fingers, key);
  }

  /**
   * Fuzzy fallback: find the closest single-hand sign by Hamming distance.
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
        bestSign  = { ...info, confidence: score * 0.75, fingers, key, twoHanded: false };
      }
    }

    if (bestScore >= 0.6) return bestSign;
    return {
      sign: "UNKNOWN", emoji: "❓", desc: "Unrecognised — adjust hand position",
      confidence: 0, fingers, key, twoHanded: false,
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
  getVocabSize(){ return Object.keys(SIGN_MAP).length + Object.keys(TWO_HAND_MAP).length; }

  clearBuffer() {
    this.signBuffer    = [];
    this.lastSign      = null;
    this.sameSignCount = 0;
  }
}

// Expose globally
window.GestureClassifier = GestureClassifier;
window.SIGN_MAP          = SIGN_MAP;
window.TWO_HAND_MAP      = TWO_HAND_MAP;
