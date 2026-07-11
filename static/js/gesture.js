/**
 * SignBridge AI — Gesture Classifier v2
 * Runs entirely in the browser using MediaPipe Hands (21 landmarks).
 *
 * WHY THIS CHANGED FROM v1:
 *   v1 mapped hand shape to a 5-bit up/down bitmask against 16 fixed
 *   combinations. That's brittle — any deviation in camera angle, hand
 *   rotation, or lighting produces a miss, and the vocabulary is frozen
 *   at compile time.
 *
 *   v2 extracts continuous features (per-finger curl angle + tip
 *   extension distance, normalized for hand size/position) and
 *   classifies with k-nearest-neighbors against a small set of samples.
 *   Those samples can be the built-in defaults (same 16 signs, kept as
 *   a fallback) OR live samples recorded from the presenter's own hand
 *   right before a demo — which is what actually fixes "only certain
 *   things get recognized": the system adapts to *your* hand instead
 *   of demanding your hand match a fixed template.
 *
 * Scalability path (unchanged intent, updated mechanism):
 *   Current  → k-NN over hand-crafted geometric features, browser-only
 *   Phase 2  → same features, trained offline on a larger labeled set,
 *              shipped as a compact model (e.g. small MLP in TF.js)
 *   Phase 3  → full sequence model (ST-GCN) for dynamic/continuous ISL
 */

// ─── MediaPipe landmark indices ───────────────────────────────────────
const WRIST = 0;
const FINGER_MCPS = [2, 5, 9, 13, 17];   // knuckles (base of each finger)
const FINGER_PIPS = [3, 6, 10, 14, 18];  // middle joints
const FINGER_TIPS = [4, 8, 12, 16, 20];  // fingertips
const MIDDLE_MCP  = 9;                   // used for scale normalization

const FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"];

// ─── Default (bootstrap) sign vocabulary — same 16 signs as v1 ────────
// Kept as bitmask so the app works out of the box with zero training.
// Once the user records custom samples for a sign, those take priority.
const DEFAULT_SIGN_MAP = {
  "1|1|1|1|1": { sign: "HELLO",  emoji: "👋", desc: "Hello / Open Hand" },
  "0|0|0|0|0": { sign: "YES",    emoji: "✊", desc: "Yes / Stop / Fist" },
  "1|0|0|0|0": { sign: "GOOD",   emoji: "👍", desc: "Good / Thumbs Up" },
  "0|1|0|0|0": { sign: "I",      emoji: "👆", desc: "I / Me / Point Up" },
  "0|1|1|1|1": { sign: "PLEASE", emoji: "🙏", desc: "Please / Four fingers up" },
  "0|1|1|1|0": { sign: "WATER",  emoji: "💧", desc: "Water / W shape" },
  "0|1|1|0|0": { sign: "PEACE",  emoji: "✌️", desc: "Peace / V shape" },
  "1|1|0|0|0": { sign: "HELP",   emoji: "🆘", desc: "Help / L shape" },
  "1|0|0|0|1": { sign: "CALL",   emoji: "🤙", desc: "Call / Y shape" },
  "1|1|1|0|0": { sign: "THREE",  emoji: "3️⃣", desc: "Three fingers" },
  "1|1|1|1|0": { sign: "FOUR",   emoji: "4️⃣", desc: "Four fingers" },
  "0|0|0|0|1": { sign: "PINKY",  emoji: "🤙", desc: "Pinky / Promise" },
  "1|1|0|0|1": { sign: "FOOD",   emoji: "🍽️", desc: "Food / Eat" },
  "0|1|0|0|1": { sign: "NO",     emoji: "🚫", desc: "No / Refuse" },
  "1|0|1|0|0": { sign: "PAIN",   emoji: "😣", desc: "Pain / Hurt" },
  "0|1|0|1|0": { sign: "DOCTOR", emoji: "👨‍⚕️", desc: "Doctor" },
};

const SIGN_TO_WORD = Object.fromEntries(
  Object.values(DEFAULT_SIGN_MAP).map(v => [v.sign, v.sign])
);

const TRAINING_STORAGE_KEY = "signbridge_custom_training_v1";

// ─── Feature extraction ────────────────────────────────────────────────

/**
 * Convert 21 raw landmarks into an 11-dim feature vector that is
 * roughly invariant to hand position, rotation-in-plane, and scale:
 *   [thumbCurl, indexCurl, middleCurl, ringCurl, pinkyCurl,
 *    thumbExt,  indexExt,  middleExt,  ringExt,  pinkyExt,
 *    thumbIndexPinchDist]
 * Curl angles are normalized to 0–1 (0 = fully curled, 1 = fully straight).
 * Extension distances are normalized by hand scale (wrist→middle-knuckle).
 */
function extractFeatures(landmarks) {
  const scale = dist(landmarks[WRIST], landmarks[MIDDLE_MCP]) || 1e-6;

  const curls = FINGER_NAMES.map((_, i) => {
    const mcp = landmarks[FINGER_MCPS[i]];
    const pip = landmarks[FINGER_PIPS[i]];
    const tip = landmarks[FINGER_TIPS[i]];
    const v1  = sub(pip, mcp);
    const v2  = sub(tip, pip);
    const angle = angleBetween(v1, v2); // 0 = straight, PI = fully folded back
    return 1 - clamp(angle / Math.PI, 0, 1); // 1 = straight/extended
  });

  const exts = FINGER_TIPS.map(tipIdx =>
    clamp(dist(landmarks[tipIdx], landmarks[WRIST]) / (scale * 3), 0, 1)
  );

  const pinch = clamp(
    dist(landmarks[FINGER_TIPS[0]], landmarks[FINGER_TIPS[1]]) / (scale * 2),
    0, 1
  );

  return [...curls, ...exts, pinch];
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}
function sub(a, b) {
  return { x: a.x - b.x, y: a.y - b.y, z: (a.z || 0) - (b.z || 0) };
}
function angleBetween(a, b) {
  const dot = a.x * b.x + a.y * b.y + a.z * b.z;
  const magA = Math.hypot(a.x, a.y, a.z) || 1e-6;
  const magB = Math.hypot(b.x, b.y, b.z) || 1e-6;
  return Math.acos(clamp(dot / (magA * magB), -1, 1));
}
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function euclidean(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] - b[i]) ** 2;
  return Math.sqrt(sum);
}

function fingersUpFromFeatures(features) {
  // features[0..4] are curl scores (1 = extended); threshold at 0.55
  return features.slice(0, 5).map(v => v > 0.55);
}

// ─── Classifier class ─────────────────────────────────────────────────

class GestureClassifier {
  constructor() {
    this.lastSign        = null;
    this.signBuffer       = [];
    this.noHandFrames     = 0;
    this.PAUSE_FRAMES     = 25;
    this.MIN_CONF         = 0.6;
    this.sameSignCount    = 0;
    this.CONFIRM_FRAMES   = 8;
    this.K                = 5;

    /** @type {{label:string, features:number[]}[]} */
    this.customSamples = [];
    this._loadCustomSamples();
  }

  // ── Calibration API ──────────────────────────────────────────────
  addTrainingSample(label, landmarks) {
    if (!landmarks || landmarks.length < 21) return false;
    const features = extractFeatures(landmarks);
    this.customSamples.push({ label: label.toUpperCase(), features });
    this._saveCustomSamples();
    return true;
  }

  getSampleCounts() {
    const counts = {};
    for (const s of this.customSamples) {
      counts[s.label] = (counts[s.label] || 0) + 1;
    }
    return counts;
  }

  clearTraining(label = null) {
    if (label) {
      this.customSamples = this.customSamples.filter(s => s.label !== label.toUpperCase());
    } else {
      this.customSamples = [];
    }
    this._saveCustomSamples();
  }

  _saveCustomSamples() {
    try {
      localStorage.setItem(TRAINING_STORAGE_KEY, JSON.stringify(this.customSamples));
    } catch (e) { /* storage unavailable — non-fatal */ }
  }

  _loadCustomSamples() {
    try {
      const raw = localStorage.getItem(TRAINING_STORAGE_KEY);
      if (raw) this.customSamples = JSON.parse(raw);
    } catch (e) { this.customSamples = []; }
  }

  // ── Classification ───────────────────────────────────────────────

  classify(landmarks) {
    if (!landmarks || landmarks.length < 21) {
      return { sign: null, confidence: 0 };
    }

    const features = extractFeatures(landmarks);

    // Prefer live-trained samples if enough exist (>=3 total, any label)
    if (this.customSamples.length >= 3) {
      const knnResult = this._knnClassify(features);
      if (knnResult) return { ...knnResult, fingers: fingersUpFromFeatures(features) };
    }

    // Fallback: original bitmask exact/fuzzy match — keeps the app
    // fully functional with zero calibration required.
    return this._bitmaskClassify(features);
  }

  _knnClassify(features) {
    const scored = this.customSamples
      .map(s => ({ label: s.label, d: euclidean(features, s.features) }))
      .sort((a, b) => a.d - b.d)
      .slice(0, Math.min(this.K, this.customSamples.length));

    const votes = {};
    for (const s of scored) votes[s.label] = (votes[s.label] || 0) + 1;

    const winner = Object.entries(votes).sort((a, b) => b[1] - a[1])[0];
    const [label, voteCount] = winner;

    const avgDist = scored
      .filter(s => s.label === label)
      .reduce((sum, s) => sum + s.d, 0) / voteCount;

    // features are 0-1 range across 11 dims -> max plausible distance ~ sqrt(11) ≈ 3.3
    const distConfidence = clamp(1 - avgDist / 1.8, 0, 1);
    const voteConfidence = voteCount / scored.length;
    const confidence = distConfidence * 0.7 + voteConfidence * 0.3;

    const meta = Object.values(DEFAULT_SIGN_MAP).find(v => v.sign === label);
    return {
      sign: label,
      emoji: meta?.emoji || "🖐️",
      desc: meta?.desc || `Custom sign: ${label}`,
      confidence,
    };
  }

  _bitmaskClassify(features) {
    const fingers = fingersUpFromFeatures(features);
    const key = fingers.map(f => f ? "1" : "0").join("|");
    const match = DEFAULT_SIGN_MAP[key];

    if (match) {
      return { sign: match.sign, emoji: match.emoji, desc: match.desc, confidence: 0.85, fingers, key };
    }
    return this._fuzzyMatch(fingers);
  }

  _fuzzyMatch(fingers) {
    let bestSign = null, bestScore = 0;
    for (const [mapKey, info] of Object.entries(DEFAULT_SIGN_MAP)) {
      const mapFingers = mapKey.split("|").map(v => v === "1");
      let matches = 0;
      for (let i = 0; i < 5; i++) if (fingers[i] === mapFingers[i]) matches++;
      const score = matches / 5;
      if (score > bestScore) { bestScore = score; bestSign = { ...info, confidence: score * 0.75, fingers }; }
    }
    if (bestScore >= 0.6) return bestSign;
    return { sign: "UNKNOWN", emoji: "❓", desc: "Unrecognised — adjust hand position", confidence: 0, fingers };
  }

  // ── Phrase buffer (unchanged behavior) ───────────────────────────

  push(result) {
    if (!result || !result.sign || result.sign === "UNKNOWN" || result.confidence < this.MIN_CONF) {
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
      this.lastSign = result.sign;
      this.sameSignCount = 1;
    }
    return null;
  }

  _flush() {
    if (this.signBuffer.length === 0) return null;
    const phrase = this.signBuffer.join(" ");
    this.signBuffer = [];
    this.noHandFrames = 0;
    this.lastSign = null;
    this.sameSignCount = 0;
    return phrase;
  }

  flush() { return this._flush(); }
  getBuffer() { return [...this.signBuffer]; }
  getVocabSize() {
    return new Set([
      ...Object.values(DEFAULT_SIGN_MAP).map(v => v.sign),
      ...this.customSamples.map(s => s.label),
    ]).size;
  }
  clearBuffer() {
    this.signBuffer = [];
    this.lastSign = null;
    this.sameSignCount = 0;
  }
}

window.GestureClassifier = GestureClassifier;
window.DEFAULT_SIGN_MAP  = DEFAULT_SIGN_MAP;
window.extractFeatures   = extractFeatures;