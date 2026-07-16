/**
 * SignBridge AI — Sequence Model (browser-side)
 * Runs the trained GRU model on a rolling window of hand landmarks,
 * using the SAME feature extraction as preprocess_signs_v3_hands_only.py:
 *   - hands only (no pose) — 21 left + 21 right = 42 points, x&y = 84 features
 *   - sequence-level (not per-frame) center/scale normalization
 *   - resampled to 32 frames
 *
 * Runs alongside gesture.js's per-frame classifier, not instead of it —
 * this catches the 64-word trained vocabulary, gesture.js's
 * bitmask/calibration system still catches the emergency word set.
 */

const SEQ_WINDOW = 32;
const SEQ_MIN_CONF = 0.55;      // below this, don't surface a guess
const SEQ_COOLDOWN_MS = 1800;   // don't re-add the same word faster than this

let seqModel = null;
let seqLabelNames = [];
let seqReliableSet = new Set();
let seqFrameBuffer = [];        // rolling raw landmark frames
let seqLastWord = null;
let seqLastWordTime = 0;

async function loadSequenceModel() {
  seqModel = await tf.loadLayersModel("/static/model/model.json");
  seqLabelNames = await (await fetch("/static/model/label_names.json")).json();
  const reliable = await (await fetch("/static/model/reliable_signs_v3.json")).json();
  seqReliableSet = new Set(reliable);
  console.log(`Sequence model loaded: ${seqLabelNames.length} classes, ${seqReliableSet.size} reliable`);
}

/**
 * Call once per frame with the raw MediaPipe landmarks (or null if no
 * hand detected this frame). Internally buffers a rolling window and
 * runs inference periodically. Returns nothing directly — pushes
 * recognized words straight into the sign buffer via addSequenceWord().
 */
function pushSequenceFrame(multiHandLandmarks) {
  if (!seqModel) return;

  const points = _extractHandPoints(multiHandLandmarks);  // (42,2) or null
  seqFrameBuffer.push(points);
  if (seqFrameBuffer.length > SEQ_WINDOW) seqFrameBuffer.shift();

  if (seqFrameBuffer.length < SEQ_WINDOW) return;

  // Only run inference if at least half the window has a detected hand —
  // avoids classifying mostly-empty windows (hand entering/leaving frame).
  const validFrames = seqFrameBuffer.filter(f => f !== null).length;
  if (validFrames < SEQ_WINDOW * 0.5) return;

  _runInference(seqFrameBuffer);
}

function _extractHandPoints(multiHandLandmarks) {
  if (!multiHandLandmarks || multiHandLandmarks.length === 0) return null;

  // MediaPipe Hands doesn't label left/right as reliably as Holistic did
  // in the dataset, so we place first detected hand in slot 0 (left) and
  // second in slot 1 (right) — consistent within a session, which is
  // what matters since normalization is relative, not absolute L/R.
  const pts = new Array(42).fill(null).map(() => [NaN, NaN]);

  multiHandLandmarks.slice(0, 2).forEach((hand, handIdx) => {
    hand.forEach((lm, i) => {
      pts[handIdx * 21 + i] = [lm.x, lm.y];
    });
  });
  return pts;
}

function _runInference(frames) {
  // Build (32, 42, 2) array, treating missing hands as NaN for now
  const raw = frames.map(f => f || new Array(42).fill([NaN, NaN]));

  // Sequence-level center/scale from wrist (idx 0) + middle MCP (idx 9),
  // per hand, matching preprocess_signs_v3_hands_only.py exactly.
  const leftWrists = [], rightWrists = [], leftScales = [], rightScales = [];
  for (const frame of raw) {
    const lw = frame[0], lm = frame[9];
    const rw = frame[21], rm = frame[30];
    if (!isNaN(lw[0])) { leftWrists.push(lw); leftScales.push(_dist(lw, lm)); }
    if (!isNaN(rw[0])) { rightWrists.push(rw); rightScales.push(_dist(rw, rm)); }
  }

  const allWrists = [...leftWrists, ...rightWrists];
  const allScales = [...leftScales, ...rightScales];
  if (allWrists.length === 0) return; // no usable hand data this window

  const center = [
    allWrists.reduce((s, p) => s + p[0], 0) / allWrists.length,
    allWrists.reduce((s, p) => s + p[1], 0) / allWrists.length,
  ];
  let scale = allScales.reduce((s, v) => s + v, 0) / allScales.length;
  if (!scale || scale < 1e-4) scale = 1.0;

  // Interpolate NaNs across time per point, then normalize
  const filled = _interpolateFrames(raw);
  const normalized = filled.map(frame =>
    frame.map(([x, y]) => [(x - center[0]) / scale, (y - center[1]) / scale])
  );

  const flat = normalized.map(frame => frame.flat()); // (32, 84)
  const inputTensor = tf.tensor([flat]); // (1, 32, 84)

  const output = seqModel.predict(inputTensor);
  const probs = output.dataSync();
  inputTensor.dispose();
  output.dispose();

  const top = _argmax(probs);
  const label = seqLabelNames[top];
  const confidence = probs[top];

  if (confidence < SEQ_MIN_CONF) return;
  if (!seqReliableSet.has(label)) return; // only surface words we validated as reliable

  addSequenceWord(label);
}

function _dist(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function _interpolateFrames(raw) {
  const T = raw.length, N = 42;
  const out = raw.map(f => f.map(p => [...p]));
  for (let pt = 0; pt < N; pt++) {
    for (let axis = 0; axis < 2; axis++) {
      let lastVal = null, lastIdx = -1;
      for (let t = 0; t < T; t++) {
        const v = out[t][pt][axis];
        if (!isNaN(v)) {
          if (lastVal !== null && t - lastIdx > 1) {
            for (let k = lastIdx + 1; k < t; k++) {
              const frac = (k - lastIdx) / (t - lastIdx);
              out[k][pt][axis] = lastVal + (v - lastVal) * frac;
            }
          }
          lastVal = v; lastIdx = t;
        }
      }
      for (let t = 0; t < T; t++) {
        if (isNaN(out[t][pt][axis])) out[t][pt][axis] = lastVal !== null ? lastVal : 0.0;
      }
    }
  }
  return out;
}

function _argmax(arr) {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

/** Pushes a recognized word into the same sign buffer the calibrated
 * classifier uses, with a cooldown so holding a sign doesn't spam it. */
function addSequenceWord(word) {
  const now = Date.now();
  if (word === seqLastWord && now - seqLastWordTime < SEQ_COOLDOWN_MS) return;
  seqLastWord = word;
  seqLastWordTime = now;

  if (window.classifier && Array.isArray(window.classifier.signBuffer)) {
    window.classifier.signBuffer.push(word.toUpperCase());
  }

  const chips = document.getElementById("bufferChips");
  if (chips) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = word.toUpperCase();
    chips.appendChild(chip);
  }
}

window.loadSequenceModel = loadSequenceModel;
window.pushSequenceFrame = pushSequenceFrame;
