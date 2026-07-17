/**
 * SignBridge AI — Sign Animation Renderer
 * Replaces the emoji-based voice->sign strip with an animated hand/arm
 * skeleton, driven by real recorded landmark sequences (see
 * extract_sign_keyframes.py) — not fabricated poses, not emoji.
 *
 * Stateful player (SignAnimationPlayer): play / pause / resume / replay /
 * stop, so a played sequence can be revisited without re-running speech
 * recognition. Load sign_animations.json once at startup, then call
 * SignAnimationPlayer.play(containerId, ["help","water"]) whenever
 * /api/speech/process returns a sign_sequence.
 */

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],           // thumb
  [0,5],[5,6],[6,7],[7,8],           // index
  [5,9],[9,10],[10,11],[11,12],      // middle
  [9,13],[13,14],[14,15],[15,16],    // ring
  [13,17],[17,18],[18,19],[19,20],   // pinky
  [0,17],                             // palm base
];

// Point layout produced by extract_sign_keyframes.py:
// 0-20 = left hand, 21-41 = right hand,
// 42=L-shoulder 43=R-shoulder 44=L-elbow 45=R-elbow 46=L-wrist 47=R-wrist
const ARM_CONNECTIONS = [
  [42, 43],       // shoulder line
  [42, 44], [44, 46], [46, 0],    // left shoulder -> elbow -> wrist -> hand
  [43, 45], [45, 47], [47, 21],   // right shoulder -> elbow -> wrist -> hand
];

let SIGN_ANIMATIONS = null;

async function loadSignAnimations(url = "/static/data/sign_animations.json") {
  const res = await fetch(url);
  SIGN_ANIMATIONS = await res.json();
  return SIGN_ANIMATIONS;
}

function _drawFrame(ctx, points, w, h) {
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "#00b4d8";
  ctx.fillStyle = "#eaf2fb";
  ctx.lineWidth = 2;

  const toPx = ([x, y]) => [x * w, y * h];

  for (const [a, b] of [...ARM_CONNECTIONS, ...offset(HAND_CONNECTIONS, 0), ...offset(HAND_CONNECTIONS, 21)]) {
    const [ax, ay] = toPx(points[a]);
    const [bx, by] = toPx(points[b]);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  }

  for (const p of points) {
    const [px, py] = toPx(p);
    ctx.beginPath();
    ctx.arc(px, py, 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

function offset(connections, by) {
  return connections.map(([a, b]) => [a + by, b + by]);
}

/**
 * Stateful sign-sequence player. States: "idle" (nothing loaded),
 * "playing", "paused", "finished" (reached the end), "empty" (none of
 * the given words had an animation).
 *
 * The canvas/label are created once per container and reused across
 * play() calls, so state persists correctly across Play/Pause/Replay.
 */
const SignAnimationPlayer = (() => {
  let container = null, canvas = null, ctx = null, label = null, statusEl = null;
  let words = [];       // filtered to only words that have an animation
  let skipped = [];     // words from the last play() call with no animation
  let wordIdx = 0;
  let frameIdx = 0;
  let timer = null;
  let state = "idle";
  const FPS = 14;
  const INTER_WORD_PAUSE_MS = 200;

  function _ensureUI(containerId) {
    const el = document.getElementById(containerId);
    if (!el) return false;
    if (container !== el) {
      container = el;
      container.innerHTML = "";
      canvas = document.createElement("canvas");
      canvas.width = 320;
      canvas.height = 320;
      canvas.className = "sign-anim-canvas";
      ctx = canvas.getContext("2d");
      label = document.createElement("div");
      label.className = "sign-anim-label";
      statusEl = document.createElement("div");
      statusEl.className = "sign-anim-status";
      container.appendChild(canvas);
      container.appendChild(label);
      container.appendChild(statusEl);
    }
    return true;
  }

  function _setStatus(text) {
    if (statusEl) statusEl.textContent = text;
  }

  function _tick() {
    if (state !== "playing") return;
    const frames = SIGN_ANIMATIONS[words[wordIdx]];
    _drawFrame(ctx, frames[frameIdx], canvas.width, canvas.height);
    label.textContent = words[wordIdx].toUpperCase();
    frameIdx++;
    if (frameIdx >= frames.length) {
      timer = setTimeout(_advanceWord, INTER_WORD_PAUSE_MS);
    } else {
      timer = setTimeout(_tick, 1000 / FPS);
    }
  }

  function _advanceWord() {
    wordIdx++;
    frameIdx = 0;
    if (wordIdx >= words.length) {
      state = "finished";
      _setStatus("Finished — press Replay to watch again");
      _updateButtons();
      return;
    }
    _tick();
  }

  /**
   * @param {string} containerId
   * @param {string[]} newWords - words to sign, in order (unmatched
   *        words are filtered out here; the caller doesn't need to
   *        pre-filter against sign_animations.json)
   */
  async function play(containerId, newWords) {
    if (!SIGN_ANIMATIONS) await loadSignAnimations();
    if (!_ensureUI(containerId)) return;

    clearTimeout(timer);
    const requested = (newWords || []).map(w => w.toLowerCase());
    words   = requested.filter(w => SIGN_ANIMATIONS[w]);
    skipped = requested.filter(w => !SIGN_ANIMATIONS[w]);
    wordIdx = 0;
    frameIdx = 0;

    if (words.length === 0) {
      state = "empty";
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      label.textContent = "";
      _setStatus(
        requested.length === 0
          ? "No signs to display."
          : `No sign animation available for: ${requested.join(", ")} (not in current 80-word vocabulary)`
      );
      _updateButtons();
      return;
    }

    state = "playing";
    _setStatus(
      skipped.length > 0
        ? `Playing ${words.length} of ${requested.length} words — no animation for: ${skipped.join(", ")}`
        : `Playing ${words.length} word${words.length > 1 ? "s" : ""}…`
    );
    _updateButtons();
    _tick();
  }

  function pause() {
    if (state !== "playing") return;
    clearTimeout(timer);
    state = "paused";
    _setStatus("Paused");
    _updateButtons();
  }

  function resume() {
    if (state !== "paused") return;
    state = "playing";
    _setStatus(`Playing ${words.length} word${words.length > 1 ? "s" : ""}…`);
    _updateButtons();
    _tick();
  }

  function replay() {
    if (words.length === 0) return;
    clearTimeout(timer);
    wordIdx = 0;
    frameIdx = 0;
    state = "playing";
    _setStatus(`Replaying ${words.length} word${words.length > 1 ? "s" : ""}…`);
    _updateButtons();
    _tick();
  }

  function stop() {
    clearTimeout(timer);
    wordIdx = 0;
    frameIdx = 0;
    state = "idle";
    if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (label) label.textContent = "";
    _setStatus("Stopped");
    _updateButtons();
  }

  /** Single button-friendly entry point: resumes if paused, else (re)plays from the start. */
  function playOrResume() {
    if (state === "paused") resume();
    else if (state === "finished" || state === "idle") replay();
  }

  function _updateButtons() {
    const playBtn  = document.getElementById("signPlayBtn");
    const pauseBtn = document.getElementById("signPauseBtn");
    const stopBtn  = document.getElementById("signStopBtn");
    if (!playBtn || !pauseBtn || !stopBtn) return; // controls not present on this page

    const hasWords = words.length > 0;
    playBtn.disabled  = !hasWords || state === "playing";
    playBtn.textContent =
      state === "paused" ? "▶️ Resume" : state === "finished" ? "🔁 Replay" : "▶️ Play";
    pauseBtn.disabled = state !== "playing";
    stopBtn.disabled  = !hasWords || state === "idle";
  }

  return { play, pause, resume, replay, stop, playOrResume };
})();

// Backward-compatible one-shot wrapper (used by anything not yet updated
// to the stateful API).
function playSignSequence(containerId, words) {
  return SignAnimationPlayer.play(containerId, words);
}

window.loadSignAnimations   = loadSignAnimations;
window.playSignSequence     = playSignSequence;
window.SignAnimationPlayer  = SignAnimationPlayer;