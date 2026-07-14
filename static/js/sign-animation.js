/**
 * SignBridge AI — Sign Animation Renderer
 * Replaces the emoji-based voice->sign strip with an animated hand/arm
 * skeleton, driven by real recorded landmark sequences (see
 * extract_sign_keyframes.py) — not fabricated poses, not emoji.
 *
 * Load sign_animations.json once at startup, then call
 * playSignSequence(containerId, ["help","water"]) whenever
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
 * Play an animated skeleton for a sequence of words, one after another.
 * Words not found in SIGN_ANIMATIONS are skipped with a console warning
 * (rather than silently failing) so gaps in your vocabulary are visible
 * during testing.
 */
async function playSignSequence(containerId, words, fps = 14) {
  if (!SIGN_ANIMATIONS) await loadSignAnimations();

  const container = document.getElementById(containerId);
  container.innerHTML = "";
  const canvas = document.createElement("canvas");
  canvas.width = 320;
  canvas.height = 320;
  canvas.className = "sign-anim-canvas";
  container.appendChild(canvas);
  const ctx = canvas.getContext("2d");

  const label = document.createElement("div");
  label.className = "sign-anim-label";
  container.appendChild(label);

  for (const word of words) {
    const key = word.toLowerCase();
    const frames = SIGN_ANIMATIONS[key];
    if (!frames) {
      console.warn(`No animation for "${word}" — not in trained vocabulary`);
      continue;
    }
    label.textContent = word.toUpperCase();
    await _playFrames(ctx, frames, canvas.width, canvas.height, fps);
    await _sleep(200); // brief pause between words
  }
}

function _playFrames(ctx, frames, w, h, fps) {
  return new Promise(resolve => {
    let i = 0;
    const interval = setInterval(() => {
      _drawFrame(ctx, frames[i], w, h);
      i++;
      if (i >= frames.length) {
        clearInterval(interval);
        resolve();
      }
    }, 1000 / fps);
  });
}

function _sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

window.loadSignAnimations = loadSignAnimations;
window.playSignSequence = playSignSequence;
