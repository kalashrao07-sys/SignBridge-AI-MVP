/**
 * SignBridge AI — Main Frontend Application
 * Direction 1: Sign Language → Knowledge Engine → Voice
 * Direction 2: Speech → Knowledge Engine → Smart Display + Sign Sequence
 *
 * Knowledge Engine is fully local (see backend knowledge_base.py) —
 * no external API, no network dependency, no rate limits.
 */

// ─── State ────────────────────────────────────────────────────────────
let camera        = null;
let hands         = null;
let canvasCtx     = null;
let classifier    = new GestureClassifier();
let lastAudio     = null;
let mediaRec      = null;
let audioChunks   = [];
let cameraActive  = false;
let kbQueryCount  = 0;   // tracks successful Knowledge Engine matches this session

const LANG_BCP47 = { en: "en-IN", hi: "hi-IN", kn: "kn-IN" };

// ─── Utilities ────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const isMobile = () => window.innerWidth <= 768;

function getOptimalCanvasSize() {
  if (window.innerWidth <= 480) return { width: 320, height: 240 };
  if (window.innerWidth <= 768) return { width: 480, height: 360 };
  return { width: 640, height: 480 };
}

function getLang() { return $("langSelect").value; }

function setStatus(msg, type = "info") {
  const bar = $("statusMsg");
  if (!bar) return;
  bar.textContent = msg;
  bar.className   = `status-${type}`;
}

// Small helper so we never build innerHTML out of raw, unescaped values.
function escapeHtml(str) {
  return String(str ?? "").replace(/[&<>"']/g, ch => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[ch]));
}

// ─── Window resize ────────────────────────────────────────────────────
window.addEventListener("resize", () => {
  if (cameraActive && camera) {
    const size   = getOptimalCanvasSize();
    const canvas = $("signCanvas");
    if (canvas.width !== size.width || canvas.height !== size.height) {
      canvas.width  = size.width;
      canvas.height = size.height;
    }
  }
});

document.addEventListener("touchstart", () => {}, { passive: true });


// ═══════════════════════════════════════════════════════════════════
//  KNOWLEDGE ENGINE PANEL
// ═══════════════════════════════════════════════════════════════════

function updateKnowledgeInsights(topic, response, category, success) {
  const panel = $("knowledgeInsightsPanel");
  if (!panel) return;

  if (!topic) return;

  panel.style.display = "block";

  $("kiTopic").textContent    = topic;
  $("kiResponse").textContent = response || "No matching entry in the Knowledge Engine.";

  const catEl       = $("kiCategory");
  catEl.textContent = (category || "general").toUpperCase();
  catEl.className   = "ki-category ki-cat-" + (category || "general");

  const dotEl = $("kbStatusDot");
  if (dotEl) dotEl.style.color = success ? "#44ff88" : "#ff6b6b";

  if (success) {
    kbQueryCount++;
    const countEl = $("kbQueryCount");
    if (countEl) countEl.textContent = kbQueryCount;
  }
}

function showKnowledgeLog(panel, input, topic, response, method, output) {
  const el = $(panel === "sign" ? "knowledgeLog" : "knowledgeLogSpeech");
  if (!el) return;
  el.style.display = "block";
  el.innerHTML = `
    <div class="kb-row">
      <span class="kb-label">📥 Input</span>
      <span class="kb-val">"${escapeHtml(input || "—")}"</span>
    </div>
    <div class="kb-row">
      <span class="kb-label">🔍 Topic Matched</span>
      <span class="kb-val kb-topic">${escapeHtml(topic || "—")}</span>
    </div>
    <div class="kb-row">
      <span class="kb-label">📤 Knowledge Engine</span>
      <span class="kb-val kb-resp">${escapeHtml(response || "—")}</span>
    </div>
    ${output ? `<div class="kb-row">
      <span class="kb-label">✅ Output</span>
      <span class="kb-val kb-out">"${escapeHtml(output)}"</span>
    </div>` : ""}
    ${method ? `<div class="kb-method">via ${
      method === "knowledge_base" ? "🧠 Knowledge Engine" : "⚙️ Rule Engine"
    }</div>` : ""}
  `;
}


// ═══════════════════════════════════════════════════════════════════
//  VOICE → SIGN  (sign sequence strip)
// ═══════════════════════════════════════════════════════════════════

/**
 * Render the ordered list of matched signs returned by the backend as a
 * strip of emoji chips under the Smart Display. Words with no known sign
 * are simply absent from `sequence` — the Smart Display text still shows
 * the full sentence.
 */
function renderSignSequence(sequence) {
  const wrap = $("signSequenceStrip");
  if (!wrap) return;

  if (!sequence || sequence.length === 0) {
    wrap.style.display = "none";
    wrap.innerHTML = "";
    return;
  }

  wrap.style.display = "flex";
  wrap.innerHTML = sequence.map(item => `
    <div class="sign-chip" title="${escapeHtml(item.desc || item.sign)}">
      <span class="sign-chip-emoji">${item.emoji || "❓"}</span>
      <span class="sign-chip-word">${escapeHtml(item.sign)}</span>
    </div>
  `).join("");
}


// ═══════════════════════════════════════════════════════════════════
//  MEDIAPIPE HANDS
// ═══════════════════════════════════════════════════════════════════

function initMediaPipe() {
  const videoEl  = $("signVideo");
  const canvasEl = $("signCanvas");
  canvasCtx      = canvasEl.getContext("2d");

  const size      = getOptimalCanvasSize();
  canvasEl.width  = size.width;
  canvasEl.height = size.height;

  hands = new Hands({
    locateFile: file =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`
  });

  hands.setOptions({
    maxNumHands:            1,
    modelComplexity:        isMobile() ? 0 : 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence:  0.6,
  });

  hands.onResults(onHandResults);

  camera = new Camera(videoEl, {
    onFrame: async () => { await hands.send({ image: videoEl }); },
    width:  size.width,
    height: size.height,
  });
}

function onHandResults(results) {
  const canvas = $("signCanvas");
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
  canvasCtx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const landmarks = results.multiHandLandmarks[0];

    drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS,
      { color: "#00B4D8", lineWidth: 2 });
    drawLandmarks(canvasCtx, landmarks,
      { color: "#FFFFFF", fillColor: "#00B4D8", lineWidth: 1, radius: 4 });

    const result = classifier.classify(landmarks);
    updateDetectionUI(result);

    const phrase = classifier.push(result);
    updateBufferUI();
    if (phrase) processSignPhrase(phrase);

  } else {
    const phrase = classifier.push(null);
    updateDetectionUI(null);
    if (phrase) processSignPhrase(phrase);
  }

  canvasCtx.restore();
}

function updateDetectionUI(result) {
  if (result && result.sign && result.sign !== "UNKNOWN" && result.confidence > 0.6) {
    $("detectedSign").textContent = `${result.emoji || ""} ${result.sign}`;
    $("detectedConf").textContent = `${Math.round(result.confidence * 100)}%`;
    $("detectedDesc").textContent = result.desc || "";
  } else {
    $("detectedSign").textContent = "No hand";
    $("detectedConf").textContent = "—";
    $("detectedDesc").textContent = "Show your hand to the camera";
  }
  $("detectionBadge").style.display = "flex";
}

function updateBufferUI() {
  const buf   = classifier.getBuffer();
  const chips = $("bufferChips");
  chips.innerHTML = "";
  buf.forEach(sign => {
    const chip       = document.createElement("span");
    chip.className   = "chip";
    chip.textContent = sign;
    chips.appendChild(chip);
  });
}


// ═══════════════════════════════════════════════════════════════════
//  CAMERA CONTROLS
// ═══════════════════════════════════════════════════════════════════

async function startCamera() {
  setStatus("Starting camera…");
  try {
    initMediaPipe();
    await camera.start();
    cameraActive = true;

    $("signVideo").style.display  = "none";
    $("signCanvas").style.display = "block";
    $("camOverlay").style.display = "none";
    $("startCamBtn").disabled     = true;
    $("stopCamBtn").disabled      = false;
    setStatus("📷 Camera active — show your hand signs", "success");
  } catch (e) {
    setStatus("❌ Camera error: " + e.message, "error");
  }
}

function stopCamera() {
  if (camera) camera.stop();
  cameraActive = false;
  $("signCanvas").style.display = "none";
  $("camOverlay").style.display = "flex";
  $("startCamBtn").disabled     = false;
  $("stopCamBtn").disabled      = true;
  setStatus("Camera stopped.");
}

async function flushBuffer() {
  const phrase = classifier.flush();
  if (phrase) await processSignPhrase(phrase);
}


// ═══════════════════════════════════════════════════════════════════
//  DIRECTION 1: SIGN → KNOWLEDGE ENGINE → VOICE
// ═══════════════════════════════════════════════════════════════════

async function processSignPhrase(phrase) {
  setStatus(`🧠 Processing: "${phrase}"…`);

  $("kbContext").style.display  = "none";
  $("kbContextText").textContent = "";

  showKnowledgeLog("sign", phrase, "Looking up…", "…");

  try {
    const res  = await fetch("/api/sign/process", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ phrase, lang: getLang() }),
    });
    const data = await res.json();

    // Fix: fetch() does not throw on 4xx/5xx — without this check a
    // backend error response (e.g. "No phrase provided") silently
    // rendered as "undefined" in every field below.
    if (!res.ok || data.error) {
      setStatus("❌ " + (data.error || "Request failed"), "error");
      showKnowledgeLog("sign", phrase, "—", data.error || "Request failed", "rules", "—");
      return;
    }

    showKnowledgeLog(
      "sign",
      data.original,
      data.kb_topic,
      data.kb_response,
      data.method,
      data.corrected
    );

    updateKnowledgeInsights(
      data.kb_topic,
      data.kb_response,
      data.kb_category,
      data.kb_success
    );

    $("signResult").style.display      = "flex";
    $("rawPhrase").textContent         = data.original;
    $("correctedSentence").textContent = data.corrected;

    $("methodTag").textContent = data.method === "knowledge_base"
      ? "✅ Knowledge Engine" : "⚙️ Rule-based";
    $("methodTag").className = "method-tag " +
      (data.method === "knowledge_base" ? "knowledge" : "rules");

    if (data.translated && getLang() !== "en") {
      $("translatedRow").style.display    = "flex";
      $("translatedSentence").textContent = data.translated;
    } else {
      $("translatedRow").style.display = "none";
    }

    $("emergencyAlert").style.display = data.emergency ? "block" : "none";

    if (data.kb_response) {
      $("kbContext").style.display   = "block";
      $("kbContextText").textContent = data.kb_response;
    }

    lastAudio = data.audio_b64;
    $("playAudioBtn").disabled = !lastAudio;

    classifier.clearBuffer();
    updateBufferUI();

    setStatus(
      data.emergency ? "🚨 Emergency detected!" : `✅ Processed: "${data.corrected}"`,
      data.emergency ? "error" : "success"
    );

  } catch (e) {
    setStatus("❌ Error: " + e.message, "error");
    showKnowledgeLog("sign", phrase, "—", "Request failed", "rules", "—");
  }
}

function playAudio() {
  if (!lastAudio) return;
  const audio = $("ttsAudio");
  audio.src   = `data:audio/mp3;base64,${lastAudio}`;
  audio.play();
}


// ═══════════════════════════════════════════════════════════════════
//  DIRECTION 2: SPEECH → KNOWLEDGE ENGINE → SMART DISPLAY + SIGN SEQUENCE
// ═══════════════════════════════════════════════════════════════════

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRec     = new MediaRecorder(stream);
    audioChunks  = [];

    mediaRec.ondataavailable = e => audioChunks.push(e.data);
    mediaRec.onstop = () => {
      stream.getTracks().forEach(t => t.stop());
      $("startRecBtn").disabled  = false;
      $("stopRecBtn").disabled   = true;
      $("micRing").classList.remove("recording");
      $("micStatus").textContent = "Processing…";
    };

    mediaRec.start();
    startWebSpeech();

    $("startRecBtn").disabled  = true;
    $("stopRecBtn").disabled   = false;
    $("micRing").classList.add("recording");
    $("micStatus").textContent = "Listening…";
    setStatus("🎙️ Recording…");

  } catch (e) {
    setStatus("❌ Mic error: " + e.message, "error");
  }
}

function stopRecording() {
  if (mediaRec && mediaRec.state !== "inactive") mediaRec.stop();
  if (window._speechRec) window._speechRec.stop();
  $("startRecBtn").disabled  = false;
  $("stopRecBtn").disabled   = true;
  $("micRing").classList.remove("recording");
  $("micStatus").textContent = "Press Record";
}

function startWebSpeech() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    setStatus("⚠️ Speech API not supported — use Chrome", "error");
    return;
  }

  const rec          = new SR();
  rec.lang           = LANG_BCP47[getLang()] || "en-IN";
  rec.continuous     = false;
  rec.interimResults = true;
  window._speechRec  = rec;

  rec.onresult = async (e) => {
    let final = "", interim = "";
    for (const r of e.results) {
      r.isFinal ? (final += r[0].transcript) : (interim += r[0].transcript);
    }
    const text = (final || interim).trim();
    $("transcriptText").textContent = text;
    $("transcriptText").className   = "transcript-text has-text";

    if (final) {
      await processSpeechText(final.trim());
      stopRecording();
    }
  };

  rec.onerror = e => setStatus("Speech error: " + e.error, "error");
  rec.start();
}

async function processSpeechText(text) {
  setStatus(`🧠 Processing: "${text}"…`);
  showKnowledgeLog("speech", text, "Looking up…", "…");

  try {
    const res  = await fetch("/api/speech/process", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text, lang: getLang() }),
    });
    const data = await res.json();

    // Fix: same fetch-error-handling gap as processSignPhrase above.
    if (!res.ok || data.error) {
      setStatus("❌ " + (data.error || "Request failed"), "error");
      showKnowledgeLog("speech", text, "—", data.error || "Request failed", "rules", "—");
      renderSignSequence([]);
      return;
    }

    showKnowledgeLog(
      "speech",
      data.original,
      data.kb_topic,
      data.kb_response,
      data.method,
      data.display
    );

    updateKnowledgeInsights(
      data.kb_topic,
      data.kb_response,
      data.kb_category,
      data.kb_success
    );

    $("smartDisplay").style.display = "block";
    $("smartInner").textContent     = data.display;
    $("alertIndicator").className   = "alert-indicator " +
      (data.emergency ? "emergency" : "");

    // Voice → Sign: render the matched sign sequence strip.
    renderSignSequence(data.sign_sequence);

    if (data.emergency) {
      $("emergencyCard").style.display = "block";
      $("emergencyBody").textContent   = data.display;
    } else {
      $("emergencyCard").style.display = "none";
    }

    if (data.translated && getLang() !== "en") {
      $("speechTransRow").style.display = "flex";
      $("speechTransText").textContent  = data.translated;
    } else {
      $("speechTransRow").style.display = "none";
    }

    $("micStatus").textContent = "Press Record";
    setStatus(
      data.emergency ? "🚨 Emergency!" : `✅ Displayed: "${data.display}"`,
      data.emergency ? "error" : "success"
    );

  } catch (e) {
    setStatus("❌ Error: " + e.message, "error");
    showKnowledgeLog("speech", text, "—", "Request failed", "rules", "—");
  }
}


// ═══════════════════════════════════════════════════════════════════
//  INIT
// ═══════════════════════════════════════════════════════════════════

document.addEventListener("DOMContentLoaded", () => {
  setStatus("🌉 SignBridge AI ready — start camera or record speech");
});
