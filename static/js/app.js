/**
 * SignBridge AI — Main Frontend Application
 */

// ─── State ────────────────────────────────────────────────────────────
let camera      = null;
let hands       = null;
let canvasCtx   = null;
let classifier  = new GestureClassifier();
let lastAudio   = null;
let mediaRec    = null;
let audioChunks = [];
let cameraActive = false;

const LANG_BCP47 = { en: "en-IN", hi: "hi-IN", kn: "kn-IN" };

// ─── Mobile detection ──────────────────────────────────────────────────
const isMobile = () => window.innerWidth <= 768;
const getOptimalCanvasSize = () => {
  if (window.innerWidth <= 480) {
    return { width: 320, height: 240 };
  } else if (window.innerWidth <= 768) {
    return { width: 480, height: 360 };
  }
  return { width: 640, height: 480 };
};

// ─── DOM refs ─────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// ─── Status ───────────────────────────────────────────────────────────
function setStatus(msg, type = "info") {
  const bar = $("statusMsg");
  bar.textContent = msg;
  bar.className = `status-${type}`;
}

function getLang() { return $("langSelect").value; }

// ─── Window resize handler ────────────────────────────────────────────
window.addEventListener("resize", () => {
  if (cameraActive && camera) {
    const size = getOptimalCanvasSize();
    const canvas = $("signCanvas");
    if (canvas.width !== size.width || canvas.height !== size.height) {
      canvas.width = size.width;
      canvas.height = size.height;
    }
  }
});

// ─── Touch-friendly interactions ───────────────────────────────────────
document.addEventListener("touchstart", () => {}, { passive: true });

// ─── MediaPipe Hands Setup ────────────────────────────────────────────
function initMediaPipe() {
  const videoEl  = $("signVideo");
  const canvasEl = $("signCanvas");
  canvasCtx      = canvasEl.getContext("2d");

  // Optimize canvas size for mobile
  const size = getOptimalCanvasSize();
  canvasEl.width = size.width;
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
    width: size.width,
    height: size.height,
  });
}

// ─── Hand Results Callback ────────────────────────────────────────────
function onHandResults(results) {
  const canvas = $("signCanvas");
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw video frame
  canvasCtx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const landmarks = results.multiHandLandmarks[0];

    // Draw connections
    drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS,
      { color: "#00B4D8", lineWidth: 2 });

    // Draw landmarks
    drawLandmarks(canvasCtx, landmarks,
      { color: "#FFFFFF", fillColor: "#00B4D8", lineWidth: 1, radius: 4 });

    // Classify
    const result = classifier.classify(landmarks);
    updateDetectionUI(result);

    // Push to buffer
    const phrase = classifier.push(result);
    updateBufferUI();

    if (phrase) {
      processSignPhrase(phrase);
    }
  } else {
    // No hand
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
    $("detectionBadge").style.display = "flex";
  } else {
    $("detectedSign").textContent = "No hand";
    $("detectedConf").textContent = "—";
    $("detectedDesc").textContent = "Show your hand to the camera";
    $("detectionBadge").style.display = "flex";
  }
}

function updateBufferUI() {
  const buf   = classifier.getBuffer();
  const chips = $("bufferChips");
  chips.innerHTML = "";
  buf.forEach(sign => {
    const chip     = document.createElement("span");
    chip.className = "chip";
    chip.textContent = sign;
    chips.appendChild(chip);
  });
}

// ─── Camera Controls ──────────────────────────────────────────────────
async function startCamera() {
  setStatus("Starting camera…");
  try {
    initMediaPipe();
    await camera.start();
    cameraActive = true;

    $("signVideo").style.display   = "none";
    $("signCanvas").style.display  = "block";
    $("camOverlay").style.display  = "none";
    $("startCamBtn").disabled      = true;
    $("stopCamBtn").disabled       = false;
    setStatus("📷 Camera active — show your hand signs", "success");
  } catch (e) {
    setStatus("❌ Camera error: " + e.message, "error");
  }
}

function stopCamera() {
  if (camera) camera.stop();
  cameraActive = false;
  $("signCanvas").style.display  = "none";
  $("camOverlay").style.display  = "flex";
  $("startCamBtn").disabled      = false;
  $("stopCamBtn").disabled       = true;
  setStatus("Camera stopped.");
}

async function flushBuffer() {
  const phrase = classifier.flush();
  if (phrase) await processSignPhrase(phrase);
}

// ─── Sign Processing (Direction 1) ────────────────────────────────────
async function processSignPhrase(phrase) {
  setStatus(`🧠 Wolfram processing: "${phrase}"…`);

  $("wfContext").style.display = "none";
  $("wfContextText").textContent = "";
  // Show Wolfram panel loading
  showWolframLog("sign", phrase, "Processing…", "…");

  try {
    const res  = await fetch("/api/sign/process", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ phrase, lang: getLang() }),
    });
    const data = await res.json();

    // Update Wolfram log
    showWolframLog("sign",
      data.original,
      data.wolfram_query,
      data.wolfram_response,
      data.wolfram_method,
      data.corrected
    );

    updateWolframInsights(
  data.wolfram_query,
  data.wolfram_response,
  data.wolfram_category,
  data.wolfram_success
  );

    // Update result panel
    $("signResult").style.display     = "flex";
    $("rawPhrase").textContent        = data.original;
    $("correctedSentence").textContent = data.corrected;
    $("methodTag").textContent        = data.wolfram_method === "wolfram"
                                        ? "✅ Wolfram Alpha" : "⚙️ Rule-based";
    $("methodTag").className          = "method-tag " +
                                        (data.wolfram_method === "wolfram" ? "wolfram" : "rules");

    if (data.translated && getLang() !== "en") {
      $("translatedRow").style.display      = "flex";
      $("translatedSentence").textContent   = data.translated;
    } else {
      $("translatedRow").style.display = "none";
    }

    $("emergencyAlert").style.display = data.emergency ? "block" : "none";
    if (data.wf_context) {
      $("wfContext").style.display   = "block";
      $("wfContextText").textContent = data.wf_context;
    }

    lastAudio = data.audio_b64;
    $("playAudioBtn").disabled = !lastAudio;

    classifier.clearBuffer();
    updateBufferUI();

    setStatus(data.emergency
      ? "🚨 Emergency detected!"
      : `✅ Processed: "${data.corrected}"`,
      data.emergency ? "error" : "success");

  } catch (e) {
    setStatus("❌ Error: " + e.message, "error");
  }
}

function showWolframLog(panel, input, query, response, method, output) {
  const el = $(panel === "sign" ? "wolframLog" : "wolframLogSpeech");
  if (!el) return;
  el.style.display = "block";
  el.innerHTML = `
    <div class="wf-row"><span class="wf-label">📥 Input</span>
      <span class="wf-val">"${input || "—"}"</span></div>
    <div class="wf-row"><span class="wf-label">🔍 Query</span>
      <span class="wf-val wf-query">${query || "—"}</span></div>
    <div class="wf-row"><span class="wf-label">📤 Wolfram</span>
      <span class="wf-val wf-resp">${response || "—"}</span></div>
    ${output ? `<div class="wf-row"><span class="wf-label">✅ Output</span>
      <span class="wf-val wf-out">"${output}"</span></div>` : ""}
    ${method ? `<div class="wf-method">via ${method === "wolfram" ? "🧠 Wolfram Alpha API" : "⚙️ Rule Engine"}</div>` : ""}
  `;
}

// ── New: populate the full-width Wolfram Insights panel ──
let wolframQueryCount = 0;

function updateWolframInsights(query, response, category, success) {
  if (!query) return;
  const panel = $("wolframInsightsPanel");
  panel.style.display = "block";

  $("wiQuery").textContent    = query;
  $("wiResponse").textContent = response || "No response from Wolfram";

  const catEl = $("wiCategory");
  catEl.textContent = category || "general";
  catEl.className   = "wf-insight-category wf-cat-" + (category || "info");

  if (success) {
    wolframQueryCount++;
    $("wolframQueryCount").textContent = wolframQueryCount;
  }
}

function playAudio() {
  if (!lastAudio) return;
  const audio = $("ttsAudio");
  audio.src   = `data:audio/mp3;base64,${lastAudio}`;
  audio.play();
}

// ─── Speech Recording (Direction 2) ──────────────────────────────────
async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRec     = new MediaRecorder(stream);
    audioChunks  = [];
    mediaRec.ondataavailable = e => audioChunks.push(e.data);
    mediaRec.onstop = async () => {
      const blob = new Blob(audioChunks, { type: "audio/webm" });

      // Try Web Speech API first (better accuracy, free, no backend)
      stream.getTracks().forEach(t => t.stop());
      $("startRecBtn").disabled = false;
      $("stopRecBtn").disabled  = true;
      $("micRing").classList.remove("recording");
      $("micStatus").textContent = "Processing…";
    };
    mediaRec.start();

    // Start Web Speech API simultaneously
    startWebSpeech();

    $("startRecBtn").disabled = true;
    $("stopRecBtn").disabled  = false;
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
  $("startRecBtn").disabled = false;
  $("stopRecBtn").disabled  = true;
  $("micRing").classList.remove("recording");
  $("micStatus").textContent = "Press Record";
}

function startWebSpeech() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { setStatus("⚠️ Speech API not supported in this browser — use Chrome", "error"); return; }

  const rec       = new SR();
  rec.lang        = LANG_BCP47[getLang()] || "en-IN";
  rec.continuous  = false;
  rec.interimResults = true;
  window._speechRec  = rec;

  rec.onresult = async (e) => {
    let final = "", interim = "";
    for (const r of e.results) {
      (r.isFinal ? (final += r[0].transcript) : (interim += r[0].transcript));
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
  setStatus(`🧠 Wolfram simplifying: "${text}"…`);
  showWolframLog("speech", text, "Processing…", "…");

  try {
    const res  = await fetch("/api/speech/process", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text, lang: getLang() }),
    });
    const data = await res.json();

    showWolframLog("speech",
      data.original,
      data.wolfram_query,
      data.wolfram_response,
      data.wolfram_method,
      data.display
    );

    // Smart display
    $("smartDisplay").style.display = "block";
    $("smartInner").textContent     = data.display;
    $("alertIndicator").className   = "alert-indicator " + (data.emergency ? "emergency" : "");

    if (data.emergency) {
      $("emergencyCard").style.display = "block";
      $("emergencyBody").textContent   = data.display;
    } else {
      $("emergencyCard").style.display = "none";
    }

    if (data.translated && getLang() !== "en") {
      $("speechTransRow").style.display  = "flex";
      $("speechTransText").textContent   = data.translated;
    } else {
      $("speechTransRow").style.display  = "none";
    }

    $("micStatus").textContent = "Press Record";
    setStatus(data.emergency ? "🚨 Emergency!" : `✅ Displayed: "${data.display}"`,
              data.emergency ? "error" : "success");

  } catch (e) {
    setStatus("❌ Error: " + e.message, "error");
  }
}

// ─── Init ─────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  setStatus("🌉 SignBridge AI ready — start camera or record speech");
});
