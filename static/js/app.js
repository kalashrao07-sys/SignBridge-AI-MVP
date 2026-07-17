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
window.classifier = classifier;
let lastAudio     = null;
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
 * Kick off playback of the ordered list of matched signs returned by the
 * backend (sign_vocabulary.py's text_to_sign_sequence — a plain array of
 * lowercase words, e.g. ["hello","food"]). SignAnimationPlayer filters
 * out any words with no animation and surfaces that in its own status
 * line, so the strip stays visible (rather than disappearing) even when
 * nothing in the sentence is in the current 80-word vocabulary — that
 * makes a vocabulary gap look like a vocabulary gap, not a bug.
 */
function renderSignSequence(sequence) {
  const wrap = $("signSequenceStrip");
  if (!wrap) return;

  wrap.style.display = "flex";
  SignAnimationPlayer.play("signSequenceStrip", sequence || []).catch(err =>
    console.error("Sign animation playback failed:", err)
  );
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
    maxNumHands:            2,
    modelComplexity:        isMobile() ? 0 : 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence:  0.6,
  });

  hands.onResults(onHandResults);

  camera = new Camera(videoEl, {
    onFrame: async () => { await hands.send({ image: videoEl }); },
    width:  size.width,
    height: size.height,
    // Prefer the rear/environment camera on mobile — "ideal" (not
    // "exact") so devices without a back camera (most laptops) still
    // fall back to whatever camera IS available instead of failing.
    facingMode: { ideal: "environment" },
  });
}

/**
 * Starts the camera, retrying with the front-facing camera if the
 * rear/environment camera isn't available on this device (e.g. laptops).
 */
async function _startCameraWithFallback() {
  try {
    await camera.start();
  } catch (e) {
    console.warn("Rear camera unavailable, falling back to front camera:", e.message);
    camera = new Camera($("signVideo"), {
      onFrame: async () => { await hands.send({ image: $("signVideo") }); },
      width:  getOptimalCanvasSize().width,
      height: getOptimalCanvasSize().height,
      facingMode: "user",
    });
    await camera.start();
  }
}

function onHandResults(results) {
  const canvas = $("signCanvas");
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
  canvasCtx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {

    pushSequenceFrame(results.multiHandLandmarks);

    // Draw every detected hand
    for (const landmarks of results.multiHandLandmarks) {
        drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
            color: "#00B4D8",
            lineWidth: 2
        });

        drawLandmarks(canvasCtx, landmarks, {
            color: "#FFFFFF",
            fillColor: "#00B4D8",
            lineWidth: 1,
            radius: 4
        });
    }

    // Pass ALL hands to the classifier
    const result = classifier.classify(results.multiHandLandmarks);

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
    await _startCameraWithFallback();
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
  // NOTE: previously this also opened a second, separate microphone
  // stream via getUserMedia + MediaRecorder, whose captured audio
  // (audioChunks) was never actually used anywhere. Having two
  // concurrent exclusive-access audio captures open at once — this one
  // plus SpeechRecognition's own internal capture below — is a known
  // source of "recognition starts but never receives audio" on Android
  // Chrome specifically (desktop Chrome tolerates it via software
  // mixing; Android's audio session handling is stricter). Removed.
  try {
    startWebSpeech();
  } catch (e) {
    setStatus("❌ Mic error: " + e.message, "error");
  }
}

function stopRecording() {
  if (window._speechRec) window._speechRec.stop();
  _resetRecordingUI();
}

function _resetRecordingUI() {
  clearTimeout(window._speechSafetyTimer);
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

  // Guard against a stray double-tap starting a second recognizer while
  // one is already active.
  if (window._speechRec) {
    try { window._speechRec.abort(); } catch (_) { /* already stopped */ }
  }

  const rec          = new SR();
  rec.lang           = LANG_BCP47[getLang()] || "en-IN";
  rec.continuous     = false;
  rec.interimResults = true;
  window._speechRec  = rec;

  let gotFinalResult = false;

  // Full lifecycle logging, as requested — this is the actual diagnostic
  // signal we need: whichever of these DOESN'T fire on the Android
  // device tells us exactly where the pipeline is dying (e.g. onstart
  // firing but onaudiostart never firing would point at mic capture
  // itself, not at SpeechRecognition's language-processing step).
  rec.onstart      = () => console.log("[SpeechRecognition] onstart");
  rec.onaudiostart = () => console.log("[SpeechRecognition] onaudiostart");
  rec.onsoundstart = () => console.log("[SpeechRecognition] onsoundstart");
  rec.onspeechstart = () => console.log("[SpeechRecognition] onspeechstart");
  rec.onspeechend  = () => console.log("[SpeechRecognition] onspeechend");
  rec.onsoundend   = () => console.log("[SpeechRecognition] onsoundend");
  rec.onaudioend   = () => console.log("[SpeechRecognition] onaudioend");
  rec.onnomatch    = () => console.log("[SpeechRecognition] onnomatch — audio captured but not recognized as speech");

  rec.onresult = async (e) => {
    console.log("[SpeechRecognition] onresult", e.results);
    let final = "", interim = "";
    for (const r of e.results) {
      r.isFinal ? (final += r[0].transcript) : (interim += r[0].transcript);
    }
    const text = (final || interim).trim();
    $("transcriptText").textContent = text;
    $("transcriptText").className   = "transcript-text has-text";

    if (final) {
      gotFinalResult = true;
      await processSpeechText(final.trim());
      stopRecording();
    }
  };

  rec.onerror = e => {
    console.log("[SpeechRecognition] onerror", e.error);
    setStatus("Speech error: " + e.error, "error");
    _resetRecordingUI();
  };

  // Critical fix: without this, if recognition ends for ANY reason
  // without producing a final result (common on Android Chrome — e.g.
  // it silently gives up if it never actually captured usable audio),
  // the UI was permanently stuck on "Listening…" with no way out except
  // manually pressing Stop. Android Chrome sometimes reports overall
  // success (no onerror) while genuinely capturing nothing, which is
  // exactly the "stuck on Listening..." symptom — this always resets
  // the UI when the recognizer stops, regardless of why.
  rec.onend = () => {
    console.log("[SpeechRecognition] onend — gotFinalResult:", gotFinalResult);
    if (!gotFinalResult) {
      setStatus("🎙️ No speech detected — try again", "error");
    }
    _resetRecordingUI();
  };

  // Second line of defense: if nothing at all has happened within 10s
  // (not even onspeechstart), force-stop rather than leaving the user
  // staring at "Listening…" indefinitely.
  clearTimeout(window._speechSafetyTimer);
  window._speechSafetyTimer = setTimeout(() => {
    if (window._speechRec === rec && !gotFinalResult) {
      console.log("[SpeechRecognition] safety timeout — forcing stop");
      rec.stop();
    }
  }, 10000);

  rec.start();

  $("startRecBtn").disabled  = true;
  $("stopRecBtn").disabled   = false;
  $("micRing").classList.add("recording");
  $("micStatus").textContent = "Listening…";
  setStatus("🎙️ Recording…");
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
      SignAnimationPlayer.stop();
      $("signSequenceStrip").style.display = "none";
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

document.addEventListener("DOMContentLoaded", () => {
  if (typeof loadSequenceModel === "function") loadSequenceModel();
});

// ═══════════════════════════════════════════════════════════════════
//  INIT
// ═══════════════════════════════════════════════════════════════════

document.addEventListener("DOMContentLoaded", () => {
  setStatus("🌉 SignBridge AI ready — start camera or record speech");
});