/**
 * SignBridge AI — Calibration UI hookup
 * Add this <script src="/static/js/calibration.js"></script> AFTER
 * gesture.js and app.js in templates/index.html.
 *
 * REQUIRED ONE-LINE PATCH in app.js: right after
 *   let classifier = new GestureClassifier();
 * add:
 *   window.classifier = classifier;
 * ('let' declarations don't auto-attach to window like 'var' does.)
 *
 * Also requires app.js's onHandResults to be a plain top-level
 * function declaration (it already is) so this file can wrap it to
 * capture the latest landmarks each frame.
 */

let latestLandmarks = null;

// app.js's onHandResults already runs on every frame — we tap into it
// by wrapping the existing function rather than editing app.js again.
(function hookLandmarkCapture() {
  const original = window.onHandResults;
  if (typeof original !== "function") return;
  window.onHandResults = function (results) {
    latestLandmarks = (results.multiHandLandmarks && results.multiHandLandmarks[0]) || null;
    original(results);
  };
})();

function refreshCalibCount() {
  const select = document.getElementById("calibSignSelect");
  const counts = window.classifier.getSampleCounts();
  const label = select.value;
  document.getElementById("calibCount").textContent =
    `${counts[label] || 0} samples recorded`;
}

document.addEventListener("DOMContentLoaded", () => {
  const select   = document.getElementById("calibSignSelect");
  const recordBtn = document.getElementById("recordSampleBtn");
  const clearSignBtn = document.getElementById("clearSignBtn");
  const clearAllBtn  = document.getElementById("clearAllBtn");
  const status   = document.getElementById("calibStatus");

  if (!select || !recordBtn) return; // panel not present on this page

  select.addEventListener("change", refreshCalibCount);

  recordBtn.addEventListener("click", () => {
    if (!latestLandmarks) {
      status.textContent = "⚠️ No hand detected — start the camera and show your hand first.";
      return;
    }
    const label = select.value;
    window.classifier.addTrainingSample(label, latestLandmarks);
    refreshCalibCount();
    status.textContent = `✅ Sample recorded for ${label}.`;
  });

  clearSignBtn.addEventListener("click", () => {
    window.classifier.clearTraining(select.value);
    refreshCalibCount();
    status.textContent = `Cleared samples for ${select.value}.`;
  });

  clearAllBtn.addEventListener("click", () => {
    window.classifier.clearTraining();
    refreshCalibCount();
    status.textContent = "All custom training cleared — using default recognition.";
  });

  refreshCalibCount();
});
