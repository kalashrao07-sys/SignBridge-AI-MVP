/**
 * SignBridge AI — Sign Sequence Model (custom GRU inference)
 * ============================================================
 * WHY THIS FILE EXISTS
 * ---------------------
 * The trained model (Keras 3 / TF, Bidirectional-GRU x2 -> Dense -> Dense)
 * uses `reset_after=True` GRU cells — the cuDNN-compatible gate formulation
 * TF/Keras uses by default. TensorFlow.js's built-in `GRU`/`GRUCell` layer
 * ONLY implements the older `reset_after=False` formulation and throws:
 *
 *     GRUCell does not support reset_after parameter set to true.
 *
 * This is a hard incompatibility in tfjs-layers itself (confirmed against
 * the real @tensorflow/tfjs deserializer) — no amount of model.json editing
 * fixes it, and forcing these weights into tfjs's GRU would silently give
 * WRONG predictions rather than an error, because the two formulations
 * compute different math from the same weight shapes.
 *
 * This module bypasses tf.loadLayersModel() entirely for this model: it
 * loads the raw weight tensors straight out of model.json's weightsManifest
 * + the .bin shard (completely unmodified — no retraining, no weight
 * surgery needed, since Masking had zero trainable params and is simply
 * not reproduced here), and runs the forward pass with hand-written ops
 * that match the exact equations the weights were trained with.
 *
 * Requires: a global `tf` (TensorFlow.js core, same CDN script already
 * used elsewhere in this app) and nothing else.
 *
 * Loaded as a classic <script> tag (no ES module syntax), consistent with
 * gesture.js / app.js. Public API is exposed as window.SignSequenceModel.
 */

const UNITS_L1 = 48;   // units per direction, first BiGRU layer
const UNITS_L2 = 32;   // units per direction, second BiGRU layer
const SEQ_LEN  = 32;   // frames per sign (fixed-length, no masking needed)
const FEATURES = 84;   // landmark features per frame

/**
 * Load model.json (for the weightsManifest) + the binary shard, and slice
 * the flat weight buffer into named tf.Tensors.
 *
 * @param {string} baseUrl - folder containing model.json + group1-shard1of1.bin
 */
async function loadSignSequenceModel(baseUrl) {
  const modelJson = await fetch(`${baseUrl}/model.json`).then(r => r.json());
  const binBuf    = await fetch(`${baseUrl}/group1-shard1of1.bin`).then(r => r.arrayBuffer());

  const specs = modelJson.weightsManifest[0].weights;
  const weights = {};
  let offset = 0;
  for (const spec of specs) {
    const numel  = spec.shape.reduce((a, b) => a * b, 1);
    const floats = new Float32Array(binBuf, offset, numel);
    weights[spec.name] = tf.tensor(Array.from(floats), spec.shape, "float32");
    offset += numel * 4;
  }
  return { weights };
}

// ── cuDNN-compatible ("reset_after=True") GRU single time step ─────────
// Gate order follows the Keras convention: z (update), r (reset), h (candidate)
function gruStep(xT, hPrev, kernel, recurrentKernel, bias, units) {
  return tf.tidy(() => {
    const biasInput = bias.slice([0, 0], [1, 3 * units]).reshape([3 * units]);
    const biasRecur = bias.slice([1, 0], [1, 3 * units]).reshape([3 * units]);

    const matrixX = xT.matMul(kernel).add(biasInput);
    const matrixH = hPrev.matMul(recurrentKernel).add(biasRecur);

    const [xz, xr, xh] = tf.split(matrixX, 3, 1);
    const [hz, hr, hh] = tf.split(matrixH, 3, 1);

    const z = tf.sigmoid(xz.add(hz));
    const r = tf.sigmoid(xr.add(hr));
    // Reset gate applied AFTER the recurrent bias is added — this is the
    // defining trait of reset_after=True and the reason a single toggle
    // in tfjs-layers can't represent it.
    const candidate = tf.tanh(xh.add(r.mul(hh)));

    return z.mul(hPrev).add(tf.scalar(1).sub(z).mul(candidate));
  });
}

// Runs one GRU direction over a full sequence already in the time order
// it should be fed (caller reverses the input for the backward direction).
function runGruSequence(seq, kernel, recurrentKernel, bias, units) {
  const [batch, T, inputDim] = seq.shape;
  let h = tf.zeros([batch, units]);
  const outputs = [];
  for (let t = 0; t < T; t++) {
    const xT = seq.slice([0, t, 0], [batch, 1, inputDim]).reshape([batch, inputDim]);
    h = gruStep(xT, h, kernel, recurrentKernel, bias, units);
    outputs.push(h);
  }
  return { all: tf.stack(outputs, 1), last: h }; // all: [batch, T, units]
}

function bidirectionalGru(seq, weights, fwdName, bwdName, units, returnSequences) {
  const fwdK = weights[`${fwdName}/gru_cell/kernel`];
  const fwdR = weights[`${fwdName}/gru_cell/recurrent_kernel`];
  const fwdB = weights[`${fwdName}/gru_cell/bias`];
  const bwdK = weights[`${bwdName}/gru_cell/kernel`];
  const bwdR = weights[`${bwdName}/gru_cell/recurrent_kernel`];
  const bwdB = weights[`${bwdName}/gru_cell/bias`];

  const fwd = runGruSequence(seq, fwdK, fwdR, fwdB, units);

  const seqRev = seq.reverse(1);
  const bwd = runGruSequence(seqRev, bwdK, bwdR, bwdB, units);

  if (returnSequences) {
    const bwdAligned = bwd.all.reverse(1); // re-align to forward time order
    return tf.concat([fwd.all, bwdAligned], 2); // [batch, T, 2*units]
  }
  return tf.concat([fwd.last, bwd.last], 1); // [batch, 2*units]
}

/**
 * Run inference on one sign sequence.
 *
 * @param {{weights: Object}} model - result of loadSignSequenceModel()
 * @param {number[][]} frames - SEQ_LEN arrays of FEATURES landmark values
 *                               (already preprocessed exactly as in training)
 * @returns {Promise<Float32Array>} length-80 softmax class probabilities
 */
async function predictSign(model, frames) {
  const { weights } = model;

  if (frames.length !== SEQ_LEN || frames[0].length !== FEATURES) {
    throw new Error(
      `Expected ${SEQ_LEN}x${FEATURES} input, got ${frames.length}x${frames[0]?.length}`
    );
  }

  const output = tf.tidy(() => {
    const x = tf.tensor(frames, undefined, "float32").reshape([1, SEQ_LEN, FEATURES]);

    const layer1 = bidirectionalGru(x, weights, "forward_gru", "backward_gru", UNITS_L1, true);
    const layer2 = bidirectionalGru(layer1, weights, "forward_gru_1", "backward_gru_1", UNITS_L2, false);

    const dense1 = tf.relu(layer2.matMul(weights["dense/kernel"]).add(weights["dense/bias"]));
    // Dropout is a no-op at inference time — intentionally omitted here.
    const logits = dense1.matMul(weights["dense_1/kernel"]).add(weights["dense_1/bias"]);
    return tf.softmax(logits);
  });

  const probs = await output.data();
  output.dispose();
  return probs;
}

/**
 * Free all loaded weight tensors when the model is no longer needed
 * (e.g. navigating away from the Sign→Speech page).
 */
function disposeSignSequenceModel(model) {
  Object.values(model.weights).forEach(t => t.dispose());
}

// Expose globally, same pattern as window.GestureClassifier in gesture.js
window.SignSequenceModel = {
  load: loadSignSequenceModel,
  predict: predictSign,
  dispose: disposeSignSequenceModel,
};
