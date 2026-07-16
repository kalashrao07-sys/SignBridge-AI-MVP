"""
SignBridge AI — Combined Experiment (run once, evaluate, decide)
====================================================================
Bundles three well-established, low-risk improvements into a single
run rather than testing them one at a time:
  1. More training data per class (uses the REAL available count,
     printed up front, not a guess)
  2. Data augmentation — mirror / jitter / scale — applied ONLY to
     the training split (after the split, not before) so validation
     numbers stay honest and uncontaminated
  3. A bidirectional GRU instead of Conv+GlobalPool — GRUs preserve
     temporal order across the whole sequence, which matters more
     now that hand-only data leans entirely on motion cues

STEP 1: run preprocess_signs_v3_hands_only.py first (bump
        MAX_SAMPLES_PER_SIGN there to whatever this script's first
        printout shows as the real per-sign ceiling).
STEP 2: run this script.
"""

import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

CACHE_PATH = "sign_training_data.npz"
LABELS_PATH = "label_names.json"
MODEL_PATH = "sign_model_v3.h5"

EPOCHS = 80
BATCH_SIZE = 32
AUG_MULTIPLIER = 3         # extra augmented copies per real training sample
RELIABLE_THRESHOLD = 0.6


# ─── Augmentation (train split only) ───────────────────────────────────

def augment(X, y, multiplier=AUG_MULTIPLIER, noise_std=0.015, scale_range=0.08, shift_range=0.03):
    """X: (N, T, 84) already-normalized hand landmark sequences.
    Layout per timestep: 42 (x,y) pairs = left hand[0:21] + right hand[21:42]."""
    N, T, F = X.shape
    all_X, all_y = [X], [y]

    for _ in range(multiplier):
        Xc = X.copy()
        Xc += np.random.normal(0, noise_std, Xc.shape).astype(np.float32)

        scale = 1 + np.random.uniform(-scale_range, scale_range, (N, 1, 1)).astype(np.float32)
        Xc *= scale

        pts = Xc.reshape(N, T, 42, 2)
        shift = np.random.uniform(-shift_range, shift_range, (N, 1, 1, 2)).astype(np.float32)
        pts = pts + shift
        Xc = pts.reshape(N, T, F)

        all_X.append(Xc)
        all_y.append(y)

        # Mirrored variant: swap left/right hand blocks, negate x —
        # represents a mirrored signer, doubles effective diversity.
        mirrored = pts.reshape(N, T, 2, 21, 2)[:, :, ::-1, :, :].copy()
        mirrored[:, :, :, :, 0] *= -1
        all_X.append(mirrored.reshape(N, T, F))
        all_y.append(y)

    return np.concatenate(all_X), np.concatenate(all_y)


# ─── Model ──────────────────────────────────────────────────────────────

def build_model(sequence_length, n_features, n_classes):
    inputs = tf.keras.Input(shape=(sequence_length, n_features))
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(48, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32))(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc")],
    )
    return model


def main():
    data = np.load(CACHE_PATH)
    X, y = data["X"], data["y"]
    with open(LABELS_PATH) as f:
        label_names = json.load(f)

    print(f"Real samples available: {len(X)} across {len(label_names)} signs "
          f"({len(X)/len(label_names):.1f} avg/class)")

    # Split BEFORE augmenting — validation set is 100% real, untouched data.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train (real): {len(X_train)}  |  Val (real, untouched): {len(X_val)}")

    X_train_aug, y_train_aug = augment(X_train, y_train)
    print(f"Train (after augmentation): {len(X_train_aug)}")

    model = build_model(X.shape[1], X.shape[2], len(label_names))
    model.summary()

    model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=12, restore_best_weights=True
        )],
    )

    val_loss, val_acc, val_top3 = model.evaluate(X_val, y_val)
    print(f"\nFINAL — val top-1 accuracy: {val_acc:.3f}   val top-3 accuracy: {val_top3:.3f}")

    model.save(MODEL_PATH)

    # Per-class breakdown, same as evaluate_model.py, on the same real val set
    y_pred = np.argmax(model.predict(X_val), axis=1)
    report = classification_report(y_val, y_pred, target_names=label_names, output_dict=True, zero_division=0)

    reliable = [(n, report[n]["recall"]) for n in label_names if report[n]["recall"] >= RELIABLE_THRESHOLD]
    reliable.sort(key=lambda t: -t[1])

    print(f"\n✅ RELIABLE ({len(reliable)}/{len(label_names)}, recall >= {RELIABLE_THRESHOLD:.0%}):")
    for name, recall in reliable:
        print(f"   {name:15s} recall={recall:.2f}")

    with open("reliable_signs_v3.json", "w") as f:
        json.dump([n for n, _ in reliable], f)
    print(f"\nSaved to reliable_signs_v3.json")


if __name__ == "__main__":
    main()
