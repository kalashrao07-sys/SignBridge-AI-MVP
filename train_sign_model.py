"""
SignBridge AI — Sign Recognition Model Training
=================================================
Trains a small sequence model on landmark data from the Kaggle
"Google - Isolated Sign Language Recognition" (asl-signs) dataset.

WHY THIS SHAPE OF MODEL:
  - Real signs are motions, not static poses, so we train on a
    SEQUENCE of frames (not a single frame like the old bitmask logic).
  - We use Keras specifically because tensorflowjs_converter has
    first-class support for Keras .h5 models — this model needs to
    run in the browser next to your existing MediaPipe pipeline.
  - We use hands + a few arm/pose points (shoulders, elbows, wrists),
    not the full 543 landmarks (which includes 468 face points you
    don't need) — smaller input, faster training, faster inference.

USAGE:
  1. Download & unzip the Kaggle dataset into ./data/
     (train.csv, sign_to_prediction_index_map.json, train_landmark_files/)
  2. Edit SELECTED_SIGNS below with the words you cross-checked against
     sign_to_prediction_index_map.json
  3. python train_sign_model.py
"""

import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ─── Config ────────────────────────────────────────────────────────────
DATA_DIR = "./data"
SEQUENCE_LENGTH = 32          # frames per sample after resampling
EPOCHS = 60
BATCH_SIZE = 16

# EDIT THIS: words you confirmed exist in sign_to_prediction_index_map.json
# AND matter for your emergency/communication use case.
SELECTED_SIGNS = [
    "help", "water", "pain", "yes", "no", "hello", "please",
    "eat", "drink", "sick", "hurt", "more", "bathroom", "hot", "cold",
]

# Landmark indices we keep (MediaPipe Holistic conventions used by the
# dataset). Pose: 11=L-shoulder,12=R-shoulder,13=L-elbow,14=R-elbow,
# 15=L-wrist,16=R-wrist. Hands: all 21 points each.
POSE_KEEP = [11, 12, 13, 14, 15, 16]
N_HAND = 21
N_POSE_KEEP = len(POSE_KEEP)
N_FEATURES = (N_HAND * 2 + N_POSE_KEEP) * 2  # (*2 hands, *2 for x,y)


# ─── Landmark loading & preprocessing ──────────────────────────────────

def load_and_preprocess_sequence(parquet_path: str) -> np.ndarray | None:
    """
    Load one sequence's parquet file, extract hands + arm landmarks,
    normalize, and resample to SEQUENCE_LENGTH frames.
    Returns array of shape (SEQUENCE_LENGTH, N_FEATURES) or None if
    the sequence has no usable hand data.
    """
    df = pd.read_parquet(parquet_path)

    frames = sorted(df["frame"].unique())
    if len(frames) < 2:
        return None

    per_frame = []
    for f in frames:
        fdf = df[df["frame"] == f]

        left = _extract_type(fdf, "left_hand", range(N_HAND))
        right = _extract_type(fdf, "right_hand", range(N_HAND))
        pose = _extract_type(fdf, "pose", POSE_KEEP)

        row = np.concatenate([left, right, pose])  # (N_FEATURES,)
        per_frame.append(row)

    arr = np.stack(per_frame)  # (n_frames, N_FEATURES)

    # Drop sequences where both hands are missing almost everywhere
    hand_cols = arr[:, : N_HAND * 2 * 2]
    if np.isnan(hand_cols).mean() > 0.85:
        return None

    arr = _interpolate_nans(arr)
    arr = _normalize(arr)
    arr = _resample(arr, SEQUENCE_LENGTH)
    return arr


def _extract_type(fdf, type_name, indices):
    sub = fdf[fdf["type"] == type_name].set_index("landmark_index")
    out = []
    for i in indices:
        if i in sub.index:
            out.extend([sub.loc[i, "x"], sub.loc[i, "y"]])
        else:
            out.extend([np.nan, np.nan])
    return np.array(out, dtype=np.float32)


def _interpolate_nans(arr: np.ndarray) -> np.ndarray:
    for col in range(arr.shape[1]):
        series = pd.Series(arr[:, col])
        series = series.interpolate(limit_direction="both")
        arr[:, col] = series.fillna(0.0).to_numpy()
    return arr


def _normalize(arr: np.ndarray) -> np.ndarray:
    # Reference point = midpoint of shoulders (last 4 pose cols: L/R shoulder x,y)
    pose_start = N_HAND * 2 * 2
    l_sh = arr[:, pose_start : pose_start + 2]
    r_sh = arr[:, pose_start + 2 : pose_start + 4]
    center = (l_sh + r_sh) / 2.0
    scale = np.linalg.norm(l_sh - r_sh, axis=1, keepdims=True)
    scale[scale < 1e-4] = 1.0

    out = arr.copy()
    for i in range(0, arr.shape[1], 2):
        out[:, i]     = (arr[:, i]     - center[:, 0]) / scale[:, 0]
        out[:, i + 1] = (arr[:, i + 1] - center[:, 1]) / scale[:, 0]
    return out


def _resample(arr: np.ndarray, target_len: int) -> np.ndarray:
    n = arr.shape[0]
    if n == target_len:
        return arr
    old_idx = np.linspace(0, 1, n)
    new_idx = np.linspace(0, 1, target_len)
    out = np.zeros((target_len, arr.shape[1]), dtype=np.float32)
    for col in range(arr.shape[1]):
        out[:, col] = np.interp(new_idx, old_idx, arr[:, col])
    return out


# ─── Dataset assembly ───────────────────────────────────────────────────

def build_dataset():
    train_csv = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    train_csv = train_csv[train_csv["sign"].isin(SELECTED_SIGNS)].reset_index(drop=True)

    if train_csv.empty:
        raise ValueError(
            "None of SELECTED_SIGNS were found in train.csv. "
            "Check spelling against sign_to_prediction_index_map.json."
        )

    label_names = sorted(train_csv["sign"].unique())
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    X, y = [], []
    skipped = 0
    for _, row in train_csv.iterrows():
        path = os.path.join(DATA_DIR, row["path"])
        seq = load_and_preprocess_sequence(path)
        if seq is None:
            skipped += 1
            continue
        X.append(seq)
        y.append(label_to_idx[row["sign"]])

    print(f"Loaded {len(X)} sequences, skipped {skipped} (insufficient hand data)")
    X = np.stack(X)
    y = np.array(y)
    return X, y, label_names


# ─── Model ───────────────────────────────────────────────────────────────

def build_model(n_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(SEQUENCE_LENGTH, N_FEATURES))
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    X, y, label_names = build_dataset()

    with open("label_names.json", "w") as f:
        json.dump(label_names, f)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_model(n_classes=len(label_names))
    model.summary()

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)],
    )

    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"Validation accuracy: {val_acc:.3f}")

    model.save("sign_model.h5")
    print("Saved sign_model.h5 — next: convert to TensorFlow.js (see roadmap step 3)")


if __name__ == "__main__":
    main()
