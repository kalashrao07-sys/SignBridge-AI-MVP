"""
SignBridge AI — Preprocessing (separated from training, cached)
==================================================================
WHY THIS EXISTS:
  train_sign_model.py re-parsed every parquet file on every run, and
  did it inefficiently — filtering the full dataframe once per frame
  (roughly O(frames^2) per sequence). That's the real reason training
  felt slow, not just "no caching." This script:

    1. Extracts landmarks using a vectorized pivot (one pass per file,
       not one filter per frame) — much faster per sequence.
    2. Runs ONCE and caches the result to sign_training_data.npz.
       train_sign_model.py then just loads the cache — re-running
       training (to tweak epochs, architecture, etc.) becomes instant
       instead of re-doing extraction every time.

Run this once per vocabulary choice. Re-run only if you change
SELECTED_SIGNS or MAX_SAMPLES_PER_SIGN.
"""

import os
import time

import numpy as np
import pandas as pd

DATA_DIR = "./asl_80"
CACHE_PATH = "sign_training_data.npz"
LABELS_PATH = "label_names.json"

SEQUENCE_LENGTH = 32
MAX_SAMPLES_PER_SIGN = 60   # cap per class — plenty for a small model, keeps this fast

POSE_KEEP = [11, 12, 13, 14, 15, 16]
N_HAND = 21
N_POSE_KEEP = len(POSE_KEEP)

# Same 80-word vocabulary as extract_sign_keyframes_v2.py, so
# recognition and animation share one vocabulary.
SELECTED_SIGNS = [
    "airplane", "alligator", "aunt", "awake", "balloon", "because", "bee",
    "bird", "blow", "brother", "brown", "bye", "cat", "closet", "cow",
    "cry", "doll", "donkey", "drink", "dry", "duck", "ear", "eye", "farm",
    "find", "fireman", "first", "flower", "food", "frog", "gift",
    "glasswindow", "goose", "gum", "hear", "hello", "home", "horse",
    "icecream", "kiss", "kitty", "lion", "lips", "listen", "look", "loud",
    "mad", "make", "man", "mom", "mouse", "mouth", "nap", "napkin", "nuts",
    "old", "orange", "owl", "pajamas", "pen", "pencil", "penny", "pizza",
    "pretend", "pretty", "sad", "shhh", "sleepy", "sun", "talk", "taste",
    "think", "tiger", "tooth", "toothbrush", "uncle", "up", "wake", "who",
    "yesterday",
]


def extract_sequence_fast(parquet_path: str):
    """Vectorized extraction: one pivot per file instead of one filter
    per frame. Returns (SEQUENCE_LENGTH, N_FEATURES) or None."""
    df = pd.read_parquet(parquet_path, columns=["frame", "type", "landmark_index", "x", "y"])

    df = df[df["type"].isin(["left_hand", "right_hand", "pose"])].copy()
    df = df[
        (df["type"] != "pose") | (df["landmark_index"].isin(POSE_KEEP))
    ]

    if df.empty:
        return None

    df["col"] = df["type"] + "_" + df["landmark_index"].astype(str)

    pivot_x = df.pivot_table(index="frame", columns="col", values="x", aggfunc="first")
    pivot_y = df.pivot_table(index="frame", columns="col", values="y", aggfunc="first")

    if len(pivot_x) < 4:
        return None

    col_order = (
        [f"left_hand_{i}" for i in range(N_HAND)]
        + [f"right_hand_{i}" for i in range(N_HAND)]
        + [f"pose_{i}" for i in POSE_KEEP]
    )

    # Reindex guarantees every expected column exists (NaN if missing
    # in this particular file), and fixes column order for every sample.
    px = pivot_x.reindex(columns=[c for c in col_order]).to_numpy()
    py = pivot_y.reindex(columns=[c for c in col_order]).to_numpy()

    # Interleave x,y -> (frames, N_FEATURES)
    n_frames = px.shape[0]
    arr = np.empty((n_frames, px.shape[1] * 2), dtype=np.float32)
    arr[:, 0::2] = px
    arr[:, 1::2] = py

    arr = _interpolate_nans(arr)
    arr = _normalize(arr)
    arr = _resample(arr, SEQUENCE_LENGTH)
    return arr


def _interpolate_nans(arr):
    for col in range(arr.shape[1]):
        s = pd.Series(arr[:, col]).interpolate(limit_direction="both")
        arr[:, col] = s.fillna(0.5).to_numpy()
    return arr


def _normalize(arr):
    pose_start = N_HAND * 2 * 2
    l_sh = arr[:, pose_start: pose_start + 2]
    r_sh = arr[:, pose_start + 2: pose_start + 4]
    center = (l_sh + r_sh) / 2.0
    scale = np.linalg.norm(l_sh - r_sh, axis=1, keepdims=True)
    scale[scale < 1e-4] = 1.0

    out = arr.copy()
    for i in range(0, arr.shape[1], 2):
        out[:, i]     = (arr[:, i]     - center[:, 0]) / scale[:, 0]
        out[:, i + 1] = (arr[:, i + 1] - center[:, 1]) / scale[:, 0]
    return out


def _resample(arr, target_len):
    n = arr.shape[0]
    if n == target_len:
        return arr
    old_idx = np.linspace(0, 1, n)
    new_idx = np.linspace(0, 1, target_len)
    out = np.zeros((target_len, arr.shape[1]), dtype=np.float32)
    for col in range(arr.shape[1]):
        out[:, col] = np.interp(new_idx, old_idx, arr[:, col])
    return out


def main():
    t0 = time.time()
    train_csv = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    train_csv = train_csv[train_csv["sign"].isin(SELECTED_SIGNS)]

    X, y = [], []
    label_names = sorted(train_csv["sign"].unique())
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    for sign in label_names:
        rows = train_csv[train_csv["sign"] == sign].head(MAX_SAMPLES_PER_SIGN)
        count = 0
        for _, row in rows.iterrows():
            seq = extract_sequence_fast(os.path.join(DATA_DIR, row["path"]))
            if seq is None:
                continue
            X.append(seq)
            y.append(label_to_idx[sign])
            count += 1
        print(f"{sign}: {count} sequences")

    X = np.stack(X)
    y = np.array(y)

    np.savez_compressed(CACHE_PATH, X=X, y=y)
    import json
    with open(LABELS_PATH, "w") as f:
        json.dump(label_names, f)

    elapsed = time.time() - t0
    print(f"\nCached {len(X)} sequences across {len(label_names)} signs to {CACHE_PATH}")
    print(f"Took {elapsed:.0f}s. train_sign_model.py will now load this instantly.")


if __name__ == "__main__":
    main()
