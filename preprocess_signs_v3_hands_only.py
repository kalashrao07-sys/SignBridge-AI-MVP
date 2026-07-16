"""
SignBridge AI — Preprocessing v3 (hands-only, matches browser exactly)
=========================================================================
CHANGED FROM v2: dropped pose/shoulder landmarks entirely — the browser
only runs MediaPipe Hands, not Pose/Holistic, so a model trained on
pose features can't be fed real data at inference time. This version
uses only the 2x21 hand landmarks, which is exactly what's available
client-side.

NORMALIZATION FIX: centering each frame on that frame's own wrist would
erase hand-trajectory motion (the very thing a sequence model needs to
tell "awake" from "wake"). Instead we compute ONE center/scale per
SEQUENCE (averaged across its frames) and apply it uniformly — this
still normalizes for different starting positions/distances-from-camera
across recordings, but preserves motion within each sign.
"""

import os
import time
import json

import numpy as np
import pandas as pd

DATA_DIR = "./asl_80"
CACHE_PATH = "sign_training_data.npz"
LABELS_PATH = "label_names.json"

SEQUENCE_LENGTH = 32
MAX_SAMPLES_PER_SIGN = 500

N_HAND = 21
N_FEATURES = N_HAND * 2 * 2  # left+right hands, x+y each = 84

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
    df = pd.read_parquet(parquet_path, columns=["frame", "type", "landmark_index", "x", "y"])
    df = df[df["type"].isin(["left_hand", "right_hand"])].copy()
    if df.empty:
        return None

    df["col"] = df["type"] + "_" + df["landmark_index"].astype(str)
    pivot_x = df.pivot_table(index="frame", columns="col", values="x", aggfunc="first")
    pivot_y = df.pivot_table(index="frame", columns="col", values="y", aggfunc="first")

    if len(pivot_x) < 4:
        return None

    col_order = [f"left_hand_{i}" for i in range(N_HAND)] + [f"right_hand_{i}" for i in range(N_HAND)]
    px = pivot_x.reindex(columns=col_order).to_numpy()
    py = pivot_y.reindex(columns=col_order).to_numpy()

    n_frames = px.shape[0]
    raw = np.empty((n_frames, N_HAND * 2, 2), dtype=np.float32)  # (frames, 42 pts, xy)
    raw[:, :, 0] = px
    raw[:, :, 1] = py

    # --- sequence-level center/scale (computed BEFORE interpolation,
    #     so NaN correctly signals "hand not present this frame") ---
    left_wrist  = raw[:, 0, :]    # landmark 0 of left hand
    right_wrist = raw[:, 21, :]   # landmark 0 of right hand
    left_mcp    = raw[:, 9, :]    # landmark 9 (middle MCP) of left hand
    right_mcp   = raw[:, 30, :]   # landmark 9 of right hand (offset by 21)

    left_scale  = np.linalg.norm(left_wrist - left_mcp, axis=1)
    right_scale = np.linalg.norm(right_wrist - right_mcp, axis=1)

    center = np.nanmean(np.stack([left_wrist, right_wrist]), axis=(0, 1))  # (2,)
    scale  = np.nanmean(np.concatenate([left_scale, right_scale]))          # scalar

    if np.isnan(center).any():
        center = np.array([0.5, 0.5], dtype=np.float32)
    if np.isnan(scale) or scale < 1e-4:
        scale = 1.0

    # --- fill remaining per-landmark gaps via interpolation over time ---
    flat = raw.reshape(n_frames, -1)
    flat = _interpolate_nans(flat)

    # --- apply the SAME center/scale to every frame ---
    flat = flat.reshape(n_frames, N_HAND * 2, 2)
    flat = (flat - center) / scale
    flat = flat.reshape(n_frames, -1)

    flat = _resample(flat, SEQUENCE_LENGTH)
    return flat


def _interpolate_nans(arr):
    for col in range(arr.shape[1]):
        s = pd.Series(arr[:, col]).interpolate(limit_direction="both")
        arr[:, col] = s.fillna(0.0).to_numpy()
    return arr


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
        available = train_csv[train_csv["sign"] == sign]
        rows = available.head(MAX_SAMPLES_PER_SIGN)
        count = 0
        for _, row in rows.iterrows():
            seq = extract_sequence_fast(os.path.join(DATA_DIR, row["path"]))
            if seq is None:
                continue
            X.append(seq)
            y.append(label_to_idx[sign])
            count += 1
        print(f"{sign}: {count} used / {len(available)} available in dataset")

    X = np.stack(X)
    y = np.array(y)

    np.savez_compressed(CACHE_PATH, X=X, y=y)
    with open(LABELS_PATH, "w") as f:
        json.dump(label_names, f)

    elapsed = time.time() - t0
    print(f"\nCached {len(X)} sequences ({X.shape[1]} frames x {X.shape[2]} features) "
          f"across {len(label_names)} signs to {CACHE_PATH}")
    print(f"Took {elapsed:.0f}s.")


if __name__ == "__main__":
    main()
