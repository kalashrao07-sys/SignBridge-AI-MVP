"""
SignBridge AI — Sign Animation Keyframe Extractor (v2 — fixed selection logic)
================================================================================
FIX FROM v1: the previous version rejected any recording with >30% missing
hand landmarks and moved on, even though the very next line of code was
capable of repairing missing data via interpolation. It also stopped at
the FIRST recording that passed the threshold instead of comparing
candidates — so a mediocre early sample could win over a much cleaner
one later in the list. Real recordings in this dataset often have one
hand out of frame, which is normal and repairable, not a reason to
discard the sample outright.

NEW LOGIC:
  1. Score every candidate (or up to MAX_CANDIDATES, for speed) by its
     NaN ratio in the hand landmarks.
  2. Pick the lowest-NaN-ratio candidate.
  3. Interpolate to repair the remaining gaps.
  4. Only warn (don't discard) if even the best candidate is still poor
     quality — that's a real signal worth seeing, not silently hiding it.

OUTPUT: sign_animations.json — { "hello": [[[x,y], ...48 pts], ...frames], ... }
"""

import json
import os
import random

import numpy as np
import pandas as pd

DATA_DIR = "./asl_80"          # <-- your actual folder name
OUT_PATH = "sign_animations.json"
TARGET_FRAMES = 24
MAX_CANDIDATES = 20             # cap per sign for speed; raise if signs still fail
WARN_NAN_RATIO = 0.5            # flag (not reject) samples worse than this

POSE_KEEP = [11, 12, 13, 14, 15, 16]
N_HAND = 21

SELECTED_SIGNS = [
    "airplane", "alligator", "aunt", "awake", "balloon", "because", "bee", "bird",
    "blow", "brother", "brown", "bye", "cat", "closet", "cow", "cry", "doll",
    "donkey", "drink", "dry", "duck", "ear", "eye", "farm", "find", "fireman",
    "first", "flower", "food", "frog", "gift", "glasswindow", "goose", "gum",
    "hear", "hello", "home", "horse", "icecream", "kiss", "kitty", "lion", "lips",
    "listen", "look", "loud", "mad", "make", "man", "mom", "mouse", "mouth",
    "nap", "napkin", "nuts", "old", "orange", "owl", "pajamas", "pen", "pencil",
    "penny", "pizza", "pretend", "pretty", "sad", "shhh", "sleepy", "sun", "talk",
    "taste", "think", "tiger", "tooth", "toothbrush", "uncle", "up", "wake",
    "who", "yesterday",
]


def extract_one_clean_sample(sign: str, paths: list[str]) -> list | None:
    candidates = paths[:MAX_CANDIDATES]
    random.shuffle(candidates)  # avoid always favoring one participant's samples

    scored = []
    for path in candidates:
        full_path = os.path.join(DATA_DIR, path)
        try:
            df = pd.read_parquet(full_path)
        except Exception as e:
            print(f"    [{sign}] ⚠️ couldn't read {path}: {e}")
            continue

        frames = sorted(df["frame"].unique())
        if len(frames) < 4:
            print(f"    [{sign}] {path}: only {len(frames)} frames — skipping (too short)")
            continue

        per_frame = []
        for f in frames:
            fdf = df[df["frame"] == f]
            left = _points(fdf, "left_hand", range(N_HAND))
            right = _points(fdf, "right_hand", range(N_HAND))
            pose = _points(fdf, "pose", POSE_KEEP)
            per_frame.append(left + right + pose)

        arr = np.array(per_frame, dtype=np.float32)
        nan_ratio = float(np.isnan(arr[:, : N_HAND * 2 * 2]).mean())
        print(f"    [{sign}] {path}: nan_ratio={nan_ratio:.2f}, frames={len(frames)}")
        scored.append((nan_ratio, arr))

    if not scored:
        return None

    scored.sort(key=lambda t: t[0])
    best_ratio, best_arr = scored[0]

    if best_ratio > WARN_NAN_RATIO:
        print(f"    [{sign}] ⚠️ best candidate still has {best_ratio:.0%} missing hand data — "
              f"animation may look imprecise for this sign")

    repaired = _fill_and_resample(best_arr, TARGET_FRAMES)
    return repaired.reshape(TARGET_FRAMES, -1, 2).tolist()


def _points(fdf, type_name, indices):
    sub = fdf[fdf["type"] == type_name].set_index("landmark_index")
    out = []
    for i in indices:
        if i in sub.index:
            out.extend([float(sub.loc[i, "x"]), float(sub.loc[i, "y"])])
        else:
            out.extend([np.nan, np.nan])
    return out


def _fill_and_resample(arr, target_len):
    for col in range(arr.shape[1]):
        s = pd.Series(arr[:, col]).interpolate(limit_direction="both").fillna(0.5)
        arr[:, col] = s.to_numpy()
    old_idx = np.linspace(0, 1, arr.shape[0])
    new_idx = np.linspace(0, 1, target_len)
    out = np.zeros((target_len, arr.shape[1]), dtype=np.float32)
    for col in range(arr.shape[1]):
        out[:, col] = np.interp(new_idx, old_idx, arr[:, col])
    return out


def main():
    train_csv = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    animations = {}

    for sign in SELECTED_SIGNS:
        candidates = train_csv[train_csv["sign"] == sign]["path"].tolist()
        if not candidates:
            print(f"⚠️  '{sign}' not found in train.csv — skipping")
            continue

        print(f"Processing '{sign}' ({len(candidates)} candidates available)...")
        result = extract_one_clean_sample(sign, candidates)
        if result is None:
            print(f"❌ '{sign}': no usable sample found even after relaxed matching")
            continue

        animations[sign] = result
        print(f"✅ {sign}: extracted successfully\n")

    with open(OUT_PATH, "w") as f:
        json.dump(animations, f)
    print(f"\nSaved {len(animations)}/{len(SELECTED_SIGNS)} sign animations to {OUT_PATH}")


if __name__ == "__main__":
    main()