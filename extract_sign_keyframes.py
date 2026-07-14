"""
SignBridge AI — Sign Animation Keyframe Extractor
====================================================
Reuses the SAME Kaggle landmark dataset used for training the
recognizer, but this time to build animation data for the
speech -> sign direction.

WHY THIS APPROACH:
  Generating a signing avatar from scratch (pose-generation model) is
  out of scope for a hackathon timeline. But we don't need to generate
  anything — the dataset already contains real people performing each
  sign correctly, as landmark coordinates. We just pick one clean
  sample per word and replay it as an animated skeleton in the
  browser. This is genuinely visually correct (it's real recorded
  sign motion, not a fabricated pose) without requiring any
  generative model.

OUTPUT:
  sign_animations.json — { "help": [[x,y,x,y,...], ...frames], ... }
  One file, small enough to ship as a static asset, loaded by the
  frontend animation renderer (see roadmap step 4).
"""

import json
import os

import numpy as np
import pandas as pd

DATA_DIR = "./data"
OUT_PATH = "sign_animations.json"
TARGET_FRAMES = 24  # smooth enough for playback, small file size

# Must match train_sign_model.py's landmark selection so recognition
# and animation are visually/semantically consistent.
POSE_KEEP = [11, 12, 13, 14, 15, 16]
N_HAND = 21

SELECTED_SIGNS = [
    "help", "water", "pain", "yes", "no", "hello", "please",
    "eat", "drink", "sick", "hurt", "more", "bathroom", "hot", "cold",
]


def extract_one_clean_sample(paths: list[str]) -> list | None:
    """Try each candidate sequence for this sign until one has usable
    hand data throughout, return it as a list of frames of [x,y] pairs
    for left hand (21), right hand (21), and pose (6) = 48 points/frame."""
    for path in paths:
        df = pd.read_parquet(os.path.join(DATA_DIR, path))
        frames = sorted(df["frame"].unique())
        if len(frames) < 4:
            continue

        per_frame = []
        valid = True
        for f in frames:
            fdf = df[df["frame"] == f]
            left = _points(fdf, "left_hand", range(N_HAND))
            right = _points(fdf, "right_hand", range(N_HAND))
            pose = _points(fdf, "pose", POSE_KEEP)
            per_frame.append(left + right + pose)

        arr = np.array(per_frame, dtype=np.float32)  # (frames, 96)
        nan_ratio = np.isnan(arr[:, : N_HAND * 2 * 2]).mean()
        if nan_ratio > 0.3:
            continue  # too much missing hand data, try next sample

        arr = _fill_and_resample(arr, TARGET_FRAMES)
        return arr.reshape(TARGET_FRAMES, -1, 2).tolist()  # [[ [x,y], ... 48 pts ], ...frames]

    return None


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
        result = extract_one_clean_sample(candidates)
        if result is None:
            print(f"⚠️  No clean sample found for '{sign}' — skipping")
            continue
        animations[sign] = result
        print(f"✅ {sign}: {len(candidates)} candidates available, using 1")

    with open(OUT_PATH, "w") as f:
        json.dump(animations, f)
    print(f"\nSaved {len(animations)} sign animations to {OUT_PATH}")


if __name__ == "__main__":
    main()
