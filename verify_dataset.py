"""
Run this FIRST, before extract_sign_keyframes.py, to confirm the dataset
matches the expected structure. Takes a few seconds — cheap insurance
against a wasted extraction run.
"""

import os
import pandas as pd

DATA_DIR = "./data"  # <-- update if your folder is named/located differently

train_csv_path = os.path.join(DATA_DIR, "train.csv")
print(f"Looking for: {train_csv_path}")
train_csv = pd.read_csv(train_csv_path)

print("\n=== train.csv ===")
print("Columns:", list(train_csv.columns))
print("Rows:", len(train_csv))
print("Unique signs:", train_csv["sign"].nunique())
print("Sample signs:", sorted(train_csv["sign"].unique())[:15])
print("Sample path value:", train_csv["path"].iloc[0])

sample_path = os.path.join(DATA_DIR, train_csv["path"].iloc[0])
print(f"\nLooking for a sample parquet at: {sample_path}")
print("Exists:", os.path.exists(sample_path))

if os.path.exists(sample_path):
    df = pd.read_parquet(sample_path)
    print("\n=== sample parquet ===")
    print("Columns:", list(df.columns))
    print("Row count:", len(df))
    print("Unique 'type' values:", df["type"].unique() if "type" in df.columns else "NO 'type' COLUMN")
    print("Unique frames:", df["frame"].nunique() if "frame" in df.columns else "NO 'frame' COLUMN")
    print(df.head(3))
else:
    print("⚠️  Path didn't resolve — check whether train.csv's 'path' column already "
          "includes 'train_landmark_files/...' or needs it prepended.")
