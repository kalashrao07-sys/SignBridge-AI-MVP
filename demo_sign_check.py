import pandas as pd

df = pd.read_csv("asl_80/train.csv")

print("Columns:")
print(df.columns)

print("\nUnique signs:", df["sign"].nunique())

print("\nFirst 100 unique signs:")
for s in sorted(df["sign"].unique())[:100]:
    print(s)