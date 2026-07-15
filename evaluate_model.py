"""
SignBridge AI — Per-Class Model Evaluation
=============================================
Aggregate accuracy hides the thing that actually matters for a live
demo: WHICH signs are reliable and which aren't. Run this after
training to get a per-class breakdown, then only wire the reliable
ones into your "live recognized" vocabulary.
"""

import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

CACHE_PATH = "sign_training_data.npz"
LABELS_PATH = "label_names.json"
MODEL_PATH = "sign_model.h5"

RELIABLE_THRESHOLD = 0.6  # per-class recall above this = safe to demo live


def main():
    data = np.load(CACHE_PATH)
    X, y = data["X"], data["y"]
    with open(LABELS_PATH) as f:
        label_names = json.load(f)

    # Same split as training (same random_state) so this evaluates on
    # the exact held-out set the model never trained on.
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = tf.keras.models.load_model(MODEL_PATH)
    y_pred = np.argmax(model.predict(X_val), axis=1)

    report = classification_report(
        y_val, y_pred, target_names=label_names, output_dict=True, zero_division=0
    )

    reliable, unreliable = [], []
    for name in label_names:
        recall = report[name]["recall"]
        support = report[name]["support"]
        (reliable if recall >= RELIABLE_THRESHOLD else unreliable).append((name, recall, int(support)))

    reliable.sort(key=lambda t: -t[1])
    unreliable.sort(key=lambda t: -t[1])

    print(f"\n✅ RELIABLE ({len(reliable)} signs, recall >= {RELIABLE_THRESHOLD:.0%}) — safe for live demo:")
    for name, recall, support in reliable:
        print(f"   {name:15s} recall={recall:.2f}  (n={support} val samples)")

    print(f"\n⚠️  UNRELIABLE ({len(unreliable)} signs, recall < {RELIABLE_THRESHOLD:.0%}) — library only, not live:")
    for name, recall, support in unreliable:
        print(f"   {name:15s} recall={recall:.2f}  (n={support} val samples)")

    # Most-confused pairs — useful if two signs keep swapping with each
    # other, since that's often fixable by dropping one from the live
    # set rather than a model problem.
    cm = confusion_matrix(y_val, y_pred)
    print("\n🔀 Most confused pairs (true → predicted, count):")
    confusions = []
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            if i != j and cm[i][j] > 0:
                confusions.append((cm[i][j], label_names[i], label_names[j]))
    confusions.sort(reverse=True)
    for count, true_name, pred_name in confusions[:15]:
        print(f"   {true_name:15s} → {pred_name:15s}  ({count}x)")

    with open("reliable_signs.json", "w") as f:
        json.dump([n for n, _, _ in reliable], f)
    print(f"\nSaved {len(reliable)} reliable sign names to reliable_signs.json")


if __name__ == "__main__":
    main()
