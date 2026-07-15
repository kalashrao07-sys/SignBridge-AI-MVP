"""
SignBridge AI — Sign Recognition Model Training (v2 — uses cache)
====================================================================
Run preprocess_signs.py ONCE first. This script just loads the cache
and trains — re-running to tweak epochs/architecture no longer
re-parses any parquet files.
"""

import json

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

CACHE_PATH = "sign_training_data.npz"
LABELS_PATH = "label_names.json"
EPOCHS = 60
BATCH_SIZE = 32


def build_model(sequence_length, n_features, n_classes):
    inputs = tf.keras.Input(shape=(sequence_length, n_features))
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    data = np.load(CACHE_PATH)
    X, y = data["X"], data["y"]
    with open(LABELS_PATH) as f:
        label_names = json.load(f)

    print(f"Loaded {len(X)} cached sequences, {len(label_names)} classes")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_model(X.shape[1], X.shape[2], len(label_names))
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
    print("Saved sign_model.h5 — next: tensorflowjs_converter --input_format=keras sign_model.h5 static/model/")


if __name__ == "__main__":
    main()