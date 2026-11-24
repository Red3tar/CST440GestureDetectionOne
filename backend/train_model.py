"""Train the gesture model (Collect -> Preprocess -> Design -> Train -> Optimize).

Run:  python train_model.py
Needs data/gesture_data.csv recorded with the front end (see README).
TensorFlow uses the GPU automatically when one is available.
"""
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import (DATA_CSV, MODEL_KERAS, MODEL_TFLITE, LABELS_JSON, GESTURES,
                    SEQ_LEN, NUM_FEATURES, COORD_COLUMNS, CSV_COLUMNS)
from preprocess import normalize_sequence

# ---------------------------------------------------------------- 1. load + preprocess
df = pd.read_csv(DATA_CSV)
if "sample_id" not in df.columns:
    # Header row is missing (file was created empty before the first save): re-read with known column names
    df = pd.read_csv(DATA_CSV, header=None, names=CSV_COLUMNS)
    print("No header found in the CSV; using the expected column names.")
df = df[df["label"].isin(GESTURES)]          # drop any malformed rows
X, y = [], []
for _, g in df.groupby("sample_id"):
    g = g.sort_values("frame")
    coords = g[COORD_COLUMNS].to_numpy(dtype=np.float32)
    X.append(normalize_sequence(coords.reshape(-1, 21, 3)))
    y.append(GESTURES.index(g["label"].iloc[0]))
if not X:
    raise SystemExit("No valid samples found in the CSV. Record some in the app first.")
X, y = np.stack(X), np.array(y)
counts = np.bincount(y, minlength=len(GESTURES))
too_few = [GESTURES[i] for i, c in enumerate(counts) if c < 6]
if too_few:
    raise SystemExit(f"Need at least 6 samples per class (20+ recommended). Too few: {too_few}")
print("samples:", X.shape, "per class:", np.bincount(y, minlength=len(GESTURES)))

# ---------------------------------------------------------------- 2. train / val / test
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=1)


def augment(batch, label):
    """Small noise + scale jitter so the model tolerates different hands and speeds."""
    batch = batch * tf.random.uniform([], 0.9, 1.1)
    batch = batch + tf.random.normal(tf.shape(batch), stddev=0.01)
    return batch, label


train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(len(X_train)).batch(32).map(augment).prefetch(tf.data.AUTOTUNE))

# ---------------------------------------------------------------- 3. model (same pattern as the lecture slide)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(SEQ_LEN, NUM_FEATURES)),
    tf.keras.layers.Conv1D(32, kernel_size=3, activation="relu"),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(64, kernel_size=3, activation="relu"),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(GESTURES), activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ---------------------------------------------------------------- 4. train + evaluate
model.fit(train_ds, validation_data=(X_val, y_val), epochs=60,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"test accuracy: {acc:.3f}")

MODEL_KERAS.parent.mkdir(exist_ok=True)
model.save(MODEL_KERAS)
LABELS_JSON.write_text(json.dumps(GESTURES))

# ---------------------------------------------------------------- 5. TensorFlow Lite + INT8 quantization
def representative_data():
    for i in range(min(100, len(X_train))):
        yield [X_train[i:i + 1]]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data
MODEL_TFLITE.write_bytes(converter.convert())
print("saved", MODEL_KERAS.name, "and", MODEL_TFLITE.name)
