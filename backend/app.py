"""Backend API. Run:  python app.py   then open http://localhost:5000

GET  /api/config   -> labels + sequence settings for the front end
POST /api/sample   -> append one labelled landmark sequence to data/gesture_data.csv
POST /api/predict  -> classify the latest landmark window
"""
import csv
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from config import (DATA_CSV, MODEL_KERAS, GESTURES, SEQ_LEN, SAMPLE_FPS,
                    NUM_LANDMARKS, COORDS, CSV_COLUMNS)
from preprocess import normalize_sequence

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
app = Flask(__name__, static_folder=str(FRONTEND), static_url_path="")
_model = None


def get_model():
    global _model
    if _model is None and MODEL_KERAS.exists():
        import tensorflow as tf
        _model = tf.keras.models.load_model(MODEL_KERAS)
    return _model


def valid_frames(frames):
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (NUM_LANDMARKS, COORDS) or len(arr) < 5:
        raise ValueError(f"frames must be (T>=5, {NUM_LANDMARKS}, {COORDS})")
    return arr


@app.get("/")
def index():
    return send_from_directory(FRONTEND, "index.html")


@app.get("/api/config")
def config():
    return jsonify(labels=GESTURES, seq_len=SEQ_LEN, sample_fps=SAMPLE_FPS)


@app.post("/api/sample")
def save_sample():
    body = request.get_json(force=True)
    try:
        frames = valid_frames(body["frames"])
        label = body["label"]
        if label not in GESTURES:
            raise ValueError(f"unknown label {label}")
    except (KeyError, ValueError) as e:
        return jsonify(error=str(e)), 400

    sample_id = uuid.uuid4().hex[:10]
    DATA_CSV.parent.mkdir(exist_ok=True)
    new_file = not DATA_CSV.exists()
    with open(DATA_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(CSV_COLUMNS)
        for i, fr in enumerate(frames):
            w.writerow([sample_id, body.get("user_id", "anon"), body.get("camera_res", ""), i, label]
                       + [round(float(v), 5) for v in fr.reshape(-1)])
    return jsonify(saved=sample_id, frames=len(frames))


@app.post("/api/predict")
def predict():
    model = get_model()
    if model is None:
        return jsonify(error="No trained model yet. Run train_model.py first."), 503
    try:
        x = normalize_sequence(valid_frames(request.get_json(force=True)["frames"]))
    except (KeyError, ValueError) as e:
        return jsonify(error=str(e)), 400
    probs = model.predict(x[None, ...], verbose=0)[0]
    best = int(np.argmax(probs))
    return jsonify(label=GESTURES[best], confidence=float(probs[best]),
                   probabilities={g: float(p) for g, p in zip(GESTURES, probs)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
