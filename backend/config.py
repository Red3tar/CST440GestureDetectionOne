"""Shared settings for the gesture detector backend.

Every gesture starts from an OPEN RIGHT HAND and transitions to one of five
end poses. IDLE_OPEN is the sixth class: the hand is held open (or moving
without finishing a gesture), so the model has something to say when the user
is NOT making one of the five gestures.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "gesture_data.csv"
MODEL_KERAS = ROOT / "models" / "gesture_model.keras"
MODEL_TFLITE = ROOT / "models" / "gesture_model_int8.tflite"
LABELS_JSON = ROOT / "models" / "labels.json"

GESTURES = ["IDLE_OPEN", "THUMBS_UP", "PENCIL", "ASL_U", "ASL_Y", "ASL_F"]

# One sample = SEQ_LEN frames sampled at SAMPLE_FPS (front end must use the
# same SAMPLE_FPS). 30 frames @ 15 fps = 2 seconds: start pose -> transition -> end pose.
SEQ_LEN = 30
SAMPLE_FPS = 15

NUM_LANDMARKS = 21          # MediaPipe Hands landmarks
COORDS = 3                  # x, y, z per landmark
NUM_FEATURES = NUM_LANDMARKS * COORDS   # 63 values per frame

# Landmark indices used for normalisation
WRIST = 0
MIDDLE_MCP = 9

# CSV header: sample_id,user_id,camera_res,frame,label,x0,y0,z0,...,x20,y20,z20
COORD_COLUMNS = [f"{a}{i}" for i in range(NUM_LANDMARKS) for a in "xyz"]
CSV_COLUMNS = ["sample_id", "user_id", "camera_res", "frame", "label"] + COORD_COLUMNS
