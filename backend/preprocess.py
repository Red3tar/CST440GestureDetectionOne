"""Preprocessing used by BOTH training and live inference (single source of truth).

The front end sends raw MediaPipe landmarks. All normalisation happens here so
the model never sees different preprocessing in training vs. deployment.
"""
import numpy as np
from config import SEQ_LEN, NUM_LANDMARKS, COORDS, WRIST, MIDDLE_MCP


def normalize_sequence(frames) -> np.ndarray:
    """frames: array-like (T, 21, 3) raw landmarks -> float32 (SEQ_LEN, 63).

    1. Translate every frame so the wrist is the origin (hand position in the image is ignored).
    2. Divide by the wrist->middle-knuckle distance (hand size / distance from camera is ignored).
    3. Pad (repeat last frame) or trim to exactly SEQ_LEN frames.
    """
    seq = np.asarray(frames, dtype=np.float32).reshape(-1, NUM_LANDMARKS, COORDS)
    if len(seq) == 0:
        raise ValueError("empty sequence")

    seq = seq - seq[:, WRIST:WRIST + 1, :]
    scale = np.linalg.norm(seq[:, MIDDLE_MCP, :2], axis=1)           # (T,)
    scale = np.clip(scale, 1e-6, None)[:, None, None]
    seq = seq / scale

    if len(seq) < SEQ_LEN:
        pad = np.repeat(seq[-1:], SEQ_LEN - len(seq), axis=0)
        seq = np.concatenate([seq, pad], axis=0)
    else:
        seq = seq[-SEQ_LEN:]

    return seq.reshape(SEQ_LEN, NUM_LANDMARKS * COORDS).astype(np.float32)
