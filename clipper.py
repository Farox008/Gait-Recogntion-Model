"""
clipper.py — Video clipping utilities.

Saves buffered frames to a .mp4 file and extracts
a representative JPEG snapshot (base64-encoded) for alerts.
"""

import cv2
import os
import base64
import tempfile
import numpy as np
from datetime import datetime

CLIPS_DIR = os.path.join(os.path.dirname(__file__), "clips")
os.makedirs(CLIPS_DIR, exist_ok=True)


def save_clip(frames: list, fps: float = 10.0, prefix: str = "clip") -> str:
    """
    Write a list of BGR numpy frames to an mp4 file.
    Returns the absolute path to the saved clip.
    """
    if not frames:
        return None

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(CLIPS_DIR, f"{prefix}_{ts}.mp4")

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(path, fourcc, fps, (w, h))

    for frame in frames:
        out.write(frame)
    out.release()

    return path


def frame_to_b64_jpeg(frame: np.ndarray, quality: int = 80) -> str:
    """Encode a BGR frame as a base64 JPEG string (for API JSON responses)."""
    success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def extract_thumbnail(video_path: str, second: float = 0.5) -> str:
    """
    Open a video file, seek to `second`, and return a base64 JPEG thumbnail.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return ""
    return frame_to_b64_jpeg(frame)


def frames_from_video(video_path: str, max_frames: int = 60,
                       skip: int = 3) -> list:
    """
    Read frames from a video file.
    `skip`: take every N-th frame (reduces processing time).
    Returns list of BGR ndarrays.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % skip == 0:
            frames.append(frame)
        idx += 1
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames


def save_tempfile(upload_bytes: bytes, suffix: str = ".mp4") -> str:
    """Write upload bytes to a named temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(upload_bytes)
    return path
