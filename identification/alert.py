"""
alert.py

Builds the AlertPayload dict that is POSTed to the backend /api/alerts/ingest.
Snapshot is encoded as a JPEG; clip is encoded as an MP4 (via OpenCV VideoWriter).
Both are base64-encoded strings.
"""
import base64
import datetime
import logging
import os
import tempfile
from typing import List, Optional

import cv2
import numpy as np

from identification.identifier import IdentificationResult

logger = logging.getLogger(__name__)


class AlertBuilder:

    def build(self,
              camera_id: str,
              track_id:  str,
              result:    IdentificationResult,
              snapshot_frame:    Optional[np.ndarray],
              gait_clip_frames:  List[np.ndarray],
              zone: str = "normal",
              fps: int  = 15) -> dict:
        """
        Returns a dict matching the GaitEngineAlert schema expected by the backend.
        """
        snapshot_b64 = self.encode_frame_to_b64(snapshot_frame) if snapshot_frame is not None else ""
        clip_b64     = self.encode_clip_to_b64(gait_clip_frames, fps=fps) if gait_clip_frames else ""

        return {
            "camera_id":                  str(camera_id),
            "track_id":                   str(track_id),
            "confidence_score":           float(result.confidence_score),
            "nearest_match_person_id":    result.nearest_match_person_id,
            "nearest_match_similarity":   float(result.nearest_match_similarity or 0.0),
            "snapshot_b64":               snapshot_b64,
            "clip_b64":                   clip_b64,
            "frame_count":                len(gait_clip_frames),
            "zone":                       zone,
            "timestamp":                  datetime.datetime.utcnow().isoformat() + "Z",
        }

    # ------------------------------------------------------------------
    @staticmethod
    def encode_frame_to_b64(frame: np.ndarray) -> str:
        """JPEG-encode a frame and return a base64 string."""
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            logger.warning("Frame JPEG encoding failed.")
            return ""
        return base64.b64encode(buf).decode("utf-8")

    @staticmethod
    def encode_clip_to_b64(frames: List[np.ndarray], fps: int = 15) -> str:
        """
        Write frames to a temp MP4 file using OpenCV VideoWriter,
        read it back, base64-encode and return.
        """
        if not frames:
            return ""

        h, w = frames[0].shape[:2]
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(tmp_fd)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))

        for f in frames:
            if f.shape[:2] != (h, w):
                f = cv2.resize(f, (w, h))
            writer.write(f)
        writer.release()

        try:
            with open(tmp_path, "rb") as fh:
                encoded = base64.b64encode(fh.read()).decode("utf-8")
            return encoded
        except Exception as exc:
            logger.error(f"Clip base64 encoding failed: {exc}")
            return ""
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
