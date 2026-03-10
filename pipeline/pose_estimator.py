import logging
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from pipeline.detector import BoundingBox

logger = logging.getLogger(__name__)

HEATMAP_H, HEATMAP_W = 128, 88
SEQ_LEN = 64
NUM_KEYPOINTS = 17
SIGMA = 2   # Gaussian spread in pixels for heatmap rendering


def _gaussian_heatmap(h: int, w: int, cx: int, cy: int, sigma: int = SIGMA) -> np.ndarray:
    """Render a single 2-D Gaussian blob centred at (cx, cy)."""
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    hm = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return hm.astype(np.float32)


class PoseEstimator:
    """
    Wraps YOLOv8-Pose to produce:
      - 17-keypoint arrays from a person crop
      - (128, 88) heatmap images (summed over joints)
      - (64, 128, 88) heatmap sequences per track
    """

    def __init__(self, model: str = "yolov8n-pose.pt", device: str = "cuda"):
        logger.info(f"Loading pose model '{model}' on '{device}' …")
        self.model = YOLO(model)
        self.device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, frame: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """
        Returns np.array shape (17, 2) of (x, y) keypoints in [0, 1] range
        relative to the crop.  Zero-filled on failure.
        """
        h_f, w_f = frame.shape[:2]
        x1 = max(0, bbox.x1); y1 = max(0, bbox.y1)
        x2 = min(w_f, bbox.x2); y2 = min(h_f, bbox.y2)

        if x2 <= x1 or y2 <= y1:
            return np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)

        results = self.model(crop, device=self.device, verbose=False)
        if (results and results[0].keypoints is not None
                and len(results[0].keypoints.xy) > 0):
            kp = results[0].keypoints.xy[0].cpu().numpy()  # (17, 2) pixel coords in crop
            if kp.shape[0] == NUM_KEYPOINTS:
                ch, cw = crop.shape[:2]
                kp[:, 0] /= max(cw, 1)   # normalise to [0, 1]
                kp[:, 1] /= max(ch, 1)
                return kp.astype(np.float32)

        return np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)

    def build_heatmap(self, keypoints: np.ndarray,
                      size: Tuple[int, int] = (HEATMAP_H, HEATMAP_W)) -> np.ndarray:
        """
        Renders keypoints onto a (H, W) float32 heatmap by summing Gaussians.
        Returns values in [0, 1].
        """
        h, w = size
        hm = np.zeros((h, w), dtype=np.float32)
        for (kx, ky) in keypoints:
            if kx <= 0 and ky <= 0:  # invisible / missing keypoint
                continue
            cx = int(kx * w)
            cy = int(ky * h)
            if 0 <= cx < w and 0 <= cy < h:
                hm += _gaussian_heatmap(h, w, cx, cy)
        return np.clip(hm, 0.0, 1.0)

    def build_sequence(self, frame_bbox_pairs: List[Tuple[np.ndarray, BoundingBox]]) -> np.ndarray:
        """
        Given a list of (frame, bbox) pairs, returns np.array shape (64, 128, 88).
        Pads with zeros if fewer than 64 pairs.
        """
        heatmaps = []
        for frame, bbox in frame_bbox_pairs:
            kps = self.estimate(frame, bbox)
            hm = self.build_heatmap(kps)
            heatmaps.append(hm)

        # Pad to SEQ_LEN
        blank = np.zeros((HEATMAP_H, HEATMAP_W), dtype=np.float32)
        while len(heatmaps) < SEQ_LEN:
            heatmaps.append(blank)

        return np.stack(heatmaps[:SEQ_LEN], axis=0)   # (64, 128, 88)
