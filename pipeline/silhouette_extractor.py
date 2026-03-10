import logging
from typing import List, Tuple

import cv2
import numpy as np

from pipeline.detector import BoundingBox

logger = logging.getLogger(__name__)

MASK_H, MASK_W = 128, 88
SEQ_LEN = 64


class SilhouetteExtractor:
    """
    Uses OpenCV MOG2 background subtractor to produce binary person masks.
    Masks are cropped to the person bounding box and resized to (128, 88).
    """

    def __init__(self, history: int = 500, var_threshold: float = 16.0):
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=True,
        )
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_background(self, frames: List[np.ndarray]):
        """Warm up the background model on initial frames (no suppression)."""
        for frame in frames:
            self.bg_sub.apply(frame)
        self.is_fitted = True
        logger.info(f"Background model fitted on {len(frames)} frames.")

    def extract(self, frame: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """
        Returns binary mask np.array uint8 shape (128, 88).
        Foreground pixels = 255, background = 0.
        """
        # Apply subtractor globally (with very slow learning so the bg stays stable)
        fg_mask = self.bg_sub.apply(frame, learningRate=0.001)

        # Shadow pixels are 127 in MOG2 — threshold them away
        _, binary = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Crop to person bbox
        h_f, w_f = binary.shape[:2]
        x1 = max(0, bbox.x1); y1 = max(0, bbox.y1)
        x2 = min(w_f, bbox.x2); y2 = min(h_f, bbox.y2)

        if x2 <= x1 or y2 <= y1:
            return np.zeros((MASK_H, MASK_W), dtype=np.uint8)

        crop = binary[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((MASK_H, MASK_W), dtype=np.uint8)

        resized = cv2.resize(crop, (MASK_W, MASK_H), interpolation=cv2.INTER_NEAREST)
        return resized

    def extract_sequence(self, frame_bbox_pairs: List[Tuple[np.ndarray, BoundingBox]]) -> np.ndarray:
        """
        Returns np.array shape (64, 128, 88) uint8.
        """
        masks = []
        for frame, bbox in frame_bbox_pairs:
            masks.append(self.extract(frame, bbox))

        blank = np.zeros((MASK_H, MASK_W), dtype=np.uint8)
        while len(masks) < SEQ_LEN:
            masks.append(blank)

        return np.stack(masks[:SEQ_LEN], axis=0)   # (64, 128, 88)
