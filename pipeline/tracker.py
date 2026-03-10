import collections
import logging
from typing import Dict, List, Optional

import numpy as np

from pipeline.detector import BoundingBox

logger = logging.getLogger(__name__)

MAX_SEQ_LEN = 64


class Track:
    """
    Maintains per-person state: rolling frame buffer + best (largest) frame.
    """

    def __init__(self, track_id: int, max_frames: int = MAX_SEQ_LEN):
        self.track_id = track_id
        self.frames: collections.deque = collections.deque(maxlen=max_frames)
        self.bboxes: collections.deque = collections.deque(maxlen=max_frames)
        self.missed = 0
        self.best_frame: Optional[np.ndarray] = None
        self._best_area = 0

    @property
    def READY(self) -> bool:
        """True once we have at least 30 frames for identification."""
        return len(self.frames) >= 30

    def update(self, frame: np.ndarray, bbox: BoundingBox):
        self.frames.append(frame)
        self.bboxes.append(bbox)
        self.missed = 0

        area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)
        if area > self._best_area:
            self._best_area = area
            self.best_frame = frame.copy()

    def frame_bbox_pairs(self):
        """Yield (frame, bbox) aligned pairs for pose/silhouette extraction."""
        return list(zip(self.frames, self.bboxes))


class GaitTracker:
    """
    Maintains a dictionary of active Track objects.
    Purges tracks that have not been seen for max_age frames.
    """

    def __init__(self, max_age: int = 30):
        self.max_age = max_age
        self._tracks: Dict[str, Track] = {}

    def update(self, detections: List[BoundingBox], frame: np.ndarray) -> List[Track]:
        seen_ids = set()

        for det in detections:
            # If YOLO tracking didn't assign an ID, skip (shouldn't happen with persist=True)
            if det.track_id is None:
                continue

            tid = str(det.track_id)
            seen_ids.add(tid)

            if tid not in self._tracks:
                self._tracks[tid] = Track(det.track_id)
                logger.debug(f"New track: {tid}")

            self._tracks[tid].update(frame, det)

        # Age out unseen tracks
        stale = []
        for tid, t in self._tracks.items():
            if tid not in seen_ids:
                t.missed += 1
                if t.missed > self.max_age:
                    stale.append(tid)
        for tid in stale:
            logger.debug(f"Dropping stale track: {tid}")
            del self._tracks[tid]

        return list(self._tracks.values())
