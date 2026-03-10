import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    track_id: Optional[int] = None


class PersonDetector:
    """
    YOLOv8n person detector.
    Runs inference every N frames and caches the last result for interim frames.
    """

    def __init__(self, model: str = "yolov8n.pt", conf: float = 0.4, device: str = "cuda",
                 detect_every: int = 3):
        logger.info(f"Loading detection model '{model}' on device '{device}' …")
        self.model = YOLO(model)
        self.conf_threshold = conf
        self.device = device
        self.detect_every = detect_every

        self._frame_count = 0
        self._cached: List[BoundingBox] = []

    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        """Return bounding boxes; uses cache every (detect_every - 1) frames."""
        self._frame_count += 1

        if self._frame_count % self.detect_every != 1 and self._cached:
            return self._cached

        results = self.model.track(
            frame,
            classes=[0],              # person only
            conf=self.conf_threshold,
            device=self.device,
            persist=True,             # keeps ByteTrack state inside YOLO
            verbose=False,
        )

        detections: List[BoundingBox] = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                conf = float(box.conf[0])
                tid = int(box.id[0]) if box.id is not None else None
                detections.append(BoundingBox(x1, y1, x2, y2, conf, tid))

        self._cached = detections
        return detections
