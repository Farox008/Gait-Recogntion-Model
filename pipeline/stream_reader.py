import cv2
import time
import logging
import threading
from queue import Queue, Full

logger = logging.getLogger(__name__)


class StreamReader:
    """
    Reads frames from an RTSP stream or video file.
    Runs in a background thread and pushes (camera_id, frame, timestamp) tuples
    into the provided queue.  Automatically reconnects on dropout.
    """

    def __init__(self, rtsp_url: str, camera_id: str, frame_queue: Queue, maxsize: int = 128):
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self.queue = frame_queue
        self.running = False
        self._thread: threading.Thread | None = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._watchdog, daemon=True, name=f"stream-{self.camera_id}")
        self._thread.start()
        logger.info(f"StreamReader started for camera {self.camera_id} → {self.rtsp_url}")

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info(f"StreamReader stopped for camera {self.camera_id}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _watchdog(self):
        """Top-level loop: keeps reopening the stream on any failure."""
        while self.running:
            try:
                self._read_stream()
            except Exception as exc:
                logger.error(f"[{self.camera_id}] Stream error: {exc}")
            if self.running:
                logger.info(f"[{self.camera_id}] Reconnecting in 5 s …")
                time.sleep(5)

    def _read_stream(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open stream: {self.rtsp_url}")

        logger.info(f"[{self.camera_id}] Stream opened successfully.")
        consecutive_failures = 0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures > 10:
                    logger.warning(f"[{self.camera_id}] Consecutive read failures – treating as dropout.")
                    break
                time.sleep(0.05)
                continue

            consecutive_failures = 0
            ts = time.time()

            try:
                self.queue.put_nowait((self.camera_id, frame, ts))
            except Full:
                # Drop the oldest frame to keep up with real-time
                try:
                    self.queue.get_nowait()
                except Exception:
                    pass
                self.queue.put_nowait((self.camera_id, frame, ts))

        cap.release()
