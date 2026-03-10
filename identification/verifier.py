"""
verifier.py

Prevents alert spam by requiring N consecutive UNKNOWN verdicts
for the same track before firing an alert, plus a per-track cooldown.
"""
import logging
import time
from typing import Dict

from identification.identifier import IdentificationResult

logger = logging.getLogger(__name__)


class UnknownVerifier:
    """
    State machine per track_id:
      - UNKNOWN increments a counter
      - KNOWN resets the counter
      - Counter reaching required_consecutive triggers an alert + starts cooldown
      - During cooldown no further alerts fire for that track
    """

    def __init__(self, required_consecutive: int = 3, cooldown_seconds: float = 300.0):
        self.required    = required_consecutive
        self.cooldown    = cooldown_seconds
        self._counts:   Dict[str, int]   = {}
        self._cooldowns: Dict[str, float] = {}   # track_id → alert timestamp

    def check(self, track_id: str, result: IdentificationResult) -> bool:
        """
        Returns True iff an alert should be fired for this track right now.
        """
        tid = str(track_id)
        now = time.time()

        if result.verdict == "KNOWN":
            self._counts[tid] = 0
            return False

        # UNKNOWN branch
        # Active cooldown?
        if tid in self._cooldowns:
            if now - self._cooldowns[tid] < self.cooldown:
                return False          # still cooling down
            else:
                del self._cooldowns[tid]   # cooldown expired

        self._counts[tid] = self._counts.get(tid, 0) + 1

        if self._counts[tid] >= self.required:
            logger.warning(
                f"Track {tid}: {self._counts[tid]} consecutive UNKNOWNs — firing alert."
            )
            self._cooldowns[tid] = now
            self._counts[tid]    = 0
            return True

        return False
