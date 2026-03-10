"""
identifier.py

GaitIdentifier: converts dual-stream sequences into a verdict (KNOWN/UNKNOWN).
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class IdentificationResult:
    verdict: str                          # "KNOWN" | "UNKNOWN"
    person_id: Optional[str]
    display_name: Optional[str]
    confidence_score: float               # how confident this is UNKNOWN (1.0 = certain unknown)
    nearest_match_person_id: Optional[str]
    nearest_match_similarity: Optional[float]
    embedding: np.ndarray = field(repr=False)


class GaitIdentifier:
    """
    Computes the gait embedding and looks it up in the gallery.
    Returns an IdentificationResult for every call.
    """

    def __init__(self, gallery, embedder, config: dict):
        self.gallery   = gallery
        self.embedder  = embedder
        self.threshold = config.get("unknown_threshold", 0.50)

    def identify(self, heatmap_seq: np.ndarray, silhouette_seq: np.ndarray) -> IdentificationResult:
        embedding = self.embedder.embed(heatmap_seq, silhouette_seq)
        matches   = self.gallery.search(embedding, top_k=1)

        if not matches:
            # Empty gallery → always UNKNOWN with score 1.0
            return IdentificationResult(
                verdict="UNKNOWN", person_id=None, display_name=None,
                confidence_score=1.0,
                nearest_match_person_id=None, nearest_match_similarity=None,
                embedding=embedding,
            )

        best = matches[0]

        if best.score >= self.threshold:
            return IdentificationResult(
                verdict="KNOWN",
                person_id=best.person_id,
                display_name=best.name,
                confidence_score=float(best.score),  # similarity = confidence for KNOWN
                nearest_match_person_id=best.person_id,
                nearest_match_similarity=float(best.score),
                embedding=embedding,
            )
        else:
            # Normalise "unknown-ness" to [0, 1]
            unknown_conf = min(1.0, max(0.0, (self.threshold - best.score) / max(self.threshold, 1e-8)))
            return IdentificationResult(
                verdict="UNKNOWN",
                person_id=None,
                display_name=None,
                confidence_score=unknown_conf,
                nearest_match_person_id=best.person_id,
                nearest_match_similarity=float(best.score),
                embedding=embedding,
            )
