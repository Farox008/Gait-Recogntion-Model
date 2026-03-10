import cv2
import numpy as np


class GaitDenoiser:
    """
    Morphological denoiser for binary silhouette masks.
    Applies open (remove noise) then close (fill holes).
    Falls back to the raw mask if denoising erodes too much foreground.
    """

    MIN_DENSITY_RATIO = 0.20   # if cleaned mask < 20 % of raw, use raw instead

    def __init__(self):
        self._kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def denoise_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Cleans a single (H, W) uint8 binary mask.
        Returns cleaned mask or the original on quality fallback.
        """
        raw_count = np.count_nonzero(mask)
        if raw_count == 0:
            return mask

        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._kernel_open)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self._kernel_close)

        clean_count = np.count_nonzero(closed)
        if clean_count < raw_count * self.MIN_DENSITY_RATIO:
            return mask   # denoising destroyed too much — use raw

        return closed

    def denoise_sequence(self, silhouette_seq: np.ndarray) -> np.ndarray:
        """
        Applies denoise_mask to every frame in (T, H, W) sequence.
        Returns cleaned array of the same shape and dtype.
        """
        cleaned = np.stack([self.denoise_mask(silhouette_seq[t])
                            for t in range(len(silhouette_seq))], axis=0)
        return cleaned
