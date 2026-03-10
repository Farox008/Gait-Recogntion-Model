"""
fusion_module.py

Cross-attention fusion of SkeletonGaitPP and DeepGaitV2 embeddings.
Both encoders output 256-d embeddings → fused to a single 256-d vector.

GaitEmbedder handles:
  - calling both encoders
  - cross-attention fusion
  - graceful fallback when one stream fails
"""
import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

EMBED_DIM = 256   # Both real encoders output 256-d


class GaitFusionModule(nn.Module):
    """
    Cross-attention: skeleton queries attend to silhouette.
    Input:  skel_feat, sil_feat — each (B, D)
    Output: fused               — (B, D)
    """
    def __init__(self, embed_dim: int = EMBED_DIM, num_heads: int = 8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.fc         = nn.Linear(embed_dim * 2, embed_dim)
        self.norm       = nn.LayerNorm(embed_dim)

    def forward(self, skel_feat: torch.Tensor, sil_feat: torch.Tensor) -> torch.Tensor:
        q = skel_feat.unsqueeze(1)   # (B, 1, D)
        k = sil_feat.unsqueeze(1)
        attended, _ = self.cross_attn(q, k, k)                        # (B, 1, D)
        fused = torch.cat([q, attended], dim=-1)                       # (B, 1, 2D)
        return self.norm(self.fc(fused)).squeeze(1)                    # (B, D)


class GaitEmbedder:
    """
    Orchestrates SkeletonEncoder + SilhouetteEncoder + GaitFusionModule.

    Important API change vs random-encoder version:
      SkeletonEncoder.encode() now takes (heatmap_seq, silhouette_seq) since
      SkeletonGaitPP uses BOTH streams internally.
      SilhouetteEncoder.encode() takes only silhouette_seq.
    """

    def __init__(self, skeleton_encoder, silhouette_encoder,
                 fusion_module: GaitFusionModule,
                 device: str = "cuda",
                 fusion_weights_path: str = ""):
        self.skel_enc = skeleton_encoder
        self.sil_enc  = silhouette_encoder
        self.device   = torch.device(device if torch.cuda.is_available() else "cpu")

        self.fusion = fusion_module.to(self.device)
        if fusion_weights_path and os.path.isfile(fusion_weights_path):
            self.fusion.load_state_dict(
                torch.load(fusion_weights_path, map_location=self.device, weights_only=False),
                strict=False
            )
            logger.info(f"Fusion weights loaded from {fusion_weights_path}")
        self.fusion.eval()

    @staticmethod
    def _l2(arr):
        n = np.linalg.norm(arr)
        return arr / n if n > 1e-8 else arr

    def embed(self, heatmap_seq: np.ndarray, silhouette_seq: np.ndarray) -> np.ndarray:
        """
        heatmap_seq   : (T, H, W) float32  — pose heatmaps
        silhouette_seq: (T, H, W) uint8    — binary masks

        Returns 256-d unit-norm numpy embedding.
        """
        skel_empty = np.sum(np.abs(heatmap_seq)) < 1e-6
        sil_empty  = np.count_nonzero(silhouette_seq) == 0

        if skel_empty and sil_empty:
            logger.warning("Both streams empty — returning zero embedding.")
            return np.zeros(EMBED_DIM, dtype=np.float32)

        # SkeletonGaitPP uses both streams internally
        try:
            skel_emb_np = self.skel_enc.encode(heatmap_seq, silhouette_seq)
        except Exception as exc:
            logger.warning(f"SkeletonEncoder failed: {exc}")
            skel_emb_np = None

        # DeepGaitV2 uses silhouette only
        try:
            sil_emb_np = self.sil_enc.encode(silhouette_seq)
        except Exception as exc:
            logger.warning(f"SilhouetteEncoder failed: {exc}")
            sil_emb_np = None

        if skel_emb_np is None and sil_emb_np is None:
            return np.zeros(EMBED_DIM, dtype=np.float32)
        if skel_emb_np is None:
            return self._l2(sil_emb_np)
        if sil_emb_np is None:
            return self._l2(skel_emb_np)

        # Fuse both
        skel_t = torch.tensor(skel_emb_np[None], dtype=torch.float32).to(self.device)
        sil_t  = torch.tensor(sil_emb_np[None],  dtype=torch.float32).to(self.device)
        with torch.no_grad():
            fused = self.fusion(skel_t, sil_t)
        return self._l2(fused.squeeze(0).cpu().numpy())
