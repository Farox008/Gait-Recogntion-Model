"""
skeleton_encoder.py

Exact PyTorch reimplementation of SkeletonGaitPP whose module names and
parameter shapes match the OpenGait checkpoint at:
  Gait3D/SkeletonGaitPP/SkeletonGaitPP/checkpoints/SkeletonGaitPP-60000.pt

Architecture (from full key inspection):
  sil_layer0   : ConvBn(1→64, 3×3)
  map_layer0   : ConvBn(2→64, 3×3)
  sil_layer1   : ForwardBlock wrapping ResBlock2D(64→64)
  map_layer1   : ForwardBlock wrapping ResBlock2D(64→64)
  fusion       : channel-attention ConvBn sequence  128→4→4→128
  layer2       : 4 × GaitResBlock3D(64→128, stride=2)
  layer3       : 4 × GaitResBlock3D(128→256)
  layer4       : 1 × GaitResBlock3D(256→512)
  FCs.fc_bin   : [16, 512, 256]  — part-based FC
  BNNecks      : [16, 256, 3000] — skipped at inference

Inference output: 256-d unit-norm embedding obtained by
  1. Temporal mean-pooling after layer4
  2. Horizontal part pooling into 16 parts
  3. FCs.fc_bin on each part  → (B, 16, 256)
  4. mean over parts           → (B, 256)
  5. L2 normalise
"""
import logging, os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ── OpenGait building blocks ────────────────────────────────────────────────

class ConvBn(nn.Sequential):
    """forward_block = [Conv2d, BN]  (no activation — OpenGait keeps it separate)"""
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
        )

class _ForwardBlock_ConvBn(nn.Module):
    """Wraps ConvBn as  .forward_block  attribute (matches OpenGait naming)."""
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.forward_block = ConvBn(in_c, out_c, k, s, p)
    def forward(self, x):
        return F.relu(self.forward_block(x), inplace=True)


class _ResBlock2D(nn.Module):
    """Plain 2-D residual block used in sil_layer1 / map_layer1."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + x, inplace=True)

class _ForwardBlock_ResBlock2D(nn.Module):
    """Wraps ResBlock2D as  .forward_block.0  attribute."""
    def __init__(self, channels):
        super().__init__()
        self.forward_block = nn.Sequential(_ResBlock2D(channels))
    def forward(self, x):
        return self.forward_block(x)


class _FusionConv(nn.Module):
    """
    fusion.conv  with keys:
      forward_block.0  ConvBn(128→4, 1×1)
      forward_block.3  ConvBn( 4→4, 3×3, p=1)
      forward_block.6  Conv2d( 4→128, 1×1)

    The 128-d gate is split into (gate_sil, gate_map) each 64-d.
    Output = gate_sil * sil + gate_map * map  → 64 channels.
    This allows layer2 to receive 64-d input.
    """
    def __init__(self):
        super().__init__()
        self.forward_block = nn.Sequential(
            nn.Conv2d(128, 4, 1, bias=False),    # index 0
            nn.BatchNorm2d(4),                    # index 1
            nn.ReLU(inplace=True),                # index 2
            nn.Conv2d(4, 4, 3, 1, 1, bias=False),# index 3
            nn.BatchNorm2d(4),                    # index 4
            nn.ReLU(inplace=True),                # index 5
            nn.Conv2d(4, 128, 1, bias=False),     # index 6
        )
    def forward(self, sil_feat, map_feat):
        combined = torch.cat([sil_feat, map_feat], dim=1)  # (B*T, 128, H, W)
        gate = torch.sigmoid(self.forward_block(combined))  # (B*T, 128, H, W)
        gate_sil = gate[:, :64, :, :]
        gate_map = gate[:, 64:, :, :]
        return gate_sil * sil_feat + gate_map * map_feat    # (B*T, 64, H, W)


class _Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = _FusionConv()
    def forward(self, sil_feat, map_feat):
        return self.conv(sil_feat, map_feat)   # (B*T, 64, H, W)


# ── 3-D residual block (temporal + spatial) ─────────────────────────────────

class _ConvBn2D_fw(nn.Module):
    """Conv2d+BN with .forward_block  (per-frame spatial conv in backbone)."""
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.forward_block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
        )
    def forward_2d(self, x):
        return F.relu(self.forward_block(x), inplace=True)


class GaitResBlock3D(nn.Module):
    """
    OpenGait hybrid block:
      conv1, conv2 → Conv2d applied per-frame (keys .forward_block.0 shape [C,C,3,3])
      shortcut3d   → Conv3d temporal key       (shape [C,C,3,1,1])
      downsample   → [Conv3d, BN3d]            (shape [out,in,1,1,1])
    """
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        s2 = stride
        self.conv1      = _ConvBn2D_fw(in_c,  out_c, k=3, s=s2, p=1)
        self.conv2      = _ConvBn2D_fw(out_c, out_c, k=3, s=1,  p=1)
        self.shortcut3d = nn.Conv3d(out_c, out_c, (3,1,1), 1, (1,0,0), bias=False)
        self.sbn        = nn.BatchNorm3d(out_c)
        self.downsample = nn.Sequential(
            nn.Conv3d(in_c, out_c, 1, (1,s2,s2), 0, bias=False),
            nn.BatchNorm3d(out_c),
        ) if (in_c != out_c or stride != 1) else None

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        identity = x

        out = x.permute(0,2,1,3,4).reshape(B*T, C, H, W)
        out = self.conv1.forward_2d(out)
        _, Co, Ho, Wo = out.shape
        out = out.reshape(B, T, Co, Ho, Wo).permute(0,2,1,3,4)

        out2 = out.permute(0,2,1,3,4).reshape(B*T, Co, Ho, Wo)
        out2 = self.conv2.forward_2d(out2)
        out2 = out2.reshape(B, T, Co, Ho, Wo).permute(0,2,1,3,4)

        temporal = self.sbn(self.shortcut3d(out2))
        out2 = F.relu(out2 + temporal, inplace=True)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(out2 + identity, inplace=True)


# ── Full SkeletonGaitPP ───────────────────────────────────────────────────────

class _SkeletonGaitPP(nn.Module):
    NUM_PARTS = 16

    def __init__(self):
        super().__init__()
        # Dual 2-D stems
        self.sil_layer0 = _ForwardBlock_ConvBn(1, 64)
        self.map_layer0 = _ForwardBlock_ConvBn(2, 64)
        self.sil_layer1 = _ForwardBlock_ResBlock2D(64)
        self.map_layer1 = _ForwardBlock_ResBlock2D(64)
        self.fusion     = _Fusion()

        # Shared 3-D backbone
        self.layer2 = nn.Sequential(
            GaitResBlock3D(64, 128, stride=2),
            GaitResBlock3D(128, 128),
            GaitResBlock3D(128, 128),
            GaitResBlock3D(128, 128),
        )
        self.layer3 = nn.Sequential(
            GaitResBlock3D(128, 256, stride=2),
            GaitResBlock3D(256, 256),
            GaitResBlock3D(256, 256),
            GaitResBlock3D(256, 256),
        )
        self.layer4 = nn.Sequential(
            GaitResBlock3D(256, 512, stride=2),
        )

        # Part-based FC head  (16 × 512 → 256)
        self.FCs     = _FCs(16, 512, 256)
        self.BNNecks = _BNNecks(16, 256, 3000)   # skipped at inference

    def forward(self, sil_seq, map_seq):
        """
        sil_seq: (B, T, H, W)  binary silhouettes (float)
        map_seq: (B, T, H, W) × 2  but we duplicate channel here for pose heatmaps
        Both reshaped to (B, C, T, H, W) for Conv3d.
        """
        B, T, H, W = sil_seq.shape

        # ── 2-D stem (frame-by-frame) ──────────────────────────────────────
        sil2d = sil_seq.view(B * T, 1, H, W) / 255.0
        map2d = map_seq.view(B * T, 1, H, W)      # heatmap already float32
        # map_layer0 expects 2 channels — stack heatmap with itself
        map2d_2ch = map2d.expand(-1, 2, -1, -1)

        sil_feat = self.sil_layer1(self.sil_layer0(sil2d))   # (B*T, 64, H, W)
        map_feat = self.map_layer1(self.map_layer0(map2d_2ch))

        fused = self.fusion(sil_feat, map_feat)               # (B*T, 128, H, W)
        _, C, Hf, Wf = fused.shape
        fused_3d = fused.view(B, T, C, Hf, Wf).permute(0, 2, 1, 3, 4)  # (B,C,T,H,W)

        # ── 3-D backbone (each Sequential applies GaitResBlock3D) ─────────────
        out = self.layer2(fused_3d)  # (B,128,T,H/2,W/2)
        out = self.layer3(out)       # (B,256,T,H/4,W/4)
        out = self.layer4(out)       # (B,512,T,H/8,W/8)

        # Temporal mean pool
        out = out.mean(dim=2)                                   # (B, 512, H'', W'')

        # Horizontal part pooling (16 strips)
        out = _horizontal_pool(out, self.NUM_PARTS)             # (B, 16, 512)

        # Part FC
        emb = self.FCs(out)                                     # (B, 16, 256)
        emb = emb.mean(dim=1)                                   # (B, 256)
        return emb


class _FCs(nn.Module):
    """fc_bin: (num_parts, in_dim, out_dim)"""
    def __init__(self, num_parts, in_dim, out_dim):
        super().__init__()
        self.fc_bin = nn.Parameter(torch.randn(num_parts, in_dim, out_dim))
    def forward(self, x):
        # x: (B, num_parts, in_dim)
        return torch.matmul(x.unsqueeze(2), self.fc_bin.unsqueeze(0)).squeeze(2)


class _BNNecks(nn.Module):
    """Only needed to hold checkpoint keys; unused at inference."""
    def __init__(self, num_parts, in_dim, num_cls):
        super().__init__()
        self.fc_bin = nn.Parameter(torch.randn(num_parts, in_dim, num_cls))
        self.bn1d   = nn.BatchNorm1d(num_parts * in_dim)


def _horizontal_pool(feat: torch.Tensor, num_parts: int) -> torch.Tensor:
    """
    feat: (B, C, H, W)
    Returns (B, num_parts, C) by splitting H into num_parts strips and avg-pooling.
    """
    B, C, H, W = feat.shape
    step = H // num_parts
    parts = []
    for i in range(num_parts):
        strip = feat[:, :, i*step:(i+1)*step, :]
        parts.append(strip.mean(dim=(2, 3)))   # (B, C)
    return torch.stack(parts, dim=1)           # (B, num_parts, C)


# ── Public encoder ────────────────────────────────────────────────────────────

class SkeletonEncoder:
    EMBED_DIM = 256

    def __init__(self, weights_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model  = _SkeletonGaitPP().to(self.device)

        if weights_path and os.path.isfile(weights_path):
            ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
            sd   = ckpt.get("model", ckpt)
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            if missing:
                logger.warning(f"SkeletonEncoder: {len(missing)} missing keys (head excluded by design).")
            logger.info(f"SkeletonEncoder: loaded weights from {weights_path}")
        else:
            logger.warning(f"SkeletonEncoder: no weights at '{weights_path}' — random init.")

        self.model.eval()

    @staticmethod
    def _l2(arr): n = np.linalg.norm(arr); return arr / n if n > 1e-8 else arr

    def _prep(self, sil_seq, hm_seq):
        sil = torch.tensor(sil_seq[None], dtype=torch.float32).to(self.device)
        hm  = torch.tensor(hm_seq[None],  dtype=torch.float32).to(self.device)
        return sil, hm

    def encode(self, heatmap_seq: np.ndarray, silhouette_seq: np.ndarray) -> np.ndarray:
        """(T,H,W) float32 + (T,H,W) uint8 → 256-d unit-norm embedding."""
        sil, hm = self._prep(silhouette_seq, heatmap_seq)
        with torch.no_grad():
            emb = self.model(sil, hm)
        return self._l2(emb.squeeze(0).cpu().numpy())

    def encode_batch(self, hm_list: List[np.ndarray], sil_list: List[np.ndarray]) -> np.ndarray:
        sil = torch.stack([torch.tensor(s, dtype=torch.float32) for s in sil_list]).to(self.device)
        hm  = torch.stack([torch.tensor(h, dtype=torch.float32) for h in hm_list]).to(self.device)
        with torch.no_grad():
            embs = self.model(sil, hm).cpu().numpy()
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return np.where(norms > 1e-8, embs / norms, embs)
