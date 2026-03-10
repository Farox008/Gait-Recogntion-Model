"""
silhouette_encoder.py

Exact PyTorch reimplementation of DeepGaitV2 whose module names and
parameter shapes match the OpenGait checkpoint at:
  Gait3D/DeepGaitV2/DeepGaitV2/checkpoints/DeepGaitV2-60000.pt

Architecture (from full key inspection):
  layer0  : ConvBn(1→64, 3×3)               ← forward_block = [Conv2d, BN]
  layer1  : ForwardBlock(ResBlock2D(64→64))  ← forward_block.0 = ResBlock2D
  layer2  : 4 × GaitResBlock3D(64→128)
  layer3  : 4 × GaitResBlock3D(128→256)
  layer4  : 1 × GaitResBlock3D(256→512)
  FCs.fc_bin   : [16, 512, 256]
  BNNecks      : skipped at inference

Output: 256-d unit-norm embedding.
"""
import logging, os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── Building blocks (same as skeleton_encoder.py) ────────────────────────────

class _ConvBn(nn.Sequential):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
        )

class _ForwardBlock_ConvBn(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.forward_block = _ConvBn(in_c, out_c)
    def forward(self, x):
        return F.relu(self.forward_block(x), inplace=True)


class _ResBlock2D(nn.Module):
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
    def __init__(self, channels):
        super().__init__()
        self.forward_block = nn.Sequential(_ResBlock2D(channels))
    def forward(self, x):
        return self.forward_block(x)


class _ConvBn2D_fw(nn.Module):
    """Conv2d+BN with .forward_block  (applied per-frame in the 3-D backbone)."""
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.forward_block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
        )
    def forward_2d(self, x):
        """x: (B*T, C, H, W) → ReLU(conv+bn)"""
        return F.relu(self.forward_block(x), inplace=True)


class GaitResBlock3D(nn.Module):
    """
    OpenGait hybrid block:
      conv1, conv2 → Conv2d  (applied per-frame, keys end in .forward_block.0.weight [C,C,3,3])
      shortcut3d   → Conv3d  (temporal, key shape [C,C,3,1,1])
      downsample   → [Conv3d, BN3d]  (shape [out_c,in_c,1,1,1])
    """
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        s2 = stride   # spatial stride for Conv2d
        self.conv1 = _ConvBn2D_fw(in_c,  out_c, k=3, s=s2, p=1)
        self.conv2 = _ConvBn2D_fw(out_c, out_c, k=3, s=1,  p=1)
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

        # Apply 2-D conv frame by frame
        out = x.permute(0,2,1,3,4).reshape(B*T, C, H, W)
        out = self.conv1.forward_2d(out)              # (B*T, out_c, H', W')
        _, Cout, Hout, Wout = out.shape
        out = out.reshape(B, T, Cout, Hout, Wout).permute(0,2,1,3,4)  # (B,out_c,T,H',W')

        out2 = out.permute(0,2,1,3,4).reshape(B*T, Cout, Hout, Wout)
        out2 = self.conv2.forward_2d(out2)            # (B*T, out_c, H', W')
        out2 = out2.reshape(B, T, Cout, Hout, Wout).permute(0,2,1,3,4)  # (B,out_c,T,H',W')

        # 3-D temporal shortcut + BN
        temporal = self.sbn(self.shortcut3d(out2))
        out2 = F.relu(out2 + temporal, inplace=True)

        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(out2 + identity, inplace=True)


class _FCs(nn.Module):
    def __init__(self, num_parts, in_dim, out_dim):
        super().__init__()
        self.fc_bin = nn.Parameter(torch.randn(num_parts, in_dim, out_dim))
    def forward(self, x):
        return torch.matmul(x.unsqueeze(2), self.fc_bin.unsqueeze(0)).squeeze(2)


class _BNNecks(nn.Module):
    def __init__(self, num_parts, in_dim, num_cls):
        super().__init__()
        self.fc_bin = nn.Parameter(torch.randn(num_parts, in_dim, num_cls))
        self.bn1d   = nn.BatchNorm1d(num_parts * in_dim)


def _horizontal_pool(feat, num_parts):
    B, C, H, W = feat.shape
    step = H // num_parts
    return torch.stack([
        feat[:, :, i*step:(i+1)*step, :].mean(dim=(2, 3))
        for i in range(num_parts)
    ], dim=1)


# ── Full DeepGaitV2 ───────────────────────────────────────────────────────────

class _DeepGaitV2(nn.Module):
    NUM_PARTS = 16

    def __init__(self):
        super().__init__()
        self.layer0 = _ForwardBlock_ConvBn(1, 64)
        self.layer1 = _ForwardBlock_ResBlock2D(64)

        self.layer2 = nn.Sequential(
            GaitResBlock3D(64,  128, stride=2),
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
        self.FCs     = _FCs(self.NUM_PARTS, 512, 256)
        self.BNNecks = _BNNecks(self.NUM_PARTS, 256, 3000)  # unused at inference

    def forward(self, sil_seq: torch.Tensor) -> torch.Tensor:
        # sil_seq: (B, T, H, W)  float32  [0..255]
        B, T, H, W = sil_seq.shape
        x = sil_seq.view(B * T, 1, H, W) / 255.0      # (B*T, 1, H, W)
        x = self.layer1(self.layer0(x))                 # (B*T, 64, H, W)

        _, C, Hf, Wf = x.shape
        x = x.view(B, T, C, Hf, Wf).permute(0, 2, 1, 3, 4)  # (B,64,T,H,W)

        x = self.layer2(x)   # (B,128,T,H/2,W/2)
        x = self.layer3(x)   # (B,256,T,H/4,W/4)
        x = self.layer4(x)   # (B,512,T,H/8,W/8)
        x = x.mean(dim=2)    # temporal mean → (B,512,H'',W'')
        x = _horizontal_pool(x, self.NUM_PARTS)   # (B,16,512)
        return self.FCs(x).mean(dim=1)             # (B,256)


# ── Public encoder ────────────────────────────────────────────────────────────

class SilhouetteEncoder:
    EMBED_DIM = 256

    def __init__(self, weights_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model  = _DeepGaitV2().to(self.device)

        if weights_path and os.path.isfile(weights_path):
            ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
            sd   = ckpt.get("model", ckpt)
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            if missing:
                logger.warning(f"SilhouetteEncoder: {len(missing)} missing keys (head skipped by design).")
            logger.info(f"SilhouetteEncoder: loaded weights from {weights_path}")
        else:
            logger.warning(f"SilhouetteEncoder: no weights at '{weights_path}' — random init.")

        self.model.eval()

    @staticmethod
    def _l2(arr): n = np.linalg.norm(arr); return arr / n if n > 1e-8 else arr

    def encode(self, silhouette_seq: np.ndarray) -> np.ndarray:
        """(T, H, W) uint8 → 256-d unit-norm embedding."""
        t = torch.tensor(silhouette_seq[None], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            emb = self.model(t)
        return self._l2(emb.squeeze(0).cpu().numpy())

    def encode_batch(self, sequences: List[np.ndarray]) -> np.ndarray:
        t = torch.stack([torch.tensor(s, dtype=torch.float32) for s in sequences]).to(self.device)
        with torch.no_grad():
            embs = self.model(t).cpu().numpy()
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return np.where(norms > 1e-8, embs / norms, embs)
