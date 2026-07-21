"""JFIF (BT.601 full-range) color conversion and 4:2:0 chroma resampling.

All tensors operate in the [0, 255] domain to mirror libjpeg exactly.
"""

import torch
import torch.nn.functional as F


def rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) RGB in [0, 255] -> YCbCr in [0, 255]."""
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    return torch.stack([y, cb, cr], dim=1)


def ycbcr_to_rgb(x: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) YCbCr in [0, 255] -> RGB in [0, 255]."""
    y, cb, cr = x[:, 0], x[:, 1] - 128.0, x[:, 2] - 128.0
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.stack([r, g, b], dim=1)


def chroma_downsample(c: torch.Tensor) -> torch.Tensor:
    """4:2:0 subsampling via 2x2 average pooling. (B, 2, H, W) -> (B, 2, H/2, W/2)."""
    return F.avg_pool2d(c, kernel_size=2, stride=2)


def chroma_upsample(c: torch.Tensor) -> torch.Tensor:
    """(B, 2, H/2, W/2) -> (B, 2, H, W), bilinear (close to libjpeg fancy upsampling)."""
    return F.interpolate(c, scale_factor=2, mode="bilinear", align_corners=False)
