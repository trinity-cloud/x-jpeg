"""Full-resolution DCT band statistics: the encoder's view of what the
quantizer will actually face.

The 256x256 thumbnail destroys exactly the high-frequency statistics that
distinguish images for quantization purposes, which collapses the predictor
to a static table. This side channel summarizes the FULL-RES image as the
per-band mean absolute DCT coefficient (3x8x8: Y + Cb + Cr), computed
identically at training and inference (CPU-cheap: one blockwise DCT).
"""

import torch

from .color import chroma_downsample, rgb_to_ycbcr
from .dct import block_dct, blockify, dct_matrix


def dct_band_stats(x: torch.Tensor, num_tables: int = 3) -> torch.Tensor:
    """x: (B, 3, H, W) RGB in [0, 1] at native resolution.
    Returns one log-mean-|DCT| map per quantization table. In two-table mode,
    the Cb and Cr maps are averaged.
    """
    if num_tables not in (2, 3):
        raise ValueError(f"num_tables must be 2 or 3, got {num_tables}")
    h, w = x.shape[-2:]
    if h < 16 or w < 16:
        raise ValueError(f"H, W must be at least 16, got {(h, w)}")
    x = x[..., : h - h % 16, : w - w % 16]
    ycc = rgb_to_ycbcr(x * 255.0)
    y = ycc[:, 0:1] - 128.0
    c = chroma_downsample(ycc[:, 1:3]) - 128.0
    dmat = dct_matrix(8, device=x.device, dtype=x.dtype)
    def band_mean(comp):
        return block_dct(blockify(comp), dmat).abs().mean(dim=(1, 2, 3))

    fy = band_mean(y)
    fcb = band_mean(c[:, 0:1])
    fcr = band_mean(c[:, 1:2])
    bands = (fy, (fcb + fcr) / 2.0) if num_tables == 2 else (fy, fcb, fcr)
    return torch.stack([torch.log1p(f) for f in bands], dim=1)
