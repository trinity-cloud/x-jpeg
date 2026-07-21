"""Differentiable JPEG round-trip with per-image predicted quantization tables.

Pipeline (matches libjpeg with 4:2:0 subsampling):
    RGB -> YCbCr -> chroma downsample -> level shift (-128) -> 8x8 block DCT
    -> divide by Q-table -> round -> multiply by Q-table -> IDCT -> +128
    -> chroma upsample -> RGB

Rounding uses the Shin & Song cubic approximation, which equals true rounding
in the forward pass value up to (x - round(x))^3 and has smooth gradients.
"""

import torch
import torch.nn as nn

from .color import (
    chroma_downsample,
    chroma_upsample,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
)
from .dct import block_dct, block_idct, blockify, dct_matrix, unblockify


def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Straight-through rounding: EXACT rounding forward, identity gradient.

    The forward value is bit-identical to real quantization, so the training
    distortion cannot be gamed by parking coefficients where an approximation
    leaks signal through.
    """
    return x + (torch.round(x) - x).detach()


def diff_round(x: torch.Tensor) -> torch.Tensor:
    """Cubic rounding x + (round(x) - x)^3 (Shin & Song, 2017). Kept for
    reference/ablation only: a LEARNED quantizer exploits its dead zone
    (passes ~84% of a coefficient at |frac| ~ 0.4 that real JPEG zeroes),
    inflating training quality by ~0.05 MS-SSIM vs real files."""
    return x + (torch.round(x) - x) ** 3


class DiffJPEG(nn.Module):
    """Differentiable JPEG codec.

    forward(x, qtables) -> (x_hat, scaled) where
      x:       (B, 3, H, W) RGB in [0, 1], H and W multiples of 16
      qtables: (B, 3, 8, 8) with entries in [1, 255]; channels are Y, Cb,
               and Cr. Two-table tensors share channel 1 across Cb/Cr.
      x_hat:   reconstruction in [0, 1]
      scaled:  dict with continuous quantizer inputs DCT/Q per component
               ('y': (B,1,hb,wb,8,8), 'c': (B,2,hb,wb,8,8)) for rate modeling
    """

    def __init__(self, rounding: str = "ste"):
        super().__init__()
        if rounding not in ("ste", "cubic"):
            raise ValueError(f"unknown rounding: {rounding}")
        self.rounding = rounding
        self.register_buffer("dmat", dct_matrix(8), persistent=False)

    def _component_forward(self, comp: torch.Tensor, table: torch.Tensor, hard: bool):
        # comp: (B, C, H, W) level-shifted; table: (B, C, 8, 8)
        blocks = blockify(comp)
        coeffs = block_dct(blocks, self.dmat)
        t = table[:, :, None, None, :, :]  # broadcast over (hb, wb) and C
        scaled = coeffs / t
        if hard:
            symbols = torch.round(scaled)
        else:
            symbols = ste_round(scaled) if self.rounding == "ste" else diff_round(scaled)
        recon = block_idct(symbols * t, self.dmat)
        return unblockify(recon), scaled

    def forward(self, x: torch.Tensor, qtables: torch.Tensor, hard: bool = False):
        if x.shape[-1] % 16 or x.shape[-2] % 16:
            raise ValueError(f"H, W must be multiples of 16, got {tuple(x.shape)}")
        ycc = rgb_to_ycbcr(x * 255.0)
        y = ycc[:, 0:1] - 128.0
        c = chroma_downsample(ycc[:, 1:3]) - 128.0

        if qtables.shape[1] == 2:
            c_tables = qtables[:, 1:2].expand(-1, 2, -1, -1)
        elif qtables.shape[1] == 3:
            c_tables = qtables[:, 1:3]
        else:
            raise ValueError(f"expected 2 or 3 Q-tables, got {qtables.shape[1]}")

        y_hat, y_scaled = self._component_forward(y, qtables[:, 0:1], hard)
        c_hat, c_scaled = self._component_forward(c, c_tables, hard)

        ycc_hat = torch.cat([y_hat + 128.0, chroma_upsample(c_hat) + 128.0], dim=1)
        x_hat = (ycbcr_to_rgb(ycc_hat) / 255.0).clamp(0.0, 1.0)
        return x_hat, {"y": y_scaled, "c": c_scaled}
