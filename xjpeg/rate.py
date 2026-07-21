"""Differentiable bitrate proxy for quantized DCT coefficients.

Each of the 64 frequency bands (separately for luma and chroma) is modeled as
a Laplacian with learned mean and scale. The rate of a quantized symbol s is
-log2 P(s - 0.5 < X < s + 0.5); during training symbols are relaxed with
additive uniform noise (Balle et al.), at eval they are hard-rounded.

This proxies the entropy-coding stage that is not differentiable in real
JPEG. It over-counts slightly (real JPEG uses DPCM on DC plus run-length
Huffman) but is monotone in the true rate, which is what the training
pressure needs; evaluation always measures real file bytes.
"""

import torch
import torch.nn as nn


class DctRateModel(nn.Module):
    def __init__(self, num_components: int = 3):
        super().__init__()
        if num_components not in (2, 3):
            raise ValueError(f"num_components must be 2 or 3, got {num_components}")
        self.num_components = num_components
        # Three-table mode: index 0 = Y, 1 = Cb, 2 = Cr.
        # Two-table mode shares index 1 across the chroma components.
        self.mean = nn.Parameter(torch.zeros(num_components, 8, 8))
        # AC symbols concentrate near zero once divided by the table: start
        # their scales small so zeros are cheap from step one; DC is wide.
        log_scale = torch.full((num_components, 8, 8), -2.0)
        log_scale[:, 0, 0] = 0.0
        self.log_scale = nn.Parameter(log_scale)

    def bits(self, scaled: torch.Tensor, comp: int) -> torch.Tensor:
        """Total bits for one component.

        scaled: (B, C, hb, wb, 8, 8) continuous quantizer input (DCT / Q).
        """
        if self.training:
            symbols = scaled + torch.empty_like(scaled).uniform_(-0.5, 0.5)
        else:
            symbols = torch.round(scaled)
        loc = self.mean[comp]
        scale = torch.exp(self.log_scale[comp]).clamp_min(1e-2)
        dist = torch.distributions.Laplace(loc, scale, validate_args=False)
        p = dist.cdf(symbols + 0.5) - dist.cdf(symbols - 0.5)
        return -torch.log2(p.clamp_min(1e-9)).sum()

    def bpp(self, scaled: dict, num_pixels: int) -> torch.Tensor:
        """Bits per pixel of the full image from the diffjpeg 'scaled' dict."""
        total = self.bits(scaled["y"], 0)
        if self.num_components == 2:
            total = total + self.bits(scaled["c"], 1)
        else:
            total = total + self.bits(scaled["c"][:, 0:1], 1)
            total = total + self.bits(scaled["c"][:, 1:2], 2)
        return total / num_pixels
