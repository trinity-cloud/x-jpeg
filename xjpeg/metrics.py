"""Quality metrics and Bjontegaard-delta computation."""

import numpy as np
import torch
from PIL import Image
from pytorch_msssim import ms_ssim

from .color import rgb_to_ycbcr


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB -> (1, 3, H, W) float tensor in [0, 1]."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def msssim(a: torch.Tensor, b: torch.Tensor) -> float:
    return ms_ssim(a, b, data_range=1.0).item()


def ycbcr611_msssim(
    a: torch.Tensor, b: torch.Tensor, size_average: bool = True,
) -> torch.Tensor:
    """BT.601 YCbCr MS-SSIM with the conventional 6:1:1 component weights.

    Inputs are RGB tensors in [0, 1]. The return type follows
    :func:`pytorch_msssim.ms_ssim`: a scalar when ``size_average`` is true,
    otherwise one score per batch element.
    """
    a_ycc = rgb_to_ycbcr(a * 255.0) / 255.0
    b_ycc = rgb_to_ycbcr(b * 255.0) / 255.0
    scores = [
        ms_ssim(
            a_ycc[:, index:index + 1], b_ycc[:, index:index + 1],
            data_range=1.0, size_average=size_average,
        )
        for index in range(3)
    ]
    return (6.0 * scores[0] + scores[1] + scores[2]) / 8.0


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def bpp_of(data: bytes, img: Image.Image) -> float:
    return len(data) * 8.0 / (img.width * img.height)


def bd_rate(rate_anchor, dist_anchor, rate_test, dist_test) -> float:
    """Bjontegaard delta-rate (%) of test vs anchor over the overlap region.

    rate_*: bits per pixel (positive); dist_*: quality metric where higher is
    better (PSNR in dB, or e.g. -10*log10(1 - MS-SSIM) for MS-SSIM curves).
    Negative result means the test codec needs fewer bits at equal quality.
    """
    lr_a, d_a = np.log(np.asarray(rate_anchor)), np.asarray(dist_anchor)
    lr_t, d_t = np.log(np.asarray(rate_test)), np.asarray(dist_test)
    pa = np.polyfit(d_a, lr_a, 3)
    pt = np.polyfit(d_t, lr_t, 3)
    lo = max(d_a.min(), d_t.min())
    hi = min(d_a.max(), d_t.max())
    if hi <= lo:
        raise ValueError("RD curves do not overlap in quality range")
    ia = np.polyint(pa)
    it = np.polyint(pt)
    avg_a = (np.polyval(ia, hi) - np.polyval(ia, lo)) / (hi - lo)
    avg_t = (np.polyval(it, hi) - np.polyval(it, lo)) / (hi - lo)
    return float((np.exp(avg_t - avg_a) - 1.0) * 100.0)


def msssim_db(v) -> np.ndarray:
    """MS-SSIM -> dB scale commonly used for BD-rate on SSIM-type metrics."""
    v = np.asarray(v, dtype=np.float64)
    return -10.0 * np.log10(np.clip(1.0 - v, 1e-12, None))
