"""8x8 block DCT-II / IDCT, exactly as used by JPEG, in batched tensor form."""

import math

import torch


def dct_matrix(n: int = 8, dtype=torch.float32, device=None) -> torch.Tensor:
    """Orthonormal DCT-II matrix D such that X_freq = D @ x @ D.T."""
    k = torch.arange(n, dtype=dtype, device=device).unsqueeze(1)
    i = torch.arange(n, dtype=dtype, device=device).unsqueeze(0)
    d = torch.cos((2 * i + 1) * k * math.pi / (2 * n)) * math.sqrt(2.0 / n)
    d[0] *= 1.0 / math.sqrt(2.0)
    return d


def blockify(x: torch.Tensor, size: int = 8) -> torch.Tensor:
    """(B, C, H, W) -> (B, C, H/size, W/size, size, size)."""
    b, c, h, w = x.shape
    x = x.reshape(b, c, h // size, size, w // size, size)
    return x.permute(0, 1, 2, 4, 3, 5).contiguous()


def unblockify(x: torch.Tensor) -> torch.Tensor:
    """Inverse of blockify."""
    b, c, hb, wb, s, _ = x.shape
    return x.permute(0, 1, 2, 4, 3, 5).reshape(b, c, hb * s, wb * s)


def block_dct(blocks: torch.Tensor, dmat: torch.Tensor) -> torch.Tensor:
    """Apply D @ block @ D.T over trailing (8, 8) dims."""
    return torch.einsum("ij,...jk,lk->...il", dmat, blocks, dmat)


def block_idct(coeffs: torch.Tensor, dmat: torch.Tensor) -> torch.Tensor:
    """Apply D.T @ coeffs @ D over trailing (8, 8) dims."""
    return torch.einsum("ji,...jk,kl->...il", dmat, coeffs, dmat)
