import numpy as np
import pytest
import torch

from xjpeg.dct import block_dct, block_idct, blockify, dct_matrix, unblockify

scipy_fft = pytest.importorskip("scipy.fft")


def test_orthonormal():
    d = dct_matrix(8, dtype=torch.float64)
    assert torch.allclose(d @ d.T, torch.eye(8, dtype=torch.float64), atol=1e-12)


def test_roundtrip():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32, 32, dtype=torch.float64)
    d = dct_matrix(8, dtype=torch.float64)
    blocks = blockify(x)
    rec = unblockify(block_idct(block_dct(blocks, d), d))
    assert torch.allclose(rec, x, atol=1e-10)


def test_matches_scipy():
    torch.manual_seed(1)
    block = torch.randn(8, 8, dtype=torch.float64)
    d = dct_matrix(8, dtype=torch.float64)
    ours = block_dct(block, d).numpy()
    ref = scipy_fft.dctn(block.numpy(), type=2, norm="ortho")
    np.testing.assert_allclose(ours, ref, atol=1e-12)


def test_blockify_layout():
    x = torch.arange(16 * 16, dtype=torch.float32).reshape(1, 1, 16, 16)
    b = blockify(x)
    assert b.shape == (1, 1, 2, 2, 8, 8)
    # top-left block must equal the top-left 8x8 patch
    assert torch.equal(b[0, 0, 0, 0], x[0, 0, :8, :8])
    assert torch.equal(unblockify(b), x)
