import numpy as np
import torch
from PIL import Image

from xjpeg.diffjpeg import DiffJPEG, diff_round, ste_round
from xjpeg.jpeg_io import STD_CHROMA, STD_LUMA, decode, encode_with_tables
from xjpeg.metrics import pil_to_tensor, psnr


def _natural_image(size=256):
    """Smooth random image (noise blurred by pooling) — natural-ish statistics."""
    torch.manual_seed(0)
    x = torch.rand(1, 3, size * 4, size * 4)
    x = torch.nn.functional.avg_pool2d(x, 4)
    return x.clamp(0, 1)


def _std_tables_batch(b=1):
    t = torch.stack([
        torch.tensor(STD_LUMA, dtype=torch.float32),
        torch.tensor(STD_CHROMA, dtype=torch.float32),
        torch.tensor(STD_CHROMA, dtype=torch.float32),
    ])
    return t.unsqueeze(0).repeat(b, 1, 1, 1)


def test_diff_round_close_to_round():
    x = torch.linspace(-3, 3, 1001)
    # |diff_round(x) - round(x)| = |d - d^3|, d = x - round(x); max 0.375 at d=0.5
    assert (diff_round(x) - torch.round(x)).abs().max() <= 0.375 + 1e-6
    ints = torch.arange(-3, 4, dtype=torch.float32)
    assert torch.equal(diff_round(ints), ints)


def test_ste_round_exact_forward_identity_grad():
    x = torch.linspace(-3, 3, 1001, requires_grad=True)
    y = ste_round(x)
    assert torch.equal(y, torch.round(x))  # bit-exact real quantization
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_training_forward_matches_hard_forward():
    """The anti-gaming property: soft (training) and hard (real) paths must
    produce IDENTICAL reconstructions under STE rounding."""
    x = _natural_image(64)
    dj = DiffJPEG()
    soft, _ = dj(x, _std_tables_batch(), hard=False)
    hard, _ = dj(x, _std_tables_batch(), hard=True)
    assert torch.equal(soft, hard)


def test_gradients_flow_to_tables():
    x = _natural_image(64)
    tables = _std_tables_batch().requires_grad_(True)
    x_hat, _ = DiffJPEG()(x, tables)
    (1 - x_hat.mean()).backward()
    assert tables.grad is not None
    assert torch.isfinite(tables.grad).all()
    assert tables.grad.abs().sum() > 0


def test_matches_real_libjpeg():
    """Hard-rounded diff-JPEG must land close to an actual libjpeg round-trip."""
    x = _natural_image(256)
    arr = (x[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    img = Image.fromarray(arr, "RGB")

    ours, _ = DiffJPEG()(x, _std_tables_batch(), hard=True)
    real = pil_to_tensor(decode(encode_with_tables(img, STD_LUMA, STD_CHROMA)))

    # Not bit-exact (libjpeg integer DCT, different chroma filters), but the
    # two round-trips must agree far more closely than either does with the
    # original: this catches any color/DCT/quantization convention error.
    agree = psnr(ours, real)
    vs_orig = psnr(real, x)
    assert agree > 30.0, f"diffjpeg vs libjpeg only {agree:.1f} dB"
    assert agree > vs_orig + 5.0, (
        f"diffjpeg({agree:.1f} dB) barely closer to libjpeg than original ({vs_orig:.1f} dB)"
    )


def _smooth_image(size=128):
    """Low-frequency content JPEG represents well, so table coarseness shows."""
    t = torch.linspace(0, 1, size)
    yy, xx = torch.meshgrid(t, t, indexing="ij")
    r = 0.5 + 0.35 * torch.sin(6.28 * (3 * xx + yy))
    g = 0.5 + 0.35 * torch.cos(6.28 * (xx + 2 * yy))
    b = 0.5 + 0.30 * torch.sin(6.28 * (xx * xx + yy))
    img = torch.stack([r, g, b]).unsqueeze(0)
    torch.manual_seed(0)
    return (img + 0.02 * torch.randn_like(img)).clamp(0, 1)


def test_coarser_tables_hurt_quality():
    x = _smooth_image(128)
    dj = DiffJPEG()
    fine, _ = dj(x, _std_tables_batch() * 0.5, hard=True)
    coarse, _ = dj(x, (_std_tables_batch() * 4).clamp(1, 255), hard=True)
    assert psnr(fine, x) > psnr(coarse, x) + 3.0


def test_separate_cb_cr_tables_both_affect_output():
    x = _natural_image(64)
    tables = _std_tables_batch()
    ref, _ = DiffJPEG()(x, tables, hard=True)
    cb_changed = tables.clone()
    cb_changed[:, 1] *= 4
    cr_changed = tables.clone()
    cr_changed[:, 2] *= 4
    out_cb, _ = DiffJPEG()(x, cb_changed, hard=True)
    out_cr, _ = DiffJPEG()(x, cr_changed, hard=True)
    assert not torch.equal(ref, out_cb)
    assert not torch.equal(ref, out_cr)
