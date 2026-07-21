import torch

from xjpeg.metrics import ycbcr611_msssim


def test_ycbcr611_identity_scalar_and_batch_modes():
    torch.manual_seed(0)
    images = torch.rand(2, 3, 192, 192)

    scalar = ycbcr611_msssim(images, images)
    per_image = ycbcr611_msssim(images, images, size_average=False)

    assert scalar.shape == ()
    assert per_image.shape == (2,)
    assert torch.allclose(scalar, torch.ones_like(scalar), atol=1e-6)
    assert torch.allclose(per_image, torch.ones_like(per_image), atol=1e-6)
