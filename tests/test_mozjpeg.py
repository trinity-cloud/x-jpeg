import os

import numpy as np
import pytest
from PIL import Image

from xjpeg.jpeg_io import decode, read_tables
from xjpeg.mozjpeg import encode_mozjpeg, find_mozjpeg, mozjpeg_version


def test_bundled_or_explicit_mozjpeg_round_trip():
    binary = find_mozjpeg(os.environ.get("XJPEG_MOZJPEG"))
    if binary is None:
        pytest.skip("MozJPEG is not available in this source install")
    yy, xx = np.indices((256, 256))
    pixels = np.stack([
        (xx + yy) % 256,
        (2 * xx + yy) % 256,
        (xx + 2 * yy) % 256,
    ], axis=-1).astype(np.uint8)
    image = Image.fromarray(pixels, "RGB")
    data = encode_mozjpeg(
        image,
        np.full((8, 8), 16),
        np.full((8, 8), 24),
    )
    assert decode(data).size == image.size
    assert len(read_tables(data)) == 2
    assert "mozjpeg" in mozjpeg_version().lower()

