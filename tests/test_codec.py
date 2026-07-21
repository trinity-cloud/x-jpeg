import numpy as np
import torch
from PIL import Image

from xjpeg.codec import XJPEG
from xjpeg.jpeg_io import read_tables
from xjpeg.model import QTableNet


def test_share_chroma_keeps_three_table_prediction_but_emits_two_dqts():
    yy, xx = np.indices((256, 256))
    image = Image.fromarray(
        np.stack([
            (xx + yy) % 256,
            (2 * xx + yy) % 256,
            (xx + 2 * yy) % 256,
        ], axis=-1).astype(np.uint8),
        "RGB",
    )
    codec = object.__new__(XJPEG)
    codec.predict_tables = lambda _: (
        np.full((8, 8), 16),
        np.full((8, 8), 20),
        np.full((8, 8), 40),
    )

    result = codec.compress(image, share_chroma=True, backend="pillow")
    emitted = read_tables(result.data)

    assert len(emitted) == 2
    assert np.array_equal(emitted[0], np.full((8, 8), 16))
    assert np.array_equal(emitted[1], np.full((8, 8), 30))
    assert np.array_equal(result.cb_table, result.cr_table)


def test_codec_infers_nondefault_compact_widths(tmp_path):
    model = QTableNet(
        architecture="compact", gdn_reparam=True,
        width_multiplier=1.25,
    )
    checkpoint = tmp_path / "wide-encoder.pt"
    torch.save(
        {
            "model": {
                f"encoder.{key}": value
                for key, value in model.encoder.state_dict().items()
            },
            "architecture": "compact",
            "gdn_reparam": True,
            "width_multiplier": 1.25,
        },
        checkpoint,
    )
    codec = XJPEG(checkpoint)
    assert codec.encoder.blocks[0].conv.out_channels == 80
    assert codec.encoder.blocks[-1].conv.out_channels == 320
