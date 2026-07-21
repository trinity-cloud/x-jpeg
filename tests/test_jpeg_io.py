import numpy as np
from PIL import Image

from xjpeg.jpeg_io import (
    STD_LUMA,
    decode,
    encode_standard,
    encode_with_tables,
    read_tables,
    standard_tables,
    table_to_zigzag,
    zigzag_to_table,
)


def _test_image(w=128, h=96):
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, size=(h // 8, w // 8, 3), dtype=np.uint8)
    arr = np.kron(base, np.ones((8, 8, 1), dtype=np.uint8))
    return Image.fromarray(arr, "RGB")


def test_zigzag_first_entries():
    zz = table_to_zigzag(STD_LUMA)
    # ITU-T T.81: zig-zag of the Annex-K luma table starts 16,11,12,14,12,10
    assert zz[:6] == [16, 11, 12, 14, 12, 10]
    assert np.array_equal(zigzag_to_table(zz), STD_LUMA)


def test_custom_table_roundtrip():
    # asymmetric table so any order mixup would be caught
    ty = np.arange(1, 65).reshape(8, 8)
    tc = np.arange(64, 0, -1).reshape(8, 8)
    data = encode_with_tables(_test_image(), ty, tc)
    tables = read_tables(data)
    assert np.array_equal(tables[0], ty)
    assert np.array_equal(tables[1], tc)


def test_three_custom_tables_and_component_selectors():
    ty = np.arange(1, 65).reshape(8, 8)
    tcb = np.arange(64, 0, -1).reshape(8, 8)
    tcr = np.full((8, 8), 37)
    data = encode_with_tables(_test_image(), ty, tcb, tcr)
    img = Image.open(__import__("io").BytesIO(data))
    tables = read_tables(data)
    assert np.array_equal(tables[0], ty)
    assert np.array_equal(tables[1], tcb)
    assert np.array_equal(tables[2], tcr)
    assert [layer[3] for layer in img.layer] == [0, 1, 2]


def test_quality50_is_annex_k():
    data = encode_standard(_test_image(), quality=50)
    assert np.array_equal(read_tables(data)[0], STD_LUMA)


def test_standard_tables_scaling():
    t50, _ = standard_tables(50)
    assert np.array_equal(t50, STD_LUMA)
    t25, _ = standard_tables(25)
    assert np.array_equal(t25, np.clip((STD_LUMA * 200 + 50) // 100, 1, 255))


def test_decode_shape():
    img = _test_image()
    out = decode(encode_standard(img, 80))
    assert out.size == img.size
