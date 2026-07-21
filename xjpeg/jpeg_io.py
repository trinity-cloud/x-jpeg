"""Real JPEG encode/decode with custom quantization tables via Pillow/libjpeg.

Pillow's `qtables` save argument and `Image.quantization` attribute use a
table order that has varied across versions (zig-zag vs natural). Rather than
hardcode an assumption, we detect the convention once at import-usage time by
saving with the standard Annex-K luma table and reading it back.
"""

import io
from functools import lru_cache

import numpy as np
from PIL import Image

# ITU-T T.81 Annex K reference tables (natural / row-major order)
STD_LUMA = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.int64,
)
STD_CHROMA = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=np.int64,
)


@lru_cache(maxsize=1)
def zigzag_positions():
    """Zig-zag scan order as a list of (row, col), length 64."""
    return sorted(
        ((r, c) for r in range(8) for c in range(8)),
        key=lambda rc: (rc[0] + rc[1], rc[0] if (rc[0] + rc[1]) % 2 else -rc[0]),
    )


def table_to_zigzag(table_8x8) -> list:
    t = np.asarray(table_8x8)
    return [int(t[r, c]) for r, c in zigzag_positions()]


def zigzag_to_table(flat64) -> np.ndarray:
    t = np.zeros((8, 8), dtype=np.int64)
    for v, (r, c) in zip(flat64, zigzag_positions()):
        t[r, c] = int(v)
    return t


@lru_cache(maxsize=1)
def _pillow_qtable_order() -> str:
    """Detect whether Pillow's qtable lists are 'natural' or 'zigzag' order."""
    img = Image.fromarray(
        (np.indices((64, 64)).sum(0) * 2 % 256).astype(np.uint8)
    ).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=50)  # quality 50 == Annex-K tables verbatim
    buf.seek(0)
    got = list(Image.open(buf).quantization[0])
    if got == [int(v) for v in STD_LUMA.flatten()]:
        return "natural"
    if got == table_to_zigzag(STD_LUMA):
        return "zigzag"
    raise RuntimeError("Cannot determine Pillow qtable order; got %r" % (got[:8],))


def _to_pillow_order(table_8x8) -> list:
    if _pillow_qtable_order() == "zigzag":
        return table_to_zigzag(table_8x8)
    return [int(v) for v in np.asarray(table_8x8).flatten()]


def _from_pillow_order(flat64) -> np.ndarray:
    if _pillow_qtable_order() == "zigzag":
        return zigzag_to_table(flat64)
    return np.asarray(flat64, dtype=np.int64).reshape(8, 8)


def encode_with_tables(
    img: Image.Image,
    luma_table,
    cb_table,
    cr_table=None,
    subsampling: int = 2,
    optimize: bool = True,
    progressive: bool = False,
) -> bytes:
    """Encode a PIL image to JPEG bytes with explicit quantization tables.

    subsampling: Pillow convention — 0: 4:4:4, 1: 4:2:2, 2: 4:2:0.
    """
    ql = np.clip(np.round(np.asarray(luma_table, dtype=np.float64)), 1, 255)
    qcb = np.clip(np.round(np.asarray(cb_table, dtype=np.float64)), 1, 255)
    qtables = {0: _to_pillow_order(ql), 1: _to_pillow_order(qcb)}
    if cr_table is not None:
        qcr = np.clip(np.round(np.asarray(cr_table, dtype=np.float64)), 1, 255)
        qtables[2] = _to_pillow_order(qcr)
    buf = io.BytesIO()
    img.convert("RGB").save(
        buf,
        format="JPEG",
        qtables=qtables,
        subsampling=subsampling,
        optimize=optimize,
        progressive=progressive,
    )
    return buf.getvalue()


def encode_standard(
    img: Image.Image,
    quality: int,
    subsampling: int = 2,
    optimize: bool = True,
    progressive: bool = False,
) -> bytes:
    """Baseline: stock libjpeg encode at the given quality factor."""
    buf = io.BytesIO()
    img.convert("RGB").save(
        buf,
        format="JPEG",
        quality=quality,
        subsampling=subsampling,
        optimize=optimize,
        progressive=progressive,
    )
    return buf.getvalue()


def decode(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def read_tables(data: bytes) -> dict:
    """Read the quantization tables stored in a JPEG file, natural order."""
    q = Image.open(io.BytesIO(data)).quantization
    return {k: _from_pillow_order(v) for k, v in q.items()}


def standard_tables(quality: int):
    """IJG quality scaling of the Annex-K tables (what libjpeg does)."""
    q = min(max(int(quality), 1), 100)
    scale = 5000 // q if q < 50 else 200 - 2 * q

    def _scale(base):
        t = (base * scale + 50) // 100
        return np.clip(t, 1, 255)

    return _scale(STD_LUMA), _scale(STD_CHROMA)
