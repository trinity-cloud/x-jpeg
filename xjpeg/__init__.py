"""X-JPEG — neural adaptive quantization tables for standard JPEG.

A small CNN predicts per-image 8x8 quantization tables for Y, Cb, and Cr.
The tables are handed to a stock libjpeg encoder, so the output is a
100% standards-compliant JPEG file that any decoder can open.
"""

from .codec import XJPEG

__version__ = "0.1.0"
__all__ = ["XJPEG"]
