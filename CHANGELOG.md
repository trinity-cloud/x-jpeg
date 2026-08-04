# Changelog

## 0.1.1 — 2026-08-04

- Adopt the public positioning: “Extending JPEG with neural networks
  (X-JPEG): Image-adaptive quantization with neural networks for JPEG.”
- Metadata-only release; model weights and codec behavior are unchanged.

## 0.1.0 — 2026-08-04

- First public release.
- Image-adaptive Y/Cb/Cr quantization-table prediction.
- Full-resolution DCT-band conditioning.
- Safetensors encoder artifact with verified prediction parity.
- Automatic bundled/system MozJPEG backend with Pillow fallback.
- Conventional complete-file `--target-bpp` mode.
- Progressive, shared-emitted-chroma release defaults.
- Training, benchmarking, model-card, and third-party license documentation.
