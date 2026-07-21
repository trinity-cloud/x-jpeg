# Benchmarks

## Low-rate release audit

The low-rate table in the README uses 100 deterministic native-resolution
COCO val2017 images sampled with seed `20260721`. Images with insufficient
support for five-scale MS-SSIM are excluded before sampling.

Protocol:

- complete emitted file size, not payload-only size;
- conventional full-file bits per source pixel;
- 4:2:0 chroma subsampling;
- progressive coding and optimized Huffman tables;
- pinned MozJPEG commit `08265790774cd0714832c9e675522acbe5581437`;
- MozJPEG float DCT and `tune-ms-ssim` trellis mode;
- Pillow WebP method 6;
- common Pillow/libjpeg decode and identical metric implementation;
- a single dataset-wide scale/quality parameter for each method.

The standard JPEG control scales the ITU-T T.81 Annex-K Y and chroma table
shapes continuously and passes them through the identical MozJPEG pipeline.
This isolates learned table shape from entropy-coder differences.

Raw, sanitized aggregate output is stored in
[`../benchmarks/coco100-lowrate.json`](../benchmarks/coco100-lowrate.json).

## How to reproduce

```bash
python scripts/benchmark_codecs.py xjpeg/weights/model.safetensors \
  --images /path/to/val2017 \
  --limit 100 \
  --seed 20260721 \
  --target-bpp 0.5 \
  --mozjpeg /path/to/pinned/cjpeg \
  --device cuda \
  --bootstrap 5000 \
  --output benchmark-050.json
```

Repeat with `--target-bpp 0.25`. The public script includes additional
controls and may report more rows than the release summary.

## Interpretation

At 0.50 bpp, X-JPEG's default two-DQT deployment has a slightly higher mean
RGB MS-SSIM than WebP, but the difference is only `0.000497`; this is not a
broad superiority claim. At 0.25 bpp, WebP is clearly ahead. X-JPEG trails
both standard JPEG and WebP on luma MS-SSIM at these points while improving
the declared RGB perceptual metric over the standard-table control.

The 100-image audit is sufficient for engineering selection but small for a
durable public frontier claim. A 1,000+ image rerun with direct paired
confidence intervals against every published comparator remains a release
follow-up.

