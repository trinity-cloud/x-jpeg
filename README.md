# X-JPEG

Extending JPEG with neural networks (X-JPEG): Image-adaptive quantization with neural networks for JPEG

X-JPEG analyzes an image, predicts three 8×8 quantization tables for Y, Cb,
and Cr, and passes them to a production JPEG encoder. The output is a normal
JPEG: browsers, operating systems, and existing decoders need no plugin or
neural network.

```text
image ──► compact CNN + full-image DCT statistics ──► adaptive Y/Cb/Cr DQTs
                                                        │
                                                        ▼
                                              MozJPEG ──► .jpg
```

The current implementation and release weights were developed by Migel
Tissera in 2026 and are released by Trinity Cloud under the MIT License.

## Install

```bash
pip install xjpeg
```

Official platform wheels bundle MozJPEG. On an unsupported platform, X-JPEG
automatically uses a verified system MozJPEG (`XJPEG_MOZJPEG=/path/to/cjpeg`)
or falls back to Pillow/libjpeg.

## Compress

```bash
# Complete-file bits per source pixel. This is conventional image-codec bpp.
xjpeg photo.png --target-bpp 0.5

# Search for the smallest file that reaches the requested RGB MS-SSIM.
xjpeg photo.png --target-msssim 0.98

# Directly control the learned tables (>1 means coarser/smaller).
xjpeg photo.png --scale 1.5
```

Progressive JPEG and a shared emitted chroma DQT are the release defaults.
The network still predicts independent Cb and Cr tables. Use `--three-dqt`
to emit all three tables or `--sequential` to disable progressive encoding.

```python
from xjpeg import XJPEG

codec = XJPEG()  # auto-selects bundled/system MozJPEG
result = codec.compress(
    "photo.png",
    output="photo.jpg",
    target_bpp=0.5,
)

print(result.bpp, result.msssim, result.backend)
print(result.luma_table)
```

## Low-rate results

Deterministic sample of 100 held-out native-resolution COCO val2017 images,
seed `20260721`. Rates include the complete emitted file. Every JPEG row used
the same pinned MozJPEG build and settings; WebP used method 6. The release
model predicts three tables and the default deployment averages Cb/Cr only at
the encoding boundary.

| 0.50 bpp target | Actual bpp | RGB MS-SSIM ↑ | Y MS-SSIM ↑ | PSNR ↑ |
|---|---:|---:|---:|---:|
| **X-JPEG, default 2-DQT emission** | 0.49975 | **0.957182** | 0.967685 | 27.045 dB |
| X-JPEG, native 3-DQT emission | 0.49917 | 0.956934 | 0.967261 | 27.015 dB |
| Standard Annex-K tables + MozJPEG | 0.50097 | 0.953630 | 0.972323 | 26.958 dB |
| WebP method 6 | 0.50015 | 0.956685 | **0.972758** | **29.164 dB** |

| 0.25 bpp target | Actual bpp | RGB MS-SSIM ↑ | Y MS-SSIM ↑ | PSNR ↑ |
|---|---:|---:|---:|---:|
| **X-JPEG, default 2-DQT emission** | 0.25111 | 0.917638 | 0.932246 | 24.899 dB |
| X-JPEG, native 3-DQT emission | 0.25002 | 0.916420 | 0.930914 | 24.853 dB |
| Standard Annex-K tables + MozJPEG | 0.25000 | 0.912369 | 0.938567 | 25.009 dB |
| WebP method 6 | 0.24996 | **0.922521** | **0.942144** | **26.609 dB** |

The result is metric-specific. X-JPEG improves RGB MS-SSIM over the declared
standard-table JPEG control at these operating points; WebP retains a clear
advantage at 0.25 bpp and on luma MS-SSIM/PSNR. The 0.50-bpp X-JPEG/WebP mean
difference is too small to claim a general win. See
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for the protocol and limitations.

## Train

```bash
pip install "xjpeg[train]"

python -m xjpeg.train \
  --data /path/to/train2017 \
  --val /path/to/kodak \
  --out runs/experiment \
  --architecture compact \
  --num-tables 3 \
  --dct-stats \
  --gdn-reparam \
  --native-crops \
  --lambda-r 0.05 \
  --lambda-luma 0.25
```

Training combines differentiable JPEG distortion, a learned coefficient-rate
proxy, soft table entropy, and an optional autoencoder reconstruction loss.
Release evaluation always uses real encoded files and complete-file byte
counts. The full objective and validation contract are documented in
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Model and releases

- Source and issues: <https://github.com/trinity-cloud/x-jpeg>
- Model weights: <https://huggingface.co/migtissera/x-jpeg>
- Python package: <https://pypi.org/project/xjpeg/>
- Model card: [`MODEL_CARD.md`](MODEL_CARD.md)

## Limitations

- Trained and evaluated primarily on natural photographs.
- Not validated for medical, scientific, text-heavy, or adversarial imagery.
- Encoding is slower than libjpeg because it adds neural inference and
  MozJPEG trellis optimization; decoding speed is unchanged.
- EXIF/ICC metadata is not preserved in version 0.1.0.
- `--target-bpp` and `--target-msssim` search per image and therefore cost
  multiple real encodes.

## License

X-JPEG code and weights are MIT licensed. Bundled MozJPEG is distributed
under its compatible upstream licenses; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
