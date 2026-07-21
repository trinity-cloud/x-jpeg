# X-JPEG 0.1 encoder model card

## Summary

The X-JPEG encoder predicts three image-dependent 8×8 JPEG quantization
tables for Y, Cb, and Cr. It combines a compact 256×256 GDN analysis transform
with full-resolution DCT-band statistics. The tables are passed to MozJPEG or
Pillow/libjpeg; decoding requires only an ordinary JPEG decoder.

- Developer: Migel Tissera / Trinity Cloud
- Version: 0.1.0
- Artifact: `xjpeg/weights/model.safetensors`
- Safetensors SHA-256: `823a305b9cdf6d16e2296644d88a343dfc68621fa8e3d4b3c0125772c50b92e8`
- Source checkpoint SHA-256: `5878df379ff23dbfa1c1e753d0a70d2b131e4625cc20be2b61fc882789304219`
- License: MIT
- Parameters distributed: encoder only
- Training data: COCO train2017

## Intended use

- Compress natural photographic images into standards-compliant JPEG files.
- Predict image-specific DQTs for codec research.
- Use the two-emitted-DQT default for small or low-rate files; use
  `--three-dqt` when separate Cb/Cr tables are required.

This is a lossy codec and is not intended for archival, forensic, medical, or
scientific preservation.

## Training

The release model was optimized through differentiable 4:2:0 JPEG using RGB
MS-SSIM, a 0.25-weight luma MS-SSIM guard, and a learned coefficient-rate
proxy with weight 0.05. A soft factorized entropy term regularized the table
bottleneck during applicable phases. The product continuation used
full-image DCT-band conditioning and did not require the decoder loss.

See `docs/METHODOLOGY.md` for the objective and validation contract.

## Evaluation

All rates are complete-file bits per source pixel. On a deterministic sample
of 100 native-resolution COCO val2017 images:

| Method | bpp | RGB MS-SSIM | Y MS-SSIM | PSNR |
|---|---:|---:|---:|---:|
| X-JPEG 2-DQT | 0.49975 | 0.957182 | 0.967685 | 27.045 dB |
| Standard Annex-K + same MozJPEG | 0.50097 | 0.953630 | 0.972323 | 26.958 dB |
| WebP method 6 | 0.50015 | 0.956685 | 0.972758 | 29.164 dB |
| X-JPEG 2-DQT | 0.25111 | 0.917638 | 0.932246 | 24.899 dB |
| Standard Annex-K + same MozJPEG | 0.25000 | 0.912369 | 0.938567 | 25.009 dB |
| WebP method 6 | 0.24996 | 0.922521 | 0.942144 | 26.609 dB |

The comparison supports an RGB MS-SSIM improvement over the declared
standard-table JPEG control. It does not support a general claim of visual
superiority over WebP, especially at 0.25 bpp or under luma/PSNR metrics.

## Limitations

- Selected for natural photographs; other image domains are not established.
- Quality varies with image content and target rate.
- The DCT entropy model is a training proxy, so all claims must be remeasured
  with real files.
- The 100-image low-rate benchmark should be expanded before strong public
  frontier claims are made.
- The package does not preserve EXIF or ICC metadata in version 0.1.0.
- Safetensors prevents pickle execution but does not make an untrusted model
  semantically safe; verify the published checksum.

## Environmental considerations

The released artifact contains only the encoder. JPEG decoding uses existing
software and no neural model. Training is substantially more expensive than
inference; the model card does not estimate training energy because reliable
instrumentation was not recorded for the final continuation.

