# Training methodology

This document describes the independently developed 2026 X-JPEG release
model and its evaluation contract.

## Objective

X-JPEG does not replace JPEG. It learns the 8×8 quantization tables stored in
JPEG's DQT segments. The neural network runs only during encoding; the output
remains fully standards-compliant.

For an RGB training crop `x` and predicted tables `T`, the training objective
is:

```text
L = λd (1 - MS-SSIM_RGB(x, JPEGdiff(x, T)))
  + λluma (1 - MS-SSIM_Y(x, JPEGdiff(x, T)))
  + λr R_DCT(x, T)
  + λtable R_table(T)
  + λae MSE(x256, decoder(T))
  + λmse MSE(x, JPEGdiff(x, T))
```

The release continuation used `λd=1`, `λluma=0.25`, and `λr=0.05`. Its final
product phase set `λae=0`; the decoder and entropy model are training-only and
are not distributed in the encoder artifact.

## Encoder

1. Resize the complete image to 256×256 RGB.
2. Center RGB to `[-1, 1]`.
3. Apply four 5×5 stride-2 convolution + reparameterized GDN blocks:
   `256 → 128 → 64 → 32 → 16` with widths `64, 128, 192, 256`.
4. Apply a final 5×5 stride-2 head to produce `3×8×8` logits.
5. Add a learned, frequency-aligned `3×8×8` projection of full-resolution
   DCT-band energy. This restores high-frequency information lost by the
   thumbnail.
6. Map logits to `[1, 255]` with a temperature-scaled sigmoid and round to
   integer DQT entries at inference.

The three channels correspond to Y, Cb, and Cr. The default release encoder
averages the predicted Cb/Cr tables immediately before writing the JPEG,
which avoids the approximately 65-byte third-DQT header cost on small and
low-rate images. `--three-dqt` preserves all three emitted tables.

## Differentiable JPEG

The training path implements RGB↔YCbCr conversion, 4:2:0 chroma sampling,
8×8 DCT, division by the predicted table, quantization, inverse DCT, and RGB
reconstruction. Quantization uses straight-through rounding:

```text
forward:  round(z)
backward: d/dz = 1
```

This is deliberate: a smooth rounding approximation can be exploited when
the quantizer itself is learned.

## Rate terms

`R_DCT` is the per-pixel cross-entropy of quantized DCT symbols under learned
per-frequency Laplace models. It provides a differentiable pressure toward
smaller entropy-coded scans, but it is not reported as file size because it
does not exactly model DC prediction, run-length coding, or optimized Huffman
tables.

`R_table` uses additive uniform noise in `[-0.5, 0.5]` and a factorized prior
over DQT values. It acts as a representation regularizer. JPEG stores each
DQT entry literally, so this soft entropy is not treated as realized header
savings.

## Autoencoder branch

During applicable training phases, a mirror transform upsamples the noisy
`3×8×8` table bottleneck to a 256×256 RGB reconstruction using transposed
convolutions and IGDN. This keeps the bottleneck image-conditioned. The
release continuation reduces this weight after conditioning is established;
only the encoder is needed for deployment.

## Validation contract

Reported release numbers must satisfy all of the following:

- lossless source pixels are provided to every encoder;
- 4:2:0 is used unless a result explicitly says otherwise;
- rate is `8 × complete_file_bytes / (source_width × source_height)`;
- DQT, Huffman, scan, and container bytes are included;
- one dataset-wide table scale is used for research comparisons;
- all JPEG outputs use the same decoder for metric calculation;
- RGB MS-SSIM, luma MS-SSIM, and PSNR are reported separately;
- adaptive tables are compared with fixed-average and wrong-image controls
  before adaptivity is claimed;
- paired bootstrap intervals are used for central comparisons.

User-facing `--target-bpp` is different from the research contract: it
searches a scale independently for each image because its purpose is to meet
the user's requested output size.

## Data

The release model was trained on COCO train2017 with random crops. Release
measurements use held-out COCO val2017 images and Kodak where stated. Dataset
licenses and terms remain those of their respective publishers; no dataset
images are included in the model or package repositories.

