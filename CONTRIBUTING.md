# Contributing

Thank you for improving X-JPEG.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test,train]"
pytest -q
```

Tests that exercise MozJPEG use `XJPEG_MOZJPEG=/path/to/cjpeg` when a bundled
binary is not present. Pillow fallback tests must also pass.

## Pull requests

- Add tests for behavioral changes.
- Report complete-file bpp rather than scan payload size.
- Keep RGB, luma, and PSNR results distinct.
- Use held-out images and paired confidence intervals for performance claims.
- Record codec versions, settings, sampling, image selection, and seed.
- Do not add datasets, private checkpoints, credentials, or generated images.

Changes to the model architecture, preprocessing, DQT ordering, or MozJPEG
flags can silently invalidate a checkpoint or benchmark. Include a prediction
parity test or a new model version when changing these contracts.

By contributing, you agree that your contribution is licensed under the MIT
License.

