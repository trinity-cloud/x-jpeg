# Release checklist

Public namespaces:

- GitHub: `trinity-cloud/x-jpeg`
- Hugging Face: `migtissera/x-jpeg`
- PyPI: `xjpeg`

## Before publishing

- [ ] Review the README, model card, benchmark scope, and limitations.
- [ ] Confirm `python tools/verify_release.py` passes.
- [ ] Confirm the full test suite passes on Linux and macOS.
- [ ] Expand the 0.25/0.50 audit beyond 100 images before using strong
      frontier language in launch material.
- [ ] Confirm the GitHub repository contains no private research artifacts,
      datasets, credentials, or pickle checkpoints.
- [ ] Confirm the Hugging Face Safetensors checksum is
      `823a305b9cdf6d16e2296644d88a343dfc68621fa8e3d4b3c0125772c50b92e8`.
- [ ] Confirm every native wheel includes MozJPEG's license and IJG notice.
- [ ] Review the MIT license and model-weight release authority.

## Recommended publication order

1. Create and push `trinity-cloud/x-jpeg`; enable branch protection,
   Dependabot, security advisories, and Actions.
2. Create and push `migtissera/x-jpeg` using Git LFS.
3. Create PyPI Trusted Publisher configuration for the package workflow.
4. Run the cross-platform wheel workflow with `publish` left disabled and
   inspect the `manylinux_2_28_x86_64`, macOS x86-64, macOS arm64, and
   Windows x86-64 artifacts.
5. Tag signed `v0.1.0`, approve the protected `pypi` environment, and publish.
6. Create the GitHub release with wheel/sdist checksums and link the model.

## Launch language

Safe claim: X-JPEG improves RGB MS-SSIM over the declared standard-table JPEG
control at matched complete-file bitrate on the documented held-out sample.

Do not claim universal visual superiority over JPEG/WebP. At 0.25 bpp WebP
leads the current evaluation; at both low-rate points it leads luma MS-SSIM
and PSNR.
