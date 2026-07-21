"""User-facing X-JPEG codec.

The neural network runs only while encoding. It predicts image-conditioned
JPEG quantization tables, which are then passed to MozJPEG or Pillow. The
result is a standards-compliant JPEG that requires no custom decoder.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import warnings

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as load_safetensors

from .jpeg_io import decode, encode_standard, encode_with_tables
from .metrics import bpp_of, msssim, pil_to_tensor, psnr
from .model import Encoder, code_to_table
from .mozjpeg import MozJPEGNotFoundError, encode_mozjpeg, find_mozjpeg
from .stats import dct_band_stats

ImageLike = Union[str, Path, Image.Image]

_ENV_CHECKPOINT = "XJPEG_CHECKPOINT"
_PACKAGED_WEIGHTS = Path(__file__).parent / "weights" / "model.safetensors"
_BACKENDS = {"auto", "mozjpeg", "pillow"}


@dataclass
class CompressResult:
    data: bytes
    bpp: float
    msssim: float
    psnr: float
    luma_table: np.ndarray
    cb_table: np.ndarray
    cr_table: np.ndarray
    table_scale: float
    backend: str

    @property
    def size_bytes(self) -> int:
        return len(self.data)

    @property
    def chroma_table(self) -> np.ndarray:
        """Compatibility alias for callers using one shared chroma table."""
        return self.cb_table


def _load_image(image: ImageLike) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def _load_encoder_artifact(path: Path, device: torch.device):
    if path.suffix == ".safetensors":
        config_path = path.with_name("config.json")
        if not config_path.is_file():
            raise FileNotFoundError(
                f"model configuration not found next to weights: {config_path}"
            )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        state = load_safetensors(str(path), device=str(device))
        return state, config

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        return checkpoint, {}
    config = {
        key: checkpoint[key]
        for key in ("architecture", "dct_stats", "gdn_reparam")
        if key in checkpoint
    }
    if isinstance(checkpoint.get("args"), dict):
        config.update({
            key: checkpoint["args"][key]
            for key in ("architecture", "dct_stats", "gdn_reparam")
            if key in checkpoint["args"]
        })
    return checkpoint.get("model", checkpoint), config


class XJPEG:
    """Predict adaptive DQTs and encode ordinary JPEG files.

    ``backend='auto'`` prefers MozJPEG and falls back to Pillow/libjpeg.
    Published platform wheels bundle MozJPEG; source installs can set
    ``XJPEG_MOZJPEG`` or pass ``mozjpeg_path``.
    """

    def __init__(
        self,
        checkpoint: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        backend: str = "auto",
        mozjpeg_path: Optional[Union[str, Path]] = None,
    ):
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {sorted(_BACKENDS)}")
        if checkpoint is None:
            checkpoint = os.environ.get(_ENV_CHECKPOINT) or (
                _PACKAGED_WEIGHTS if _PACKAGED_WEIGHTS.exists() else None
            )
        if checkpoint is None:
            raise FileNotFoundError(
                "No model was found. Install an official wheel, pass "
                "checkpoint=..., or set XJPEG_CHECKPOINT."
            )

        self.device = torch.device(device)
        self.backend_preference = backend
        self.mozjpeg_path = Path(mozjpeg_path) if mozjpeg_path else None
        state, config = _load_encoder_artifact(Path(checkpoint), self.device)
        if any(key.startswith("encoder.") for key in state):
            state = {
                key.removeprefix("encoder."): value
                for key, value in state.items()
                if key.startswith("encoder.")
            }

        dct_stats = bool(config.get("dct_stats", False))
        gdn_reparam = bool(config.get("gdn_reparam", False))
        architecture = config.get("architecture", "deep")
        try:
            num_tables = int(state["head.weight"].shape[0])
        except (KeyError, AttributeError) as exc:
            raise ValueError("checkpoint does not contain an encoder head") from exc

        if architecture == "compact":
            widths = tuple(
                int(state[f"blocks.{index}.conv.weight"].shape[0])
                for index in range(4)
            ) + (int(state["head.weight"].shape[1]),)
        elif architecture == "deep":
            widths = tuple(
                int(state[f"blocks.{index}.down.weight"].shape[0])
                for index in range(5)
            )
        else:
            raise ValueError(f"unsupported encoder architecture: {architecture}")

        self.encoder = Encoder(
            widths=widths,
            dct_stats=dct_stats,
            num_tables=num_tables,
            gdn_reparam=gdn_reparam,
            architecture=architecture,
        ).to(self.device).eval()
        self.encoder.load_state_dict(state)

    @torch.no_grad()
    def predict_tables(self, image: ImageLike):
        """Return integer 8x8 tables in component order: Y, Cb, and Cr."""
        image = _load_image(image)
        thumbnail = image.resize((256, 256), Image.Resampling.BICUBIC)
        tensor = pil_to_tensor(thumbnail).to(self.device)
        stats = None
        if self.encoder.dct_stats:
            stats = dct_band_stats(
                pil_to_tensor(image).to(self.device), self.encoder.num_tables
            )
        tables = code_to_table(self.encoder(tensor, stats)).round().clamp(1, 255)
        tables = tables[0].cpu().numpy().astype(np.int64)
        return tuple(tables[index] for index in range(tables.shape[0]))

    def _resolve_backend(self, requested: str | None) -> str:
        backend = requested or self.backend_preference
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {sorted(_BACKENDS)}")
        if backend == "pillow":
            return backend
        if find_mozjpeg(self.mozjpeg_path) is not None:
            return "mozjpeg"
        if backend == "mozjpeg":
            raise MozJPEGNotFoundError(
                "MozJPEG was explicitly requested but no verified binary was found"
            )
        warnings.warn(
            "MozJPEG is unavailable; falling back to Pillow/libjpeg. "
            "Install an official platform wheel or set XJPEG_MOZJPEG.",
            RuntimeWarning,
            stacklevel=3,
        )
        return "pillow"

    def _encode(
        self, backend, image, luma, cb, cr, *, subsampling, optimize,
        progressive,
    ) -> bytes:
        if backend == "mozjpeg":
            return encode_mozjpeg(
                image, luma, cb, cr,
                subsampling=subsampling,
                optimize=optimize,
                progressive=progressive,
                executable=self.mozjpeg_path,
            )
        return encode_with_tables(
            image, luma, cb, cr,
            subsampling=subsampling,
            optimize=optimize,
            progressive=progressive,
        )

    def compress(
        self,
        image: ImageLike,
        output: Optional[Union[str, Path]] = None,
        table_scale: float = 1.0,
        target_bpp: Optional[float] = None,
        target_msssim: Optional[float] = None,
        subsampling: int = 2,
        optimize: bool = True,
        progressive: bool = True,
        share_chroma: bool = True,
        backend: Optional[str] = None,
    ) -> CompressResult:
        """Compress an image to a standard JPEG.

        ``target_bpp`` searches real complete-file size. ``target_msssim``
        searches for the smallest file meeting the requested RGB MS-SSIM.
        They are mutually exclusive and override ``table_scale``.
        """
        if target_bpp is not None and target_msssim is not None:
            raise ValueError("target_bpp and target_msssim are mutually exclusive")
        if target_bpp is not None and target_bpp <= 0:
            raise ValueError("target_bpp must be positive")
        image = _load_image(image)
        selected_backend = self._resolve_backend(backend)
        predicted = self.predict_tables(image)
        luma, cb = predicted[:2]
        emitted_cr = predicted[2] if len(predicted) == 3 else None
        if share_chroma and emitted_cr is not None:
            cb = (cb.astype(np.float64) + emitted_cr) / 2.0
            emitted_cr = None
        cr_for_result = emitted_cr if emitted_cr is not None else cb

        encode_kwargs = {
            "subsampling": subsampling,
            "optimize": optimize,
            "progressive": progressive,
        }

        def encode_at(scale: float) -> bytes:
            return self._encode(
                selected_backend,
                image,
                luma * scale,
                cb * scale,
                None if emitted_cr is None else emitted_cr * scale,
                **encode_kwargs,
            )

        scale = float(table_scale)
        if target_bpp is not None:
            scale = self._search_scale_for_bpp(image, encode_at, target_bpp)
        elif target_msssim is not None:
            scale = self._search_scale_for_msssim(
                image, encode_at, target_msssim
            )

        data = encode_at(scale)
        reference = pil_to_tensor(image)
        reconstruction = pil_to_tensor(decode(data))
        result = CompressResult(
            data=data,
            bpp=bpp_of(data, image),
            msssim=msssim(reference, reconstruction),
            psnr=psnr(reference, reconstruction),
            luma_table=np.clip(np.rint(luma * scale), 1, 255).astype(np.int64),
            cb_table=np.clip(np.rint(cb * scale), 1, 255).astype(np.int64),
            cr_table=np.clip(np.rint(cr_for_result * scale), 1, 255).astype(np.int64),
            table_scale=scale,
            backend=selected_backend,
        )
        if output is not None:
            Path(output).write_bytes(data)
        return result

    @staticmethod
    def _search_scale_for_bpp(
        image: Image.Image,
        encode,
        target: float,
        lo: float = 0.02,
        hi: float = 32.0,
        iters: int = 14,
    ) -> float:
        cache = {}

        def rate(scale):
            key = round(float(scale), 10)
            if key not in cache:
                cache[key] = bpp_of(encode(scale), image)
            return cache[key]

        if rate(lo) <= target:
            return lo
        if rate(hi) >= target:
            return hi
        lower, upper = lo, hi
        for _ in range(iters):
            middle = (lower + upper) / 2.0
            if rate(middle) > target:
                lower = middle
            else:
                upper = middle
        return min(cache, key=lambda scale: abs(cache[scale] - target))

    @staticmethod
    def _search_scale_for_msssim(
        image: Image.Image,
        encode,
        target: float,
        lo: float = 0.02,
        hi: float = 32.0,
        iters: int = 12,
    ) -> float:
        reference = pil_to_tensor(image)

        def quality(scale: float) -> float:
            reconstruction = pil_to_tensor(decode(encode(scale)))
            return msssim(reference, reconstruction)

        if quality(hi) >= target:
            return hi
        if quality(lo) < target:
            return lo
        feasible, infeasible = lo, hi
        for _ in range(iters):
            middle = (feasible + infeasible) / 2.0
            if quality(middle) >= target:
                feasible = middle
            else:
                infeasible = middle
        return feasible

    def compare_with_baseline(self, image: ImageLike, **compress_kwargs) -> dict:
        """Compare against Pillow/libjpeg at matched RGB MS-SSIM."""
        image = _load_image(image)
        ours = self.compress(image, **compress_kwargs)
        quality, data, score = _baseline_matching_msssim(
            image,
            ours.msssim,
            subsampling=compress_kwargs.get("subsampling", 2),
            optimize=compress_kwargs.get("optimize", True),
            progressive=compress_kwargs.get("progressive", True),
        )
        return {
            "xjpeg": ours,
            "baseline_quality": quality,
            "baseline_bytes": len(data),
            "baseline_bpp": bpp_of(data, image),
            "baseline_msssim": score,
            "savings_pct": 100.0 * (1.0 - len(ours.data) / len(data)),
        }


def _baseline_matching_msssim(
    image: Image.Image, target: float, **kwargs
) -> tuple:
    reference = pil_to_tensor(image)
    lower, upper = 1, 95
    best = None
    while lower <= upper:
        quality = (lower + upper) // 2
        data = encode_standard(image, quality, **kwargs)
        score = msssim(reference, pil_to_tensor(decode(data)))
        if score >= target:
            best = (quality, data, score)
            upper = quality - 1
        else:
            lower = quality + 1
    if best is None:
        data = encode_standard(image, 95, **kwargs)
        best = (
            95,
            data,
            msssim(reference, pil_to_tensor(decode(data))),
        )
    return best
