"""Benchmark X-JPEG against production JPEG encoders at matched real bitrate.

Every encoder receives the same lossless PPM pixels, emits an ordinary 4:2:0
JPEG, and is scored after decoding through the same Pillow/libjpeg path.  A
single global operating parameter is selected for each method; X-JPEG retains
only its learned per-image table variation plus one dataset-wide table scale.

The script intentionally reports the closest *real* operating point instead
of interpolating or assigning a different quality setting to every image.
"""

import argparse
import io
import json
import random
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image, features
from pytorch_msssim import ms_ssim

from xjpeg.codec import XJPEG
from xjpeg.jpeg_io import (
    STD_CHROMA,
    STD_LUMA,
    decode,
    encode_standard,
    encode_with_tables,
)
from xjpeg.metrics import bpp_of, pil_to_tensor, ycbcr611_msssim


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--images", default="data/kodak")
    p.add_argument("--target-bpp", type=float, default=0.8)
    p.add_argument("--domain-256", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--iters", type=int, default=14)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--mozjpeg",
        type=Path,
        help="path to MozJPEG's cjpeg (enables the MozJPEG baseline)",
    )
    p.add_argument(
        "--jpegli",
        type=Path,
        help="path to jpegli's cjpegli (enables the jpegli baseline)",
    )
    p.add_argument(
        "--sequential",
        action="store_true",
        help="disable progressive coding for all encoders",
    )
    p.add_argument(
        "--hybrid-controls",
        action="store_true",
        help="also test fixed-average and wrong-image tables through MozJPEG",
    )
    return p.parse_args()


def _luma(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def _run(command):
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"encoder failed ({result.returncode}): {command}\n{stderr}")


@torch.no_grad()
def main():
    args = parse_args()
    if args.mozjpeg and not args.mozjpeg.is_file():
        raise SystemExit(f"MozJPEG encoder not found: {args.mozjpeg}")
    if args.jpegli and not args.jpegli.is_file():
        raise SystemExit(f"jpegli encoder not found: {args.jpegli}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    paths = sorted(
        path for path in Path(args.images).iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise SystemExit(f"no images found under {args.images}")
    excluded_small = 0
    if not args.domain_256:
        eligible = []
        for path in paths:
            with Image.open(path) as probe:
                if min(probe.size) > 160:
                    eligible.append(path)
                else:
                    excluded_small += 1
        paths = eligible
        if not paths:
            raise SystemExit("no native images satisfy five-scale MS-SSIM min-side > 160")
    if args.limit and args.limit < len(paths):
        paths = sorted(random.Random(args.seed).sample(paths, args.limit))

    images = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        if args.domain_256:
            image = image.resize((256, 256), Image.Resampling.BICUBIC)
        images.append((path.name, image.copy()))

    model = XJPEG(args.checkpoint, device=str(device))
    predicted = [model.predict_tables(image) for _, image in images]
    fixed_tables = tuple(
        np.mean([tables[i] for tables in predicted], axis=0)
        for i in range(len(predicted[0]))
    )
    fixed = [fixed_tables] * len(predicted)
    shuffled = predicted[1:] + predicted[:1]
    stacked = np.stack([np.stack(tables) for tables in predicted])
    relative_std = stacked.std(axis=0) / np.maximum(stacked.mean(axis=0), 1.0)
    progressive = not args.sequential

    print(
        f"n={len(images)} target={args.target_bpp:.3f} bpp "
        f"domain={'256' if args.domain_256 else 'native'} "
        f"progressive={progressive} excluded-small={excluded_small}",
        flush=True,
    )
    print(
        f"X-JPEG per-image table variability: "
        f"{100 * relative_std.mean():.3f}% mean relative std",
        flush=True,
    )

    def mean_bpp(datas):
        return float(np.mean([
            bpp_of(data, image)
            for data, (_, image) in zip(datas, images)
        ]))

    def threaded(function):
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            return list(pool.map(function, range(len(images))))

    with tempfile.TemporaryDirectory(prefix="xjpeg-benchmark-") as temp_name:
        temp = Path(temp_name)
        ppm_paths = []
        for index, (_, image) in enumerate(images):
            path = temp / f"{index:06d}.ppm"
            image.save(path, format="PPM")
            ppm_paths.append(path)

        def xjpeg_encode(scale, share_chroma=False):
            def one(i):
                source_tables = predicted[i]
                if share_chroma and len(source_tables) == 3:
                    source_tables = (
                        source_tables[0],
                        (np.asarray(source_tables[1])
                         + np.asarray(source_tables[2])) / 2.0,
                    )
                return encode_with_tables(
                    images[i][1],
                    *(np.asarray(table) * scale for table in source_tables),
                    progressive=progressive,
                )

            return threaded(one)

        def turbo_encode(quality):
            return threaded(lambda i: encode_standard(
                images[i][1],
                quality=int(quality),
                progressive=progressive,
            ))

        def annex_k_encode(scale):
            """Ordinary JPEG using globally scaled Annex-K tables."""

            return threaded(lambda i: encode_with_tables(
                images[i][1],
                np.asarray(STD_LUMA) * scale,
                np.asarray(STD_CHROMA) * scale,
                progressive=progressive,
            ))

        def webp_encode(quality):
            def one(i):
                output = io.BytesIO()
                images[i][1].save(
                    output,
                    format="WEBP",
                    quality=float(quality),
                    method=6,
                )
                return output.getvalue()

            return threaded(one)

        def mozjpeg_encode(quality):
            def one(i):
                output = temp / f"moz-{i:06d}.jpg"
                command = [
                    str(args.mozjpeg),
                    "-quality", str(int(quality)),
                    "-sample", "2x2,1x1,1x1",
                    "-optimize",
                    "-quant-table", "2",
                    "-tune-ms-ssim",
                    "-outfile", str(output),
                ]
                command.append("-baseline" if args.sequential else "-progressive")
                command.append(str(ppm_paths[i]))
                _run(command)
                return output.read_bytes()

            return threaded(one)

        def xjpeg_mozjpeg_encode(
            scale, tables_for_images=predicted, share_chroma=False,
        ):
            """Learned per-image tables plus MozJPEG's entropy machinery."""

            def one(i):
                output = temp / f"xmoz-{i:06d}.jpg"
                qtables = temp / f"xmoz-{i:06d}.qtables"
                source_tables = tables_for_images[i]
                if share_chroma and len(source_tables) == 3:
                    source_tables = (
                        source_tables[0],
                        (np.asarray(source_tables[1])
                         + np.asarray(source_tables[2])) / 2.0,
                    )
                scaled = [
                    np.clip(np.rint(np.asarray(table) * scale), 1, 255)
                    .astype(np.int64)
                    for table in source_tables
                ]
                qtables.write_text(
                    "\n\n".join(
                        "\n".join(
                            " ".join(str(int(value)) for value in row)
                            for row in table
                        )
                        for table in scaled
                    )
                    + "\n"
                )
                command = [
                    str(args.mozjpeg),
                    "-qtables", str(qtables),
                    "-qslots", "0,1,1" if len(scaled) == 2 else "0,1,2",
                    "-sample", "2x2,1x1,1x1",
                    "-optimize",
                    "-tune-ms-ssim",
                    "-outfile", str(output),
                ]
                command.append("-baseline" if args.sequential else "-progressive")
                command.append(str(ppm_paths[i]))
                _run(command)
                return output.read_bytes()

            return threaded(one)

        def jpegli_encode(distance):
            def one(i):
                output = temp / f"jpegli-{i:06d}.jpg"
                command = [
                    str(args.jpegli),
                    str(ppm_paths[i]),
                    str(output),
                    f"--distance={distance:.8f}",
                    "--chroma_subsampling=420",
                    f"--progressive_level={0 if args.sequential else 2}",
                ]
                _run(command)
                return output.read_bytes()

            return threaded(one)

        def continuous_search(encode, lo, hi, rate_decreases):
            cache = {}

            def evaluate(parameter):
                key = round(float(parameter), 10)
                if key not in cache:
                    datas = encode(parameter)
                    cache[key] = (mean_bpp(datas), datas)
                return cache[key]

            lo_bpp, _ = evaluate(lo)
            hi_bpp, _ = evaluate(hi)
            bracketed = (
                lo_bpp >= args.target_bpp >= hi_bpp
                if rate_decreases
                else lo_bpp <= args.target_bpp <= hi_bpp
            )
            if not bracketed:
                raise RuntimeError(
                    f"target {args.target_bpp} not bracketed by "
                    f"{lo_bpp:.4f} and {hi_bpp:.4f} bpp"
                )
            for _ in range(args.iters):
                mid = (lo + hi) / 2.0
                bpp, _ = evaluate(mid)
                too_large = bpp > args.target_bpp
                if too_large == rate_decreases:
                    lo = mid
                else:
                    hi = mid
            candidates = [
                (abs(bpp - args.target_bpp), parameter, bpp, datas)
                for parameter, (bpp, datas) in cache.items()
            ]
            return min(candidates, key=lambda row: row[0])

        def integer_quality_search(encode):
            cache = {}
            lo, hi = 1, 100
            while lo <= hi:
                quality = (lo + hi) // 2
                datas = encode(quality)
                bpp = mean_bpp(datas)
                cache[quality] = (bpp, datas)
                if bpp < args.target_bpp:
                    lo = quality + 1
                else:
                    hi = quality - 1
            for quality in {max(1, hi), min(100, lo)} - set(cache):
                datas = encode(quality)
                cache[quality] = (mean_bpp(datas), datas)
            candidates = [
                (abs(bpp - args.target_bpp), quality, bpp, datas)
                for quality, (bpp, datas) in cache.items()
            ]
            return min(candidates, key=lambda row: row[0])

        searches = {
            "xjpeg-adaptive": lambda: continuous_search(
                xjpeg_encode, 0.02, 16.0, rate_decreases=True
            ),
            "xjpeg-2dqt": lambda: continuous_search(
                lambda scale: xjpeg_encode(scale, share_chroma=True),
                0.02, 16.0, rate_decreases=True,
            ),
            "libjpeg-turbo": lambda: integer_quality_search(turbo_encode),
            "jpeg-annex-k": lambda: continuous_search(
                annex_k_encode, 0.02, 16.0, rate_decreases=True
            ),
            "webp-method6": lambda: continuous_search(
                webp_encode, 1.0, 100.0, rate_decreases=False
            ),
        }
        if args.mozjpeg:
            searches["xjpeg+mozjpeg"] = lambda: continuous_search(
                xjpeg_mozjpeg_encode, 0.02, 16.0, rate_decreases=True
            )
            searches["xjpeg+mozjpeg-2dqt"] = lambda: continuous_search(
                lambda scale: xjpeg_mozjpeg_encode(scale, share_chroma=True),
                0.02, 16.0, rate_decreases=True,
            )
            if args.hybrid_controls:
                searches["xjpeg+moz-fixed"] = lambda: continuous_search(
                    lambda scale: xjpeg_mozjpeg_encode(
                        scale, tables_for_images=fixed
                    ),
                    0.02, 16.0, rate_decreases=True,
                )
                searches["xjpeg+moz-shuffled"] = lambda: continuous_search(
                    lambda scale: xjpeg_mozjpeg_encode(
                        scale, tables_for_images=shuffled
                    ),
                    0.02, 16.0, rate_decreases=True,
                )
            searches["mozjpeg-ms-ssim"] = lambda: integer_quality_search(
                mozjpeg_encode
            )
        if args.jpegli:
            searches["jpegli"] = lambda: continuous_search(
                jpegli_encode, 0.01, 25.0, rate_decreases=True
            )

        same_shape = len({image.size for _, image in images}) == 1

        def measure(datas):
            if not same_shape:
                rows = []
                for data, (_, image) in zip(datas, images):
                    ref = pil_to_tensor(image).to(device)
                    rec = pil_to_tensor(decode(data)).to(device)
                    mse = (ref - rec).square().mean().clamp_min(1e-12)
                    rows.append((
                        ms_ssim(ref, rec, data_range=1.0).item(),
                        ms_ssim(_luma(ref), _luma(rec), data_range=1.0).item(),
                        ycbcr611_msssim(ref, rec).item(),
                        (10.0 * torch.log10(1.0 / mse)).item(),
                    ))
                return np.asarray(rows)

            rows = []
            for start in range(0, len(images), args.batch_size):
                stop = min(start + args.batch_size, len(images))
                refs = torch.cat([
                    pil_to_tensor(image) for _, image in images[start:stop]
                ]).to(device)
                recs = torch.cat([
                    pil_to_tensor(decode(data)) for data in datas[start:stop]
                ]).to(device)
                rgb_ms = ms_ssim(
                    refs, recs, data_range=1.0, size_average=False
                )
                y_ms = ms_ssim(
                    _luma(refs), _luma(recs), data_range=1.0,
                    size_average=False,
                )
                ycc611_ms = ycbcr611_msssim(
                    refs, recs, size_average=False
                )
                mse = (refs - recs).square().mean(dim=(1, 2, 3))
                psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
                rows.extend(
                    torch.stack(
                        [rgb_ms, y_ms, ycc611_ms, psnr], dim=1
                    ).cpu().numpy()
                )
            return np.asarray(rows)

        results = {}
        metric_rows = {}
        for name, search in searches.items():
            _, parameter, bpp, datas = search()
            metrics = measure(datas)
            metric_rows[name] = metrics
            results[name] = {
                "parameter": float(parameter),
                "bpp": bpp,
                "rgb_ms_ssim": float(metrics[:, 0].mean()),
                "y_ms_ssim": float(metrics[:, 1].mean()),
                "ycbcr_611_ms_ssim": float(metrics[:, 2].mean()),
                "psnr_db": float(metrics[:, 3].mean()),
            }
            print(
                f"{name:>17}: {bpp:.4f} bpp  param {parameter:.5g}  "
                f"RGB-MS {metrics[:, 0].mean():.5f}  "
                f"Y-MS {metrics[:, 1].mean():.5f}  "
                f"611-MS {metrics[:, 2].mean():.5f}  "
                f"PSNR {metrics[:, 3].mean():.2f} dB",
                flush=True,
            )

        if args.bootstrap:
            adaptive = metric_rows["xjpeg-adaptive"][:, 0]
            rng = np.random.default_rng(args.seed)
            print("paired RGB-MS deltas (X-JPEG - codec):", flush=True)
            for name, values in metric_rows.items():
                if name == "xjpeg-adaptive":
                    continue
                rate_gap = (
                    results["xjpeg-adaptive"]["bpp"] - results[name]["bpp"]
                )
                if abs(rate_gap) > 0.01:
                    print(
                        f"  vs {name:>15}: skipped (closest real operating "
                        f"point has mean-bpp gap {rate_gap:+.5f})",
                        flush=True,
                    )
                    continue
                delta = adaptive - values[:, 0]
                indices = rng.integers(
                    0, len(delta), size=(args.bootstrap, len(delta))
                )
                means = delta[indices].mean(axis=1)
                lo, hi = np.percentile(means, [2.5, 97.5])
                print(
                    f"  vs {name:>15}: {delta.mean():+.6f} "
                    f"(95% CI {lo:+.6f}..{hi:+.6f}); "
                    f"mean-bpp gap {rate_gap:+.5f}",
                    flush=True,
                )

            two_dqt = metric_rows["xjpeg-2dqt"][:, 0]
            print("paired RGB-MS deltas (X-JPEG 2-DQT - codec):", flush=True)
            for name in (
                "libjpeg-turbo", "jpeg-annex-k", "webp-method6",
                "mozjpeg-ms-ssim", "jpegli"
            ):
                if name not in metric_rows:
                    continue
                rate_gap = results["xjpeg-2dqt"]["bpp"] - results[name]["bpp"]
                if abs(rate_gap) > 0.01:
                    print(
                        f"  vs {name:>15}: skipped (closest real operating "
                        f"point has mean-bpp gap {rate_gap:+.5f})",
                        flush=True,
                    )
                    continue
                delta = two_dqt - metric_rows[name][:, 0]
                indices = rng.integers(
                    0, len(delta), size=(args.bootstrap, len(delta))
                )
                means = delta[indices].mean(axis=1)
                lo, hi = np.percentile(means, [2.5, 97.5])
                print(
                    f"  vs {name:>15}: {delta.mean():+.6f} "
                    f"(95% CI {lo:+.6f}..{hi:+.6f}); "
                    f"mean-bpp gap {rate_gap:+.5f}",
                    flush=True,
                )

            if args.hybrid_controls:
                hybrid = metric_rows["xjpeg+mozjpeg"][:, 0]
                print(
                    "paired hybrid RGB-MS deltas "
                    "(adaptive tables - control tables):",
                    flush=True,
                )
                for name in ("xjpeg+moz-fixed", "xjpeg+moz-shuffled"):
                    delta = hybrid - metric_rows[name][:, 0]
                    indices = rng.integers(
                        0, len(delta), size=(args.bootstrap, len(delta))
                    )
                    means = delta[indices].mean(axis=1)
                    lo, hi = np.percentile(means, [2.5, 97.5])
                    rate_gap = (
                        results["xjpeg+mozjpeg"]["bpp"]
                        - results[name]["bpp"]
                    )
                    print(
                        f"  vs {name:>18}: {delta.mean():+.6f} "
                        f"(95% CI {lo:+.6f}..{hi:+.6f}); "
                        f"mean-bpp gap {rate_gap:+.5f}",
                        flush=True,
                    )

    report = {
        "checkpoint": str(args.checkpoint),
        "images": str(args.images),
        "n": len(images),
        "seed": args.seed,
        "domain_256": args.domain_256,
        "target_bpp": args.target_bpp,
        "excluded_small": excluded_small,
        "progressive": progressive,
        "pillow_jpeg_version": features.version_codec("jpg"),
        "pillow_libjpeg_turbo": features.check_feature("libjpeg_turbo"),
        "pillow_webp_version": features.version("webp"),
        "xjpeg_table_relative_std": float(relative_std.mean()),
        "mozjpeg": str(args.mozjpeg) if args.mozjpeg else None,
        "jpegli": str(args.jpegli) if args.jpegli else None,
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
