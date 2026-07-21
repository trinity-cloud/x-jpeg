#!/usr/bin/env python3
"""Measure an oracle per-image rate allocation over sampled X-JPEG scales.

This is a decision experiment, not a deployable codec mode: it uses reference
images to choose each image's operating point.  The reported fractional value
is an upper bound only for the convex relaxation of the sampled table-scale
points.  It is deliberately *not* described as an upper bound for JPEG.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_msssim import ms_ssim

from xjpeg.codec import XJPEG
from xjpeg.jpeg_io import decode, encode_with_tables
from xjpeg.metrics import pil_to_tensor, ycbcr611_msssim
from xjpeg.oracle import allocate_sampled_rd


METRIC_NAMES = ("rgb_msssim", "y_msssim", "ycbcr611_msssim", "psnr")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--images", default="data/kodak")
    parser.add_argument("--target-bpp", type=float, default=0.8)
    parser.add_argument("--objective", choices=METRIC_NAMES, default="rgb_msssim")
    parser.add_argument("--domain-256", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-scales", type=int, default=32)
    parser.add_argument("--min-scale", type=float, default=0.15)
    parser.add_argument("--max-scale", type=float, default=12.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _luma(value):
    return 0.299 * value[:, 0:1] + 0.587 * value[:, 1:2] + 0.114 * value[:, 2:3]


def _load_images(args):
    paths = sorted(
        path for path in Path(args.images).iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
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
    if args.limit and args.limit < len(paths):
        paths = sorted(random.Random(args.seed).sample(paths, args.limit))
    images = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
            if args.domain_256:
                image = image.resize((256, 256), Image.Resampling.BICUBIC)
            images.append((path.name, image.copy()))
    if not images:
        raise SystemExit("no eligible images found")
    return images, excluded_small


def _measure_same_shape(images, datas, device, batch_size, workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        reconstructions = list(pool.map(decode, datas))
    rows = []
    for start in range(0, len(images), batch_size):
        stop = min(start + batch_size, len(images))
        refs = torch.cat([pil_to_tensor(image) for _, image in images[start:stop]]).to(device)
        recs = torch.cat([pil_to_tensor(image) for image in reconstructions[start:stop]]).to(device)
        rgb = ms_ssim(refs, recs, data_range=1.0, size_average=False)
        y = ms_ssim(_luma(refs), _luma(recs), data_range=1.0, size_average=False)
        ycc611 = ycbcr611_msssim(refs, recs, size_average=False)
        mse = (refs - recs).square().mean(dim=(1, 2, 3))
        psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
        rows.append(torch.stack((rgb, y, ycc611, psnr), dim=1).cpu().numpy())
    return np.concatenate(rows, axis=0)


def _measure_varied_shape(images, datas, device, workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        reconstructions = list(pool.map(decode, datas))
    rows = []
    for (_, image), reconstruction in zip(images, reconstructions):
        ref = pil_to_tensor(image).to(device)
        rec = pil_to_tensor(reconstruction).to(device)
        rgb = ms_ssim(ref, rec, data_range=1.0).item()
        y = ms_ssim(_luma(ref), _luma(rec), data_range=1.0).item()
        ycc611 = ycbcr611_msssim(ref, rec).item()
        mse = (ref - rec).square().mean().item()
        psnr = 10.0 * np.log10(1.0 / max(mse, 1e-12))
        rows.append((rgb, y, ycc611, psnr))
    return np.asarray(rows)


def _summarize_selection(rates, metrics, selection):
    rows = np.arange(len(selection))
    selected_metrics = metrics[rows, selection]
    return {
        "mean_bpp": float(rates[rows, selection].mean()),
        **{
            name: float(selected_metrics[:, index].mean())
            for index, name in enumerate(METRIC_NAMES)
        },
    }


@torch.no_grad()
def main():
    args = parse_args()
    if args.num_scales < 2 or args.min_scale <= 0 or args.max_scale <= args.min_scale:
        raise SystemExit("scale grid requires 2+ points and 0 < min-scale < max-scale")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    images, excluded_small = _load_images(args)
    same_shape = len({image.size for _, image in images}) == 1
    codec = XJPEG(args.checkpoint, device=str(device))
    predicted = [codec.predict_tables(image) for _, image in images]
    scales = np.geomspace(args.min_scale, args.max_scale, args.num_scales)
    rates = np.empty((len(images), len(scales)), dtype=np.float64)
    metrics = np.empty((len(images), len(scales), len(METRIC_NAMES)), dtype=np.float64)
    pixels = np.asarray([image.width * image.height for _, image in images])

    print(
        f"sampled X-JPEG oracle: n={len(images)} domain="
        f"{'256' if args.domain_256 else 'native'} target={args.target_bpp:.5f} "
        f"objective={args.objective} scales={len(scales)} device={device} "
        f"excluded-small={excluded_small}",
        flush=True,
    )
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for scale_index, scale in enumerate(scales):
            jobs = (
                (image, tuple(np.asarray(table) * scale for table in tables))
                for (_, image), tables in zip(images, predicted)
            )
            datas = list(pool.map(lambda job: encode_with_tables(job[0], *job[1]), jobs))
            rates[:, scale_index] = np.asarray([len(data) * 8.0 for data in datas]) / pixels
            if same_shape:
                values = _measure_same_shape(
                    images, datas, device, args.batch_size, args.workers,
                )
            else:
                values = _measure_varied_shape(images, datas, device, args.workers)
            metrics[:, scale_index] = values
            print(
                f"  [{scale_index + 1:02d}/{len(scales):02d}] scale={scale:.5f} "
                f"mean={rates[:, scale_index].mean():.5f} bpp "
                f"RGB={values[:, 0].mean():.6f} Y={values[:, 1].mean():.6f} "
                f"611={values[:, 2].mean():.6f}",
                flush=True,
            )

    objective_index = METRIC_NAMES.index(args.objective)
    objective = metrics[:, :, objective_index]
    mean_rates = rates.mean(axis=0)
    feasible_global = np.flatnonzero(mean_rates <= args.target_bpp)
    if not len(feasible_global):
        raise RuntimeError(
            f"no sampled global scale reaches {args.target_bpp:.6f} bpp; "
            "increase --max-scale"
        )
    global_index = int(feasible_global[np.argmax(objective[:, feasible_global].mean(axis=0))])
    global_selection = np.full(len(images), global_index, dtype=np.int64)

    per_image_selection = np.empty(len(images), dtype=np.int64)
    for image_index in range(len(images)):
        feasible = np.flatnonzero(rates[image_index] <= args.target_bpp)
        if not len(feasible):
            feasible = np.asarray([int(np.argmin(rates[image_index]))])
        per_image_selection[image_index] = int(
            feasible[np.argmax(objective[image_index, feasible])]
        )

    allocation = allocate_sampled_rd(rates, objective, args.target_bpp)
    global_summary = _summarize_selection(rates, metrics, global_selection)
    capped_summary = _summarize_selection(rates, metrics, per_image_selection)
    oracle_summary = _summarize_selection(rates, metrics, allocation.selection)
    fractional_objective = allocation.fractional_total_quality / len(images)
    oracle_summary["fractional_objective_upper_bound"] = float(fractional_objective)
    oracle_summary["unused_mean_bpp"] = float(
        args.target_bpp - allocation.total_rate / len(images)
    )

    print("\nstrict sampled-grid results", flush=True)
    for label, summary in (
        ("global", global_summary),
        ("per-image cap", capped_summary),
        ("oracle allocation", oracle_summary),
    ):
        print(
            f"  {label:>17}: {summary['mean_bpp']:.6f} bpp "
            f"RGB={summary['rgb_msssim']:.6f} Y={summary['y_msssim']:.6f} "
            f"611={summary['ycbcr611_msssim']:.6f} "
            f"PSNR={summary['psnr']:.3f}",
            flush=True,
        )
    print(
        f"  fractional sampled-curve upper bound ({args.objective}): "
        f"{fractional_objective:.8f}",
        flush=True,
    )
    print(
        "  scope: upper bound only for convex mixtures of the sampled "
        "predicted-table scale points; not an upper bound for JPEG",
        flush=True,
    )

    result = {
        "checkpoint": str(args.checkpoint),
        "images": str(args.images),
        "num_images": len(images),
        "domain_256": args.domain_256,
        "excluded_small": excluded_small,
        "target_bpp": args.target_bpp,
        "objective": args.objective,
        "scales": scales.tolist(),
        "elapsed_seconds": time.monotonic() - started,
        "global": {**global_summary, "scale": float(scales[global_index])},
        "per_image_cap": capped_summary,
        "oracle": oracle_summary,
        "fractional_scope": (
            "upper bound for convex mixtures of sampled predicted-table scale "
            "points only; not an upper bound for JPEG"
        ),
        "per_image": [
            {
                "name": name,
                "oracle_scale": float(scales[allocation.selection[index]]),
                "oracle_bpp": float(rates[index, allocation.selection[index]]),
            }
            for index, (name, _) in enumerate(images)
        ],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
