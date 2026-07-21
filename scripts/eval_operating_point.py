"""Compare table methods at the same mean *real-file* bitrate.

Unlike ``repro_eval.py`` (a per-image quality guarantee), this evaluator uses
one global scale per method. It therefore preserves the predictor's relative
per-image rate decisions and directly tests claims such as "0.8 bpp at
MS-SSIM > 0.99". All search decisions use complete libjpeg file lengths.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_msssim import ms_ssim

from xjpeg.codec import XJPEG
from xjpeg.jpeg_io import STD_CHROMA, STD_LUMA, decode, encode_with_tables
from xjpeg.metrics import bpp_of, pil_to_tensor, psnr, ycbcr611_msssim


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--images", default="data/kodak")
    p.add_argument("--target-bpp", type=float, default=0.8)
    p.add_argument("--domain-256", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--iters", type=int, default=16)
    p.add_argument("--limit", type=int, default=0,
                   help="deterministic random image subset (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32,
                   help="metric batch size when all images share dimensions")
    p.add_argument("--bootstrap", type=int, default=2000,
                   help="paired bootstrap replicates for adaptive-control deltas")
    return p.parse_args()


def _luma(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    codec = XJPEG(args.checkpoint, device=str(device))
    paths = sorted(
        p for p in Path(args.images).iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
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
        img = Image.open(path).convert("RGB")
        if args.domain_256:
            img = img.resize((256, 256), Image.BICUBIC)
        images.append((path.name, img))

    predicted = [codec.predict_tables(img) for _, img in images]
    num_tables = len(predicted[0])
    fixed = tuple(
        np.mean([tables[i] for tables in predicted], axis=0)
        for i in range(num_tables)
    )
    methods = {
        "adaptive": predicted,
        "shuffled": predicted[1:] + predicted[:1],
        "fixed-avg": [fixed] * len(images),
        "annex-k": [(STD_LUMA, STD_CHROMA)] * len(images),
    }

    stacked = np.stack([np.stack(t) for t in predicted])
    entry_std = stacked.std(axis=0)
    relative_std = entry_std / np.maximum(stacked.mean(axis=0), 1.0)
    print(
        f"n={len(images)} target={args.target_bpp:.3f} bpp "
        f"domain={'256' if args.domain_256 else 'native'} "
        f"tables={num_tables} excluded-small={excluded_small}", flush=True,
    )
    print(
        f"predicted table variability: mean std {entry_std.mean():.3f}, "
        f"relative {100 * relative_std.mean():.3f}%", flush=True,
    )

    def encode_all(tables_for_images, scale):
        return [
            encode_with_tables(
                img, *(np.asarray(table) * scale for table in tables)
            )
            for (_, img), tables in zip(images, tables_for_images)
        ]

    def mean_bpp(datas):
        return float(np.mean([
            bpp_of(data, img) for data, (_, img) in zip(datas, images)
        ]))

    def find_scale(tables_for_images):
        # Bpp is almost monotone in table scale but integer clipping and
        # optimized Huffman coding introduce small plateaus. Retain every
        # candidate and select by achieved real bpp rather than trusting the
        # final binary-search endpoint.
        candidates = []

        def evaluate(scale):
            datas = encode_all(tables_for_images, scale)
            bpp = mean_bpp(datas)
            candidates.append((abs(bpp - args.target_bpp), scale, bpp, datas))
            return bpp

        lo, hi = 0.02, 16.0
        blo, bhi = evaluate(lo), evaluate(hi)
        if blo < args.target_bpp or bhi > args.target_bpp:
            raise RuntimeError(
                f"target {args.target_bpp} not bracketed by "
                f"{blo:.3f}..{bhi:.3f} bpp"
            )
        for _ in range(args.iters):
            mid = (lo + hi) / 2.0
            if evaluate(mid) > args.target_bpp:
                lo = mid
            else:
                hi = mid
        return min(candidates, key=lambda row: row[0])

    metric_rows = {}
    achieved_bpp = {}
    same_shape = len({img.size for _, img in images}) == 1

    def measure(datas):
        if not same_shape:
            rows = []
            for data, (_, img) in zip(datas, images):
                ref = pil_to_tensor(img).to(device)
                rec = pil_to_tensor(decode(data)).to(device)
                rows.append((
                    ms_ssim(ref, rec, data_range=1.0).item(),
                    ms_ssim(_luma(ref), _luma(rec), data_range=1.0).item(),
                    ycbcr611_msssim(ref, rec).item(),
                    psnr(ref, rec),
                ))
            return np.asarray(rows)

        rows = []
        for start in range(0, len(images), args.batch_size):
            stop = min(start + args.batch_size, len(images))
            refs = torch.cat([
                pil_to_tensor(img) for _, img in images[start:stop]
            ]).to(device)
            recs = torch.cat([
                pil_to_tensor(decode(data)) for data in datas[start:stop]
            ]).to(device)
            rgb_ms = ms_ssim(
                refs, recs, data_range=1.0, size_average=False
            )
            y_ms = ms_ssim(
                _luma(refs), _luma(recs), data_range=1.0, size_average=False
            )
            ycc611_ms = ycbcr611_msssim(
                refs, recs, size_average=False
            )
            mse = (refs - recs).square().mean(dim=(1, 2, 3))
            batch_psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
            rows.extend(torch.stack(
                [rgb_ms, y_ms, ycc611_ms, batch_psnr], dim=1
            ).cpu().numpy())
        return np.asarray(rows)

    for method, tables_for_images in methods.items():
        _, scale, bpp, datas = find_scale(tables_for_images)
        values = measure(datas)
        metric_rows[method] = values
        achieved_bpp[method] = bpp
        print(
            f"{method:>10}: {bpp:.4f} bpp  scale {scale:.4f}  "
            f"RGB-MS {values[:, 0].mean():.5f}  "
            f"Y-MS {values[:, 1].mean():.5f}  "
            f"611-MS {values[:, 2].mean():.5f}  "
            f"PSNR {values[:, 3].mean():.2f} dB",
            flush=True,
        )

    if args.bootstrap and "adaptive" in metric_rows:
        rng = np.random.default_rng(args.seed)
        print("paired RGB-MS deltas (adaptive - control):", flush=True)
        for control in ("fixed-avg", "shuffled", "annex-k"):
            delta = metric_rows["adaptive"][:, 0] - metric_rows[control][:, 0]
            indices = rng.integers(
                0, len(delta), size=(args.bootstrap, len(delta))
            )
            means = delta[indices].mean(axis=1)
            lo, hi = np.percentile(means, [2.5, 97.5])
            rate_gap = achieved_bpp["adaptive"] - achieved_bpp[control]
            print(
                f"  vs {control:>9}: {delta.mean():+.6f} "
                f"(95% CI {lo:+.6f}..{hi:+.6f}); "
                f"mean-bpp gap {rate_gap:+.5f}", flush=True,
            )


if __name__ == "__main__":
    main()
