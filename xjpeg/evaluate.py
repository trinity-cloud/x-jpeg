"""Rate-distortion evaluation on Kodak: X-JPEG vs baseline libjpeg.

For each checkpoint, predicted tables are swept over a global scale to trace
an RD curve from real encoded files; the baseline is the standard Annex-K
tables swept over IJG quality. Reports BD-rate (MS-SSIM-dB and PSNR) and an
adaptivity ablation (the set-average fixed table under the same sweep).

Usage:
    python -m xjpeg.evaluate --checkpoints runs/*/ckpt_final.pt \
        --kodak data/kodak --out results
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .codec import XJPEG
from .data import kodak_images
from .jpeg_io import decode, encode_standard, encode_with_tables
from .metrics import bd_rate, bpp_of, msssim, msssim_db, pil_to_tensor, psnr


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--kodak", required=True)
    p.add_argument("--out", default="results")
    p.add_argument("--scales", default=None,
                   help="comma-separated table scales (default geomspace 0.25..4, 13 pts)")
    p.add_argument("--qualities", default=None,
                   help="comma-separated baseline qualities (default 10..95 step 5)")
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


def _measure(img, data):
    ref, rec = pil_to_tensor(img), pil_to_tensor(decode(data))
    return bpp_of(data, img), msssim(ref, rec), psnr(ref, rec)


def sweep_tables(images, tables_per_image, scales):
    """Returns rows of (scale, bpp, msssim, psnr) for 2- or 3-table tuples."""
    rows = []
    for s in scales:
        pts = []
        for (name, img), tables in zip(images, tables_per_image):
            pts.append(_measure(
                img, encode_with_tables(img, *(table * s for table in tables))
            ))
        m = np.mean(pts, axis=0)
        rows.append((s, *m))
    return rows


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    images = kodak_images(args.kodak)
    scales = (np.array([float(s) for s in args.scales.split(",")])
              if args.scales else np.geomspace(0.25, 4.0, 13))
    qualities = ([int(q) for q in args.qualities.split(",")]
                 if args.qualities else list(range(10, 100, 5)))

    # Baseline: standard tables over quality sweep
    base_rows = []
    for q in qualities:
        pts = [_measure(img, encode_standard(img, q)) for _, img in images]
        m = np.mean(pts, axis=0)
        base_rows.append((q, *m))
    _write_csv(out / "baseline.csv", ["quality", "bpp", "msssim", "psnr"], base_rows)
    b_bpp = np.array([r[1] for r in base_rows])
    b_ms = np.array([r[2] for r in base_rows])
    b_ps = np.array([r[3] for r in base_rows])

    summary = {}
    for ckpt in args.checkpoints:
        name = Path(ckpt).parent.name or Path(ckpt).stem
        codec = XJPEG(checkpoint=ckpt, device=args.device)
        tables = [codec.predict_tables(img) for _, img in images]

        rows = sweep_tables(images, tables, scales)
        _write_csv(out / f"xjpeg_{name}.csv",
                   ["table_scale", "bpp", "msssim", "psnr"], rows)
        x_bpp = np.array([r[1] for r in rows])
        x_ms = np.array([r[2] for r in rows])
        x_ps = np.array([r[3] for r in rows])

        # Adaptivity ablation: one fixed (set-average) table for all images
        avg_t = tuple(
            np.mean([t[i] for t in tables], axis=0)
            for i in range(len(tables[0]))
        )
        fix_rows = sweep_tables(images, [avg_t] * len(images), scales)
        _write_csv(out / f"fixedavg_{name}.csv",
                   ["table_scale", "bpp", "msssim", "psnr"], fix_rows)
        f_bpp = np.array([r[1] for r in fix_rows])
        f_ms = np.array([r[2] for r in fix_rows])

        # Operating point of the raw predicted tables (scale = 1)
        op = sweep_tables(images, tables, [1.0])[0]

        summary[name] = {
            "bd_rate_msssim_vs_jpeg": bd_rate(b_bpp, msssim_db(b_ms), x_bpp, msssim_db(x_ms)),
            "bd_rate_psnr_vs_jpeg": bd_rate(b_bpp, b_ps, x_bpp, x_ps),
            "bd_rate_msssim_vs_fixedavg": bd_rate(f_bpp, msssim_db(f_ms), x_bpp, msssim_db(x_ms)),
            "operating_point": {"bpp": op[1], "msssim": op[2], "psnr": op[3]},
        }
        print(f"{name}: BD-rate(MS-SSIM) vs JPEG = "
              f"{summary[name]['bd_rate_msssim_vs_jpeg']:+.2f}%  "
              f"| vs fixed-avg table = "
              f"{summary[name]['bd_rate_msssim_vs_fixedavg']:+.2f}%  "
              f"| op point {op[1]:.3f} bpp @ MS-SSIM {op[2]:.4f}")

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    _plot(out, base_rows, args.checkpoints)
    print(f"results written to {out}/")


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([[f"{v:.5f}" if isinstance(v, float) else v for v in r] for r in rows])


def _plot(out, base_rows, checkpoints):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    b = np.array(base_rows, dtype=float)
    for ax, col, label in ((axes[0], 2, "MS-SSIM"), (axes[1], 3, "PSNR (dB)")):
        ax.plot(b[:, 1], b[:, col], "k--o", ms=3, label="JPEG (Annex-K + IJG scaling)")
        for ckpt in checkpoints:
            name = Path(ckpt).parent.name or Path(ckpt).stem
            path = out / f"xjpeg_{name}.csv"
            r = np.genfromtxt(path, delimiter=",", skip_header=1)
            ax.plot(r[:, 1], r[:, col], "-o", ms=3, label=f"X-JPEG {name}")
        ax.set_xlabel("bits per pixel")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Kodak rate-distortion: X-JPEG adaptive tables vs standard JPEG")
    fig.tight_layout()
    fig.savefig(out / "rd_curves.png", dpi=150)


if __name__ == "__main__":
    main()
