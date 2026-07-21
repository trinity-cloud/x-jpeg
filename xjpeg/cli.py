"""X-JPEG command line interface.

Examples:
    xjpeg photo.png                      # -> photo.jpg with adaptive tables
    xjpeg photo.png -o small.jpg --target-bpp 0.5
    xjpeg photo.png -o faithful.jpg --target-msssim 0.98
    xjpeg *.png --scale 1.5              # coarser tables, smaller files
"""

import argparse
import sys
from pathlib import Path

from .codec import XJPEG

_SUBSAMPLING = {"444": 0, "422": 1, "420": 2}


def _output_path(inp: Path) -> Path:
    if inp.suffix.lower() in (".jpg", ".jpeg"):
        return inp.with_suffix(".xjpeg.jpg")
    return inp.with_suffix(".jpg")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="xjpeg",
        description="Compress images to standard JPEG with neural adaptive quantization tables.",
    )
    p.add_argument("inputs", nargs="+", help="input image(s)")
    p.add_argument("-o", "--output", help="output path (single input only)")
    p.add_argument("--checkpoint", help="model checkpoint (default: packaged weights or $XJPEG_CHECKPOINT)")
    p.add_argument("--target-msssim", type=float, default=None,
                   help="guarantee at least this MS-SSIM (e.g. 0.98); searches table scale per image")
    p.add_argument(
        "--target-bpp", type=float, default=None,
        help="target complete-file bits per source pixel (e.g. 0.5)",
    )
    p.add_argument("--scale", type=float, default=1.0,
                   help="global multiplier on predicted tables (>1 = smaller files)")
    p.add_argument("--subsampling", choices=_SUBSAMPLING, default="420")
    p.add_argument(
        "--sequential", action="store_true",
        help="disable the default progressive encoding",
    )
    p.add_argument(
        "--three-dqt", action="store_true",
        help="emit separate Cb and Cr DQTs instead of the default shared chroma DQT",
    )
    p.add_argument(
        "--backend", choices=("auto", "mozjpeg", "pillow"), default="auto",
        help="JPEG encoder backend; auto prefers bundled/system MozJPEG",
    )
    p.add_argument(
        "--mozjpeg", help="explicit path to MozJPEG cjpeg (also: XJPEG_MOZJPEG)",
    )
    p.add_argument("--no-optimize", action="store_true", help="disable optimized Huffman tables")
    p.add_argument("--device", default="cpu")
    p.add_argument("--compare", action="store_true",
                   help="also encode a standard JPEG matched to the same MS-SSIM and report savings")
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args(argv)

    if args.output and len(args.inputs) > 1:
        p.error("-o/--output only works with a single input")
    if args.target_bpp is not None and args.target_msssim is not None:
        p.error("--target-bpp and --target-msssim are mutually exclusive")

    codec = XJPEG(
        checkpoint=args.checkpoint,
        device=args.device,
        backend=args.backend,
        mozjpeg_path=args.mozjpeg,
    )
    kwargs = dict(
        table_scale=args.scale,
        target_bpp=args.target_bpp,
        target_msssim=args.target_msssim,
        subsampling=_SUBSAMPLING[args.subsampling],
        optimize=not args.no_optimize,
        progressive=not args.sequential,
        share_chroma=not args.three_dqt,
    )
    for inp in args.inputs:
        inp = Path(inp)
        out = Path(args.output) if args.output else _output_path(inp)
        if args.compare:
            cmp = codec.compare_with_baseline(inp, output=out, **kwargs)
            res = cmp["xjpeg"]
        else:
            res = codec.compress(inp, output=out, **kwargs)
        if not args.quiet:
            print(
                f"{inp} -> {out}  {res.size_bytes/1024:.1f} KiB  "
                f"{res.bpp:.3f} bpp  MS-SSIM {res.msssim:.4f}  "
                f"PSNR {res.psnr:.2f} dB  (table scale {res.table_scale:.2f})"
                f"  backend={res.backend}"
            )
            if args.compare:
                print(
                    f"  vs JPEG q{cmp['baseline_quality']}: "
                    f"{cmp['baseline_bytes']/1024:.1f} KiB  "
                    f"{cmp['baseline_bpp']:.3f} bpp  "
                    f"MS-SSIM {cmp['baseline_msssim']:.4f}  ->  "
                    f"X-JPEG is {cmp['savings_pct']:+.1f}% smaller at matched quality"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
