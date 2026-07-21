"""Train the X-JPEG Q-table network end-to-end through differentiable JPEG.

Loss = lambda_d * (1 - MS-SSIM(x, diffJPEG(x, T)))     distortion
     + lambda_r * bpp_proxy(quantized DCT coefficients)  rate
     + lambda_y * bpp(T) under the factorized prior      table entropy (regularizer)
     + lambda_ae * MSE(x, decoder(T~))                   autoencoder reconstruction

Validation runs the REAL pipeline: integer tables -> libjpeg -> file bytes.

Usage:
    python -m xjpeg.train --data data/train2017 --val data/kodak --out runs/lr0.03 \
        --lambda-r 0.03 --steps 100000
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader

from .data import ImageCrops, PairedThumbCrops, kodak_images
from .stats import dct_band_stats
from .diffjpeg import DiffJPEG
from .jpeg_io import decode, encode_with_tables
from .metrics import bpp_of, msssim, pil_to_tensor, psnr
from .model import QTableNet
from .rate import DctRateModel


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--val", required=True, help="Kodak directory for real-JPEG validation")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-entropy", type=float, default=1e-3,
                   help="lr for the entropy/rate models (must track faster than the encoder)")
    p.add_argument("--rate-warmup", type=int, default=500,
                   help="steps during which the rate loss trains only the price models")
    p.add_argument("--ae-pretrain", type=int, default=0,
                   help="AE-only encoder warmup steps before joint JPEG optimization")
    p.add_argument("--lambda-d", type=float, default=1.0)
    p.add_argument(
        "--lambda-luma", type=float, default=0.0,
        help="additional (1 - luma MS-SSIM) product guard alongside RGB MS-SSIM",
    )
    p.add_argument("--lambda-r", type=float, default=0.03)
    p.add_argument("--lambda-y", type=float, default=0.01)
    p.add_argument("--lambda-ae", type=float, default=1.0)
    p.add_argument("--lambda-ae-final", type=float, default=None,
                   help="optional AE weight after post-pretrain linear decay")
    p.add_argument("--ae-decay-steps", type=int, default=0,
                   help="steps after AE pretraining over which to reach --lambda-ae-final")
    p.add_argument("--lambda-mse", type=float, default=0.0,
                   help="MSE guard term on the diff-JPEG output (anti MS-SSIM gaming)")
    p.add_argument("--native-crops", action="store_true",
                   help="predict tables from the whole-image thumbnail but compress a "
                        "native-resolution crop (deployment-faithful training)")
    p.add_argument("--dct-stats", action="store_true",
                   help="feed the encoder full-res DCT band statistics (adaptivity channel)")
    p.add_argument("--num-tables", type=int, choices=(2, 3), default=3,
                   help="predict Y/Cb/Cr tables (3) or shared-chroma tables (2)")
    p.add_argument("--luma-metric", action="store_true",
                   help="optimize MS-SSIM/MSE on the luma channel only (2018-era convention)")
    p.add_argument("--gdn-reparam", action="store_true",
                   help="use Ballé-style non-negative reparameterization in encoder GDN")
    p.add_argument("--igdn-decoder", action="store_true",
                   help="restore reparameterized IGDN in the mirror decoder")
    p.add_argument("--architecture", choices=("deep", "compact"), default="deep",
                   help="network topology; compact is the release encoder family")
    p.add_argument("--conditioned-init", action="store_true",
                   help="small nonzero table-head init so conditioning trains from step one")
    p.add_argument(
        "--width-multiplier", type=float, default=1.0,
        help="multiply every encoder/decoder stage width",
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-every", type=int, default=2500)
    p.add_argument("--ckpt-every", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None)
    p.add_argument(
        "--init-from", default=None,
        help="initialize model/rate weights from a checkpoint but use a fresh "
             "optimizer and step counter (permits compatible product extensions)",
    )
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def _luma(t):
    return 0.299 * t[:, 0:1] + 0.587 * t[:, 1:2] + 0.114 * t[:, 2:3]


@torch.no_grad()
def validate_real(model, val_set, device, use_stats=False, luma_metric=False):
    """Predicted integer tables -> real libjpeg files on the val set."""
    model.eval()
    rows = []
    for _, img in val_set:
        ref = pil_to_tensor(img)
        thumb = pil_to_tensor(img.resize((256, 256), Image.BICUBIC)).to(device)
        st = dct_band_stats(ref.to(device), model.num_tables) if use_stats else None
        t = model.predict_tables(thumb, st)[0].cpu().numpy()
        data = encode_with_tables(img, *t)
        rec = pil_to_tensor(decode(data))
        metric_ref, metric_rec = (_luma(ref), _luma(rec)) if luma_metric else (ref, rec)
        rows.append((
            bpp_of(data, img), msssim(metric_ref, metric_rec),
            psnr(metric_ref, metric_rec),
        ))
    model.train()
    n = len(rows)
    return tuple(sum(s[i] for s in rows) / n for i in range(3))


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    if args.resume and args.init_from:
        raise SystemExit("choose only one of --resume and --init-from")
    model = QTableNet(
        dct_stats=args.dct_stats, num_tables=args.num_tables,
        gdn_reparam=args.gdn_reparam, decoder_igdn=args.igdn_decoder,
        architecture=args.architecture, conditioned_init=args.conditioned_init,
        width_multiplier=args.width_multiplier,
    ).to(device)
    djpeg = DiffJPEG().to(device)
    rate = DctRateModel(num_components=args.num_tables).to(device)
    entropy_params = list(rate.parameters()) + list(model.entropy.parameters())
    entropy_ids = {id(p) for p in entropy_params}
    main_params = [p for p in model.parameters() if id(p) not in entropy_ids]
    opt = torch.optim.Adam([
        {"params": main_params, "lr": args.lr},
        {"params": entropy_params, "lr": args.lr_entropy},
    ])

    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        rate.load_state_dict(ck["rate"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        print(f"resumed from {args.resume} at step {start_step}")
    elif args.init_from:
        ck = torch.load(args.init_from, map_location=device, weights_only=True)
        incompatible = model.load_state_dict(ck["model"], strict=False)
        allowed_missing = {"encoder.stats_scale"} if args.dct_stats else set()
        missing = set(incompatible.missing_keys) - allowed_missing
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"incompatible --init-from checkpoint; missing={sorted(missing)}, "
                f"unexpected={sorted(incompatible.unexpected_keys)}"
            )
        if "rate" in ck:
            rate.load_state_dict(ck["rate"])
        print(
            f"initialized weights from {args.init_from}; fresh optimizer at step 0"
        )

    ds = (PairedThumbCrops(
              args.data, num_tables=args.num_tables, compute_stats=args.dct_stats
          )
          if args.native_crops else ImageCrops(args.data))
    print(f"train images: {len(ds)}  (native-crops: {args.native_crops})")
    dl = DataLoader(
        ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        drop_last=True, pin_memory=True, persistent_workers=args.workers > 0,
    )
    val_set = kodak_images(args.val)

    log_path = out / "log.csv"
    new_log = not log_path.exists()
    log_f = open(log_path, "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow([
            "step", "loss", "msssim", "bpp_proxy", "table_bpp", "ae_mse",
            "mean_qy", "mean_qcb", "mean_qcr",
            "val_bpp", "val_msssim", "val_psnr", "ips",
        ])

    def save(step, name):
        torch.save(
            {"step": step, "model": model.state_dict(), "rate": rate.state_dict(),
             "opt": opt.state_dict(), "args": vars(args)},
            out / name,
        )
        torch.save({"model": {f"encoder.{k}": v for k, v in
                              model.encoder.state_dict().items()},
                    "dct_stats": args.dct_stats,
                    "gdn_reparam": args.gdn_reparam,
                    "architecture": args.architecture,
                    "width_multiplier": args.width_multiplier},
                   out / "encoder.pt")

    model.train()
    step = start_step
    nan_skips = 0
    t0, n0 = time.time(), start_step
    it = iter(dl)
    while step < args.steps:
        try:
            x = next(it)
        except StopIteration:
            it = iter(dl)
            x = next(it)
        if args.native_crops:
            thumb, target, bstats = (t.to(device, non_blocking=True) for t in x)
        else:
            thumb = target = x.to(device, non_blocking=True)
            bstats = (dct_band_stats(target, args.num_tables)
                      if args.dct_stats else None)
        b = target.shape[0]
        npix = b * target.shape[-1] * target.shape[-2]

        tables, table_bits, recon = model(thumb, bstats if args.dct_stats else None)
        x_hat, scaled = djpeg(target, tables)

        y_target, y_hat = _luma(target), _luma(x_hat)
        if args.luma_metric:
            mx, mh = y_target, y_hat
            d_ms = ms_ssim(mx, mh, data_range=1.0)
            distortion = 1.0 - d_ms
        else:
            mx, mh = target, x_hat
            d_ms = ms_ssim(mx, mh, data_range=1.0)
            distortion = 1.0 - d_ms
            if args.lambda_luma:
                distortion = distortion + args.lambda_luma * (
                    1.0 - ms_ssim(y_target, y_hat, data_range=1.0)
                )
        # During warmup the price models calibrate to the symbol distribution
        # without pushing mispriced rate gradients into the encoder.
        detach_rate = step < max(args.rate_warmup, args.ae_pretrain)
        rate_in = ({k: v.detach() for k, v in scaled.items()}
                   if detach_rate else scaled)
        bpp = rate.bpp(rate_in, npix)
        table_bpp = table_bits / npix
        ae = F.mse_loss(recon, thumb)
        mse = F.mse_loss(mh, mx) if args.luma_metric else F.mse_loss(x_hat, target)
        ae_weight = args.lambda_ae
        if args.lambda_ae_final is not None and step >= args.ae_pretrain:
            if args.ae_decay_steps > 0:
                progress = min(
                    (step - args.ae_pretrain) / args.ae_decay_steps, 1.0
                )
                ae_weight = (args.lambda_ae
                             + progress * (args.lambda_ae_final - args.lambda_ae))
            else:
                ae_weight = args.lambda_ae_final
        if step < args.ae_pretrain:
            # Give the bottleneck and mirror decoder time to become genuinely
            # image-conditioned before JPEG rate pressure can settle into the
            # easier global-table solution. The detached DCT rate term still
            # calibrates the rate model during this stage.
            loss = (args.lambda_r * bpp + args.lambda_y * table_bpp
                    + args.lambda_ae * ae)
        else:
            loss = (args.lambda_d * distortion + args.lambda_r * bpp
                    + args.lambda_y * table_bpp + ae_weight * ae
                    + args.lambda_mse * mse)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(rate.parameters()), 1.0
        )
        # One bad batch must not poison the parameters: skip non-finite steps.
        if torch.isfinite(norm):
            opt.step()
        else:
            nan_skips += 1
            print(f"step {step}: non-finite grad norm, step skipped "
                  f"({nan_skips} total)", flush=True)
        step += 1

        if step % args.log_every == 0 or step == 1:
            ips = (step - n0) * args.batch / max(time.time() - t0, 1e-9)
            t0, n0 = time.time(), step
            mq = tables.detach().mean(dim=(0, 2, 3))
            mqcr = mq[2] if len(mq) == 3 else mq[1]
            row = [step, f"{loss.item():.5f}", f"{d_ms.item():.5f}",
                   f"{bpp.item():.4f}", f"{table_bpp.item():.6f}",
                   f"{ae.item():.5f}", f"{mq[0].item():.1f}",
                   f"{mq[1].item():.1f}", f"{mqcr.item():.1f}",
                   "", "", "", f"{ips:.1f}"]
            stage = "  [ae-pretrain]" if step <= args.ae_pretrain else ""
            print(f"step {step}{stage}  loss {loss.item():.4f}  ms-ssim {d_ms.item():.4f}  "
                  f"bpp~ {bpp.item():.3f}  tab-bpp {table_bpp.item():.5f}  "
                  f"ae {ae.item():.4f} (w={ae_weight:.3f})  qY {mq[0].item():.1f}  "
                  f"qCb {mq[1].item():.1f}  qCr {mqcr.item():.1f}  "
                  f"{ips:.0f} img/s", flush=True)
            log.writerow(row)
            log_f.flush()

        if step % args.val_every == 0:
            vb, vm, vp = validate_real(
                model, val_set, device, args.dct_stats, args.luma_metric
            )
            print(f"[val@{step}] real-JPEG kodak: {vb:.3f} bpp  "
                  f"MS-SSIM {vm:.4f}  PSNR {vp:.2f} dB", flush=True)
            log.writerow([step, "", "", "", "", "", "", "", "",
                          f"{vb:.4f}", f"{vm:.5f}", f"{vp:.3f}", ""])
            log_f.flush()

        if step % args.ckpt_every == 0:
            save(step, "ckpt_latest.pt")

    save(step, "ckpt_final.pt")
    vb, vm, vp = validate_real(
        model, val_set, device, args.dct_stats, args.luma_metric
    )
    print(f"[final] real-JPEG kodak: {vb:.3f} bpp  MS-SSIM {vm:.4f}  PSNR {vp:.2f} dB")
    log_f.close()


if __name__ == "__main__":
    main()
