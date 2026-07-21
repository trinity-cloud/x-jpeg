"""Q-table prediction network.

Encoder: 256x256 RGB -> five stride-2 conv blocks -> 3x8x8 bottleneck.
The bottleneck IS the three quantization tables (Y, Cb, Cr), mapped to
[1, 255] with a scaled sigmoid. During training the tables get additive
uniform noise (+/- 0.5) — the exact relaxation of the 8-bit integer rounding
applied when the table is written into the JPEG DQT segment — and a
factorized entropy model provides the soft entropy constraint. A mirror
decoder reconstructs the 256x256 input from the (noisy) tables, closing the
autoencoder.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .entropy_bottleneck import FactorizedEntropyModel

# sigmoid temperature for code -> table mapping
_TABLE_TEMP = 4.0
# head bias so that initial tables start near 16 (standard-table magnitude)
_INIT_TABLE = 16.0
_HEAD_BIAS = _TABLE_TEMP * math.log((_INIT_TABLE - 1.0) / (255.0 - _INIT_TABLE))


class _LowerBound(torch.autograd.Function):
    """clamp_min whose gradient still passes when pushing back into bounds."""

    @staticmethod
    def forward(ctx, x, bound: float):
        ctx.save_for_backward(x)
        ctx.bound = bound
        return x.clamp_min(bound)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        can_pass = (x >= ctx.bound) | (grad_output < 0)
        return grad_output * can_pass, None


class GDN(nn.Module):
    """Generalized Divisive Normalization (Balle, Laparra & Simoncelli 2016).

    y_i = x_i / sqrt(beta_i + sum_j gamma_ij * x_j^2); the inverse form
    (IGDN) multiplies instead of divides and is used in the decoder.
    """

    def __init__(self, channels: int, inverse: bool = False):
        super().__init__()
        self.inverse = inverse
        self.beta = nn.Parameter(torch.ones(channels))
        self.gamma = nn.Parameter(0.1 * torch.eye(channels))

    def forward(self, x):
        c = x.shape[1]
        beta = _LowerBound.apply(self.beta, 1e-6)
        gamma = _LowerBound.apply(self.gamma, 0.0)
        norm = torch.sqrt(F.conv2d(x * x, gamma.view(c, c, 1, 1), beta))
        return x * norm if self.inverse else x / norm


class ReparameterizedGDN(nn.Module):
    """Numerically stable GDN/IGDN parameterization used by Ballé-style models.

    The trainable parameters live in square-root space, which keeps beta and
    gamma non-negative without repeatedly pinning raw parameters at zero.
    It is separate from :class:`GDN` because the parameter semantics differ.
    """

    def __init__(
        self, channels: int, inverse: bool = False,
        beta_min: float = 1e-6, reparam_offset: float = 2 ** -18,
    ):
        super().__init__()
        self.inverse = inverse
        self.pedestal = reparam_offset ** 2
        self.beta_bound = math.sqrt(beta_min + self.pedestal)
        self.gamma_bound = reparam_offset
        self.beta = nn.Parameter(torch.sqrt(torch.ones(channels) + self.pedestal))
        gamma = 0.1 * torch.eye(channels)
        self.gamma = nn.Parameter(torch.sqrt(gamma + self.pedestal))

    def forward(self, x):
        c = x.shape[1]
        beta = _LowerBound.apply(self.beta, self.beta_bound).square() - self.pedestal
        gamma = _LowerBound.apply(self.gamma, self.gamma_bound).square() - self.pedestal
        norm = torch.sqrt(F.conv2d(x.square(), gamma.view(c, c, 1, 1), beta))
        return x * norm if self.inverse else x / norm


class EncBlock(nn.Module):
    def __init__(self, cin, cout, gdn_reparam: bool = False):
        super().__init__()
        norm = ReparameterizedGDN if gdn_reparam else GDN
        self.down = nn.Conv2d(cin, cout, 3, stride=2, padding=1)
        self.gdn1 = norm(cout)
        self.conv = nn.Conv2d(cout, cout, 3, padding=1)
        self.gdn2 = norm(cout)

    def forward(self, x):
        x = self.gdn1(self.down(x))
        return self.gdn2(self.conv(x))


class CompactEncBlock(nn.Module):
    """One learned downsampling transform in the compact encoder."""

    def __init__(self, cin, cout, gdn_reparam: bool = False):
        super().__init__()
        norm = ReparameterizedGDN if gdn_reparam else GDN
        self.conv = nn.Conv2d(cin, cout, 5, stride=2, padding=2)
        self.norm = norm(cout)

    def forward(self, x):
        return self.norm(self.conv(x))


class DecBlock(nn.Module):
    def __init__(self, cin, cout, use_igdn: bool = False):
        super().__init__()
        self.use_igdn = use_igdn
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        if use_igdn:
            self.norm1 = ReparameterizedGDN(cout, inverse=True)
            self.norm2 = ReparameterizedGDN(cout, inverse=True)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv1(x)
        x = self.norm1(x) if self.use_igdn else F.gelu(x)
        x = self.conv2(x)
        return self.norm2(x) if self.use_igdn else F.gelu(x)


class CompactDecBlock(nn.Module):
    """One learned upsampling transform followed by IGDN."""

    def __init__(self, cin, cout, use_igdn: bool = True):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 5, stride=2, padding=2,
                                     output_padding=1)
        self.norm = (ReparameterizedGDN(cout, inverse=True)
                     if use_igdn else None)

    def forward(self, x):
        x = self.up(x)
        return self.norm(x) if self.norm is not None else F.gelu(x)


class Encoder(nn.Module):
    def __init__(
        self, widths=(64, 128, 192, 256, 256), dct_stats: bool = False,
        num_tables: int = 3, gdn_reparam: bool = False,
        architecture: str = "deep", conditioned_init: bool = False,
    ):
        super().__init__()
        if num_tables not in (2, 3):
            raise ValueError(f"num_tables must be 2 or 3, got {num_tables}")
        if architecture not in ("deep", "compact"):
            raise ValueError(f"unknown architecture: {architecture}")
        self.dct_stats = dct_stats
        self.num_tables = num_tables
        self.gdn_reparam = gdn_reparam
        self.architecture = architecture
        self.conditioned_init = conditioned_init
        if architecture == "compact":
            # Four conv+GDN stages produce a 16x16 representation. The fifth
            # downsampling convolution directly emits the 3x8x8 bottleneck;
            # as in Ballé-style compression transforms, no GDN follows the
            # entropy-constrained bottleneck itself.
            hist_widths = tuple(widths[:4])
            chans = [3] + list(hist_widths)
            self.blocks = nn.Sequential(*[
                CompactEncBlock(
                    chans[i], chans[i + 1], gdn_reparam=gdn_reparam
                )
                for i in range(len(hist_widths))
            ])
            self.head = nn.Conv2d(
                hist_widths[-1], num_tables, 5, stride=2, padding=2
            )
            if dct_stats:
                # Expose full-image coefficient energy at
                # the matching component/frequency. A separate gain for every
                # DQT entry preserves the 8x8 frequency semantics instead of
                # convolving neighbouring, unrelated DCT bands together.
                self.stats_scale = nn.Parameter(
                    torch.full((1, num_tables, 8, 8), 0.1)
                )
        else:
            chans = [3] + list(widths)
            self.blocks = nn.Sequential(
                *[
                    EncBlock(chans[i], chans[i + 1], gdn_reparam=gdn_reparam)
                    for i in range(len(widths))
                ]
            )
            self.head = nn.Conv2d(
                widths[-1] + (num_tables if dct_stats else 0),
                num_tables, 3, padding=1,
            )
        if conditioned_init:
            # Keep the initial operating point close to Q=16 while allowing
            # image-dependent gradients to reach every analysis transform on
            # the first optimization step. A zero head severs that gradient
            # path and strongly biases short runs toward a global table.
            nn.init.xavier_normal_(self.head.weight, gain=0.1)
        else:
            nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, _HEAD_BIAS)

    def forward(self, x, stats=None):
        # Center once here so training and inference cannot disagree.
        h = self.blocks(x * 2.0 - 1.0)
        if self.dct_stats and self.architecture == "deep":
            if stats is None:
                raise ValueError("this encoder requires dct_band_stats input")
            h = torch.cat([h, stats / 5.0], dim=1)  # log1p scale ~[0,7] -> ~[0,1.4]
        raw = self.head(h)
        if self.dct_stats and self.architecture == "compact":
            if stats is None:
                raise ValueError("this encoder requires dct_band_stats input")
            raw = raw + self.stats_scale * (stats / 5.0)
        return raw  # (B, num_tables, 8, 8) raw code


class Decoder(nn.Module):
    def __init__(
        self, widths=(256, 192, 128, 64, 48), num_tables: int = 3,
        use_igdn: bool = False, architecture: str = "deep",
    ):
        super().__init__()
        if architecture not in ("deep", "compact"):
            raise ValueError(f"unknown architecture: {architecture}")
        self.architecture = architecture
        if architecture == "compact":
            chans = list(widths[:4])
            if len(chans) != 4:
                raise ValueError("compact decoder requires four stage widths")
            self.stem = CompactDecBlock(
                num_tables, chans[0], use_igdn=use_igdn
            )
            self.blocks = nn.Sequential(*[
                CompactDecBlock(chans[i], chans[i + 1], use_igdn=use_igdn)
                for i in range(len(chans) - 1)
            ])
            self.head = nn.ConvTranspose2d(
                chans[-1], 3, 5, stride=2, padding=2, output_padding=1
            )
        else:
            self.stem = nn.Conv2d(num_tables, widths[0], 3, padding=1)
            blocks = []
            chans = list(widths)
            for i in range(len(chans) - 1):
                blocks.append(DecBlock(chans[i], chans[i + 1], use_igdn=use_igdn))
            blocks.append(DecBlock(chans[-1], chans[-1], use_igdn=use_igdn))
            self.blocks = nn.Sequential(*blocks)
            self.head = nn.Conv2d(chans[-1], 3, 3, padding=1)

    def forward(self, t):
        return torch.sigmoid(self.head(self.blocks(self.stem(t))))


def code_to_table(z: torch.Tensor) -> torch.Tensor:
    """Raw code -> continuous Q-table in [1, 255]."""
    return 1.0 + 254.0 * torch.sigmoid(z / _TABLE_TEMP)


class QTableNet(nn.Module):
    def __init__(
        self, dct_stats: bool = False, num_tables: int = 3,
        gdn_reparam: bool = False, decoder_igdn: bool = False,
        architecture: str = "deep", conditioned_init: bool = False,
        width_multiplier: float = 1.0,
    ):
        super().__init__()
        if width_multiplier <= 0:
            raise ValueError("width_multiplier must be positive")
        self.width_multiplier = float(width_multiplier)
        encoder_widths = tuple(
            max(1, int(round(width * self.width_multiplier)))
            for width in (64, 128, 192, 256, 256)
        )
        decoder_widths = tuple(
            max(1, int(round(width * self.width_multiplier)))
            for width in (256, 192, 128, 64, 48)
        )
        self.num_tables = num_tables
        self.architecture = architecture
        self.encoder = Encoder(
            widths=encoder_widths,
            dct_stats=dct_stats, num_tables=num_tables,
            gdn_reparam=gdn_reparam, architecture=architecture,
            conditioned_init=conditioned_init,
        )
        self.decoder = Decoder(
            widths=decoder_widths,
            num_tables=num_tables, use_igdn=decoder_igdn,
            architecture=architecture,
        )
        self.entropy = FactorizedEntropyModel(channels=num_tables)

    def forward(self, x256: torch.Tensor, stats: torch.Tensor = None):
        """x256: (B, 3, 256, 256) in [0, 1].

        Returns (tables, table_bits, recon):
          tables:     (B, num_tables, 8, 8), noisy-relaxed (train) or integer (eval)
          table_bits: total bits of the tables under the factorized prior
          recon:      (B, 3, 256, 256) autoencoder reconstruction
        """
        z = self.encoder(x256, stats)
        t = code_to_table(z)
        if self.training:
            t = t + torch.empty_like(t).uniform_(-0.5, 0.5)
        else:
            t = torch.round(t)
        t = t.clamp(1.0, 255.0)
        table_bits = self.entropy.bits(t)
        recon = self.decoder(t / 255.0 - 0.5)
        return t, table_bits, recon

    @torch.no_grad()
    def predict_tables(self, x256: torch.Tensor, stats: torch.Tensor = None) -> torch.Tensor:
        """Integer tables for inference, values in [1, 255]."""
        was_training = self.training
        self.eval()
        t, _, _ = self.forward(x256, stats)
        if was_training:
            self.train()
        return t
