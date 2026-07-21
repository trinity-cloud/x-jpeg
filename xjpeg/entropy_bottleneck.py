"""Fully-factorized entropy model (Balle et al. 2018, appendix 6.1).

Models each channel of the bottleneck with a flexible learned univariate
density, evaluated as the probability mass on [v - 0.5, v + 0.5]. Used here as
the soft entropy constraint on the predicted Q-tables (values live in
[1, 255], i.e. the integer domain actually written to the JPEG DQT segment).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedEntropyModel(nn.Module):
    def __init__(self, channels: int = 2, filters=(3, 3, 3), init_scale: float = 64.0):
        super().__init__()
        self._channels = channels
        sizes = (1,) + tuple(filters) + (1,)
        self._num_layers = len(sizes) - 1
        scale = init_scale ** (1.0 / self._num_layers)

        self._matrices = nn.ParameterList()
        self._biases = nn.ParameterList()
        self._factors = nn.ParameterList()
        for k in range(self._num_layers):
            init = math.log(math.expm1(1.0 / scale / sizes[k + 1]))
            m = nn.Parameter(torch.full((channels, sizes[k + 1], sizes[k]), init))
            b = nn.Parameter(torch.empty(channels, sizes[k + 1], 1).uniform_(-0.5, 0.5))
            self._matrices.append(m)
            self._biases.append(b)
            if k < self._num_layers - 1:
                self._factors.append(nn.Parameter(torch.zeros(channels, sizes[k + 1], 1)))

    def _logits_cdf(self, x: torch.Tensor) -> torch.Tensor:
        # x: (channels, 1, N)
        for k in range(self._num_layers):
            x = torch.matmul(F.softplus(self._matrices[k]), x) + self._biases[k]
            if k < self._num_layers - 1:
                x = x + torch.tanh(self._factors[k]) * torch.tanh(x)
        return x

    def bits(self, v: torch.Tensor) -> torch.Tensor:
        """Total bits for values v of shape (B, channels, ...)."""
        x = v.transpose(0, 1).reshape(self._channels, 1, -1)
        lower = self._logits_cdf(x - 0.5)
        upper = self._logits_cdf(x + 0.5)
        # Numerically stable sigmoid difference (compressai trick).
        sign = -torch.sign(lower + upper).detach()
        p = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
        return -torch.log2(p.clamp_min(1e-9)).sum()
