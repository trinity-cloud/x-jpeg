"""The decisive correctness check for both soft-entropy models: when fitted
to data from a KNOWN distribution, the average bits they report must converge
to that distribution's true discrete entropy."""

import math

import torch

from xjpeg.entropy_bottleneck import FactorizedEntropyModel
from xjpeg.rate import DctRateModel


def test_factorized_model_recovers_uniform_entropy():
    torch.manual_seed(0)
    m = FactorizedEntropyModel(channels=1, init_scale=30.0)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    # integers uniform on [-20, 20]: true entropy = log2(41) ~ 5.358 bits
    for _ in range(500):
        v = torch.randint(-20, 21, (4, 1, 1024)).float()
        noisy = v + torch.empty_like(v).uniform_(-0.5, 0.5)
        loss = m.bits(noisy) / v.numel()
        opt.zero_grad()
        loss.backward()
        opt.step()
    v = torch.randint(-20, 21, (4, 1, 4096)).float()
    bits = (m.bits(v) / v.numel()).item()
    true = math.log2(41)
    assert abs(bits - true) < 0.35, f"model says {bits:.3f} bits, truth {true:.3f}"


def test_factorized_model_positive_shifted_support():
    """Q-tables live in [1, 255], far from zero: the model must fit an
    off-center distribution too (this is exactly our use case)."""
    torch.manual_seed(1)
    m = FactorizedEntropyModel(channels=1, init_scale=64.0)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    # integers uniform on [10, 73]: entropy = log2(64) = 6 bits
    for _ in range(600):
        v = torch.randint(10, 74, (4, 1, 1024)).float()
        noisy = v + torch.empty_like(v).uniform_(-0.5, 0.5)
        loss = m.bits(noisy) / v.numel()
        opt.zero_grad()
        loss.backward()
        opt.step()
    v = torch.randint(10, 74, (4, 1, 4096)).float()
    bits = (m.bits(v) / v.numel()).item()
    assert abs(bits - 6.0) < 0.4, f"model says {bits:.3f} bits, truth 6.0"


def test_dct_rate_model_recovers_laplacian_entropy():
    torch.manual_seed(0)
    m = DctRateModel()
    opt = torch.optim.Adam(m.parameters(), lr=5e-2)
    b_true = 3.0
    dist = torch.distributions.Laplace(0.0, b_true)
    m.train()
    for _ in range(300):
        s = dist.sample((8, 1, 4, 4, 8, 8))
        loss = m.bits(s, 0) / s.numel()
        opt.zero_grad()
        loss.backward()
        opt.step()
    # true entropy of the integer-rounded Laplace(0, 3)
    ks = torch.arange(-400, 401).float()
    p = dist.cdf(ks + 0.5) - dist.cdf(ks - 0.5)
    true = -(p * torch.log2(p.clamp_min(1e-12))).sum().item()
    m.eval()  # eval path hard-rounds, matching the discretized entropy
    s = dist.sample((8, 1, 8, 8, 8, 8))
    bits = (m.bits(s, 0) / s.numel()).item()
    assert abs(bits - true) < 0.2, f"model says {bits:.3f} bits, truth {true:.3f}"
