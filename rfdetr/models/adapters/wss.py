"""Wavelet-Selective Shrinkage (WSS) Adapter.

Inserted between frozen ViT blocks, WSS performs fixed Haar wavelet
decomposition and applies statistics-driven soft-shrinkage to suppress
noise-like high-frequency responses while preserving structured edges.

Design goals:
  - Fixed wavelet filters (no learned spatial kernels)
  - Lightweight learnable thresholds conditioned on simple stats
  - Residual injection with per-layer learnable scale initialized to 0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WSSConfig:
    """Configuration for :class:`WSSAdapter2D`."""

    levels: int = 1
    reduction_dim: int = 64
    detach_stats: bool = True
    eps: float = 1e-6


def _make_haar_kernels(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return 4 Haar 2D kernels: LL, LH, HL, HH. Shape [4, 1, 2, 2]."""
    s = 0.5 ** 0.5
    low = torch.tensor([s, s], device=device, dtype=dtype)
    high = torch.tensor([s, -s], device=device, dtype=dtype)
    ll = torch.einsum("i,j->ij", low, low)
    lh = torch.einsum("i,j->ij", low, high)
    hl = torch.einsum("i,j->ij", high, low)
    hh = torch.einsum("i,j->ij", high, high)
    ker = torch.stack([ll, lh, hl, hh], dim=0)  # [4,2,2]
    return ker.unsqueeze(1)  # [4,1,2,2]


class _HaarDWT(nn.Module):
    """Fixed Haar DWT/iDWT implemented with grouped convs."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def dwt(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One-level 2D Haar DWT.

        Args:
            x: [B, C, H, W] (H and W must be even)

        Returns:
            (ll, lh, hl, hh): each [B, C, H/2, W/2]
        """
        B, C, H, W = x.shape
        if (H % 2) != 0 or (W % 2) != 0:
            # pad to even (right/bottom)
            x = F.pad(x, (0, W % 2, 0, H % 2))
            B, C, H, W = x.shape

        ker = _make_haar_kernels(x.device, x.dtype)  # [4,1,2,2]
        weight = ker.repeat(C, 1, 1, 1)  # [4C,1,2,2]
        y = F.conv2d(x, weight=weight, bias=None, stride=2, padding=0, groups=C)  # [B,4C,H/2,W/2]
        y = y.view(B, C, 4, H // 2, W // 2)
        ll, lh, hl, hh = y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]
        return ll, lh, hl, hh

    @staticmethod
    def idwt(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        """One-level inverse 2D Haar DWT.

        Args:
            ll/lh/hl/hh: [B, C, H/2, W/2]
            out_hw: (H, W) to crop back (handles padding)

        Returns:
            x: [B, C, H, W]
        """
        B, C, h2, w2 = ll.shape
        y = torch.stack([ll, lh, hl, hh], dim=2).view(B, 4 * C, h2, w2)
        ker = _make_haar_kernels(ll.device, ll.dtype)  # [4,1,2,2]
        weight = ker.repeat(C, 1, 1, 1)  # [4C,1,2,2]
        x = F.conv_transpose2d(y, weight=weight, bias=None, stride=2, padding=0, groups=C)
        H, W = out_hw
        return x[..., :H, :W]


class _Stats2Tau(nn.Module):
    """Map (SFM, Kurtosis) -> per-band shrink threshold."""

    def __init__(self):
        super().__init__()
        # Learned scalars (per band) controlling how stats modulate the threshold.
        # tau = softplus(base) * sigmoid(a*SFM - b*Kurt + c)
        self.base = nn.Parameter(torch.tensor(0.1))
        self.a = nn.Parameter(torch.tensor(2.0))
        self.b = nn.Parameter(torch.tensor(1.0))
        self.c = nn.Parameter(torch.tensor(0.0))

    def forward(self, sfm: torch.Tensor, kurt: torch.Tensor) -> torch.Tensor:
        base = F.softplus(self.base)  # >0
        gate = torch.sigmoid(self.a * sfm - self.b * kurt + self.c)
        return base * gate


class WSSAdapter2D(nn.Module):
    """Wavelet-Selective Shrinkage adapter on 2D feature maps.

    Input/Output: [B, C, H, W]
    """

    def __init__(self, channels: int, cfg: WSSConfig | None = None):
        super().__init__()
        self.cfg = cfg or WSSConfig()
        self.channels = int(channels)
        self.levels = int(self.cfg.levels)
        assert self.levels >= 1, "levels must be >= 1"

        d = int(self.cfg.reduction_dim)
        self.use_bottleneck = d > 0 and d < self.channels
        if self.use_bottleneck:
            self.down = nn.Conv2d(self.channels, d, kernel_size=1, bias=False)
            self.up = nn.Conv2d(d, self.channels, kernel_size=1, bias=False)
            proc_c = d
        else:
            self.down = None
            self.up = None
            proc_c = self.channels

        # per-level, per-band stats->tau
        self.tau_lh = nn.ModuleList([_Stats2Tau() for _ in range(self.levels)])
        self.tau_hl = nn.ModuleList([_Stats2Tau() for _ in range(self.levels)])
        self.tau_hh = nn.ModuleList([_Stats2Tau() for _ in range(self.levels)])

        self._dwt = _HaarDWT()
        self.scale = nn.Parameter(torch.tensor(0.0))  # initialized to 0

        # small projection after reconstruction to keep adapter expressive
        self.proj = nn.Sequential(
            nn.Conv2d(proc_c, proc_c, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(proc_c, proc_c, kernel_size=1, bias=False),
        )

    def _sfm_and_kurt(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute spectral flatness (SFM) and kurtosis over (C,H,W) as scalars per sample."""
        eps = self.cfg.eps
        xa = x.abs() + eps
        # flatten over channel+spatial
        flat = xa.view(x.shape[0], -1)
        sfm = torch.exp(torch.mean(torch.log(flat), dim=1)) / (torch.mean(flat, dim=1) + eps)  # [B]
        # kurtosis-like: E[x^4] / (E[x^2]^2)
        x2 = (x ** 2).view(x.shape[0], -1)
        e2 = torch.mean(x2, dim=1) + eps
        e4 = torch.mean(x2 ** 2, dim=1)
        kurt = e4 / (e2 ** 2)
        return sfm, kurt

    def _shrink_band(self, band: torch.Tensor, tau_mod: _Stats2Tau) -> torch.Tensor:
        sfm, kurt = self._sfm_and_kurt(band)
        if self.cfg.detach_stats:
            sfm = sfm.detach()
            kurt = kurt.detach()
        tau = tau_mod(sfm, kurt).view(-1, 1, 1, 1)  # [B,1,1,1]
        # soft thresholding
        return torch.sign(band) * F.relu(band.abs() - tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply WSS and inject as residual.

        Args:
            x: [B, C, H, W]
        Returns:
            y: [B, C, H, W]
        """
        x_in = x
        if self.use_bottleneck:
            x = self.down(x)

        # multi-level DWT
        ll_stack = []
        lh_stack = []
        hl_stack = []
        hh_stack = []
        hw_stack = []

        cur = x
        for lv in range(self.levels):
            hw_stack.append((cur.shape[-2], cur.shape[-1]))
            ll, lh, hl, hh = self._dwt.dwt(cur)
            lh = self._shrink_band(lh, self.tau_lh[lv])
            hl = self._shrink_band(hl, self.tau_hl[lv])
            hh = self._shrink_band(hh, self.tau_hh[lv])
            ll_stack.append(ll)
            lh_stack.append(lh)
            hl_stack.append(hl)
            hh_stack.append(hh)
            cur = ll

        # reconstruct
        rec = cur
        for lv in reversed(range(self.levels)):
            rec = self._dwt.idwt(rec, lh_stack[lv], hl_stack[lv], hh_stack[lv], out_hw=hw_stack[lv])

        rec = self.proj(rec)
        if self.use_bottleneck:
            rec = self.up(rec)

        # residual injection
        return x_in + self.scale * (rec - x_in)
