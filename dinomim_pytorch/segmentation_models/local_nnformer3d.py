"""Local 3D nnFormer-style U-Net + attention bottleneck. [B,C,D,H,W] -> [B,K,D,H,W]."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


class _Block(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv3d(c_in, c_out, 3, padding=1, bias=False),
            nn.InstanceNorm3d(c_out, affine=True),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(c_out, c_out, 3, padding=1, bias=False),
            nn.InstanceNorm3d(c_out, affine=True),
            nn.LeakyReLU(0.1, True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c(x)


class _BottleneckAttn2(nn.Module):
    def __init__(self, c: int, heads: int = 4) -> None:
        super().__init__()
        _ = heads
        self.mha = nn.MultiheadAttention(c, max(1, c // 64) or 1, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        t = x.view(b, c, -1).permute(0, 2, 1)
        a, _ = self.mha(t, t, t, need_weights=False)
        t = t + a
        return t.permute(0, 2, 1).contiguous().view(b, c, d, h, w)


class LocalNNFormer3D(nn.Module):
    """Registry name: LocalnnFormer3D (class LocalNNFormer3D)."""

    def __init__(self, c_in: int, c_out: int, f0: int = 32) -> None:
        super().__init__()
        c1, c2, c3 = f0, f0 * 2, f0 * 4
        self.d1, self.d2, self.d3 = _Block(c_in, c1), _Block(c1, c2), _Block(c2, c3)
        self.p1, self.p2, self.p3 = nn.MaxPool3d(2), nn.MaxPool3d(2), nn.MaxPool3d(2)
        self.bot = _BottleneckAttn2(c3)
        self.u3, self.u2, self.u1 = (
            nn.ConvTranspose3d(c3, c3, 2, 2),
            nn.ConvTranspose3d(c2, c2, 2, 2),
            nn.ConvTranspose3d(c1, c1, 2, 2),
        )
        self.s3, self.s2, self.s1 = _Block(2 * c3, c2), _Block(2 * c2, c1), _Block(2 * c1, c1)
        self.out = nn.Conv3d(c1, c_out, 1)

    @staticmethod
    def _c(t: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return t[:, :, : r.size(2), : r.size(3), : r.size(4)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.d1(x)
        e2 = self.d2(self.p1(e1))
        e3 = self.d3(self.p2(e2))
        t0 = self.p3(e3)
        t = self.bot(t0)
        u3 = self._c(self.u3(t), e3)
        s3 = self.s3(torch.cat([u3, e3], 1))
        u2 = self._c(self.u2(s3), e2)
        s2 = self.s2(torch.cat([u2, e2], 1))
        u1 = self._c(self.u1(s2), e1)
        s1 = self.s1(torch.cat([u1, e1], 1))
        return self.out(s1)


def build_local_nnformer3d(c: Dict[str, Any]) -> nn.Module:
    c_in, c_out = int(c.get("in_channels", 1)), int(c.get("out_channels", 2))
    f0 = int(c.get("feature_size", 32))
    return LocalNNFormer3D(c_in, c_out, f0)
