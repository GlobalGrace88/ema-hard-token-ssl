"""Local 3D SegResNet-style network (residual blocks) when MONAI SegResNet is unavailable."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


class _ResBlock3d(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(c_in, c_out, 3, padding=1, bias=False)
        self.n1 = nn.InstanceNorm3d(c_out, affine=True)
        self.conv2 = nn.Conv3d(c_out, c_out, 3, padding=1, bias=False)
        self.n2 = nn.InstanceNorm3d(c_out, affine=True)
        self.act = nn.LeakyReLU(0.1, True)
        self.skip = (
            nn.Conv3d(c_in, c_out, 1, bias=False)
            if c_in != c_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.skip(x)
        y = self.act(self.n1(self.conv1(x)))
        y = self.n2(self.conv2(y))
        return self.act(y + s)


class LocalSegResNet3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, f0: int = 32) -> None:
        super().__init__()
        c1, c2, c3 = f0, f0 * 2, f0 * 4
        self.d1 = nn.Sequential(_ResBlock3d(in_ch, c1), _ResBlock3d(c1, c1))
        self.d2 = nn.Sequential(nn.MaxPool3d(2), _ResBlock3d(c1, c2), _ResBlock3d(c2, c2))
        self.d3 = nn.Sequential(nn.MaxPool3d(2), _ResBlock3d(c2, c3), _ResBlock3d(c3, c3))
        self.p4 = nn.MaxPool3d(2)
        self.bot = _ResBlock3d(c3, c3)
        self.u3 = nn.ConvTranspose3d(c3, c3, 2, stride=2)
        self.s3 = _ResBlock3d(c3 + c3, c2)
        self.u2 = nn.ConvTranspose3d(c2, c2, 2, stride=2)
        self.s2 = _ResBlock3d(c2 + c2, c1)
        self.u1 = nn.ConvTranspose3d(c1, c1, 2, stride=2)
        self.s1 = _ResBlock3d(c1 + c1, c1)
        self.out = nn.Conv3d(c1, out_ch, 1)

    @staticmethod
    def _crop(t: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return t[:, :, : r.size(2), : r.size(3), : r.size(4)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.d1(x)
        e2 = self.d2(e1)
        e3 = self.d3(e2)
        t0 = self.bot(self.p4(e3))
        u3 = self._crop(self.u3(t0), e3)
        s3 = self.s3(torch.cat([u3, e3], 1))
        u2 = self._crop(self.u2(s3), e2)
        s2 = self.s2(torch.cat([u2, e2], 1))
        u1 = self._crop(self.u1(s2), e1)
        s1 = self.s1(torch.cat([u1, e1], 1))
        return self.out(s1)


def build_local_segresnet3d(c: Dict[str, Any]) -> nn.Module:
    return LocalSegResNet3D(
        int(c.get("in_channels", 1)),
        int(c.get("out_channels", 2)),
        f0=int(c.get("feature_size", 32)),
    )
