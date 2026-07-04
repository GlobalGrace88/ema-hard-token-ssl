"""Local 3D Swin-UNet (attention bottleneck). Used when MONAI has no exact Swin-UNet match."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from dinomim_pytorch.segmentation_models.local_nnformer3d import _Block, _BottleneckAttn2


class LocalSwinUnet3D(nn.Module):
    def __init__(self, c_in: int, c_out: int, f0: int = 32) -> None:
        super().__init__()
        f1, f2 = f0 * 2, f0 * 4
        self.d1, self.d2, self.d3 = _Block(c_in, f0), _Block(f0, f1), _Block(f1, f2)
        self.p1, self.p2, self.p3 = nn.MaxPool3d(2), nn.MaxPool3d(2), nn.MaxPool3d(2)
        self.bot = _BottleneckAttn2(f2)
        self.u3, self.u2, self.u1 = (
            nn.ConvTranspose3d(f2, f2, 2, 2),
            nn.ConvTranspose3d(f1, f1, 2, 2),
            nn.ConvTranspose3d(f0, f0, 2, 2),
        )
        self.s3, self.s2, self.s1 = _Block(2 * f2, f1), _Block(2 * f1, f0), _Block(2 * f0, f0)
        self.out = nn.Conv3d(f0, c_out, 1)

    @staticmethod
    def _c(t: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return t[:, :, : r.size(2), : r.size(3), : r.size(4)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a1 = self.d1(x)
        a2 = self.d2(self.p1(a1))
        a3 = self.d3(self.p2(a2))
        t2 = self.bot(self.p3(a3))
        u2 = self._c(self.u3(t2), a3)
        s2 = self.s3(torch.cat([u2, a3], 1))
        u1 = self._c(self.u2(s2), a2)
        s1 = self.s2(torch.cat([u1, a2], 1))
        u0 = self._c(self.u1(s1), a1)
        o = self.s1(torch.cat([u0, a1], 1))
        return self.out(o)


def build_local_swinunet3d(c: Dict[str, Any]) -> nn.Module:
    return LocalSwinUnet3D(
        int(c.get("in_channels", 1)),
        int(c.get("out_channels", 2)),
        int(c.get("feature_size", 32)),
    )
