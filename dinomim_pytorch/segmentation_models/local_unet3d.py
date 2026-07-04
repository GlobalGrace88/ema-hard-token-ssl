"""Local 3D U-Net (fallback when MONAI UNet is unavailable). Input [B,C,D,H,W] -> logits [B,K,D,H,W]."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


def _conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(c_in, c_out, 3, padding=1, bias=False),
        nn.InstanceNorm3d(c_out, affine=True),
        nn.LeakyReLU(0.1, True),
        nn.Conv3d(c_out, c_out, 3, padding=1, bias=False),
        nn.InstanceNorm3d(c_out, affine=True),
        nn.LeakyReLU(0.1, True),
    )


class LocalUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        feature_size: int = 32,
        levels: int = 4,
    ) -> None:
        super().__init__()
        assert levels >= 2
        ch: list[int] = [feature_size * (2**i) for i in range(levels)]
        self.down = nn.ModuleList()
        self.pool = nn.ModuleList()
        prev = in_channels
        for c in ch:
            self.down.append(_conv_block(prev, c))
            self.pool.append(nn.MaxPool3d(2))
            prev = c
        self.mid = _conv_block(prev, prev)
        self.up = nn.ModuleList()
        self.up_t = nn.ModuleList()
        for i in range(levels - 1, 0, -1):
            c_hi, c_lo = ch[i], ch[i - 1]
            self.up_t.append(nn.ConvTranspose3d(c_hi, c_hi, 2, stride=2))
            self.up.append(_conv_block(c_hi + c_lo, c_lo))
        self.out = nn.Conv3d(ch[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for i, (block, pool) in enumerate(zip(self.down, self.pool, strict=True)):
            x = block(x)
            skips.append(x)
            if i < len(self.pool) - 1:
                x = pool(x)
        x = self.mid(x)
        for j, (up_t, block) in enumerate(zip(self.up_t, self.up, strict=True)):
            sk = skips[-(j + 2)]
            x = up_t(x)
            if x.shape[2:] != sk.shape[2:]:
                x = x[:, :, : sk.size(2), : sk.size(3), : sk.size(4)]
            x = torch.cat([x, sk], dim=1)
            x = block(x)
        return self.out(x)


def build_local_unet3d(c: Dict[str, Any]) -> nn.Module:
    return LocalUNet3D(
        int(c.get("in_channels", 1)),
        int(c.get("out_channels", 2)),
        feature_size=int(c.get("feature_size", 32)),
        levels=int(c.get("levels", 4)),
    )
