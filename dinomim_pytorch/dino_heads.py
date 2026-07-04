"""
DINO projection head: MLP + L2 norm + weight-norm final layer (see local reference ``utils.Head``).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        n_layers: int = 3,
        use_bn_in_head: bool = False,
        norm_last_layer: bool = True,
    ):
        super().__init__()
        if n_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers: list = []
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_bn_in_head:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn_in_head:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


def infer_in_dim_from_backbone(
    backbone: nn.Module,
    sample: torch.Tensor,
    feature_dim_override: Optional[int] = None,
) -> int:
    if feature_dim_override is not None and feature_dim_override > 0:
        return int(feature_dim_override)
    with torch.no_grad():
        out = backbone(sample)
    if isinstance(out, (list, tuple)):
        out = out[-1]
    if out.dim() == 4:
        return int(out.shape[1])
    if out.dim() == 2:
        return int(out.shape[1])
    if out.dim() == 5:
        return int(out.shape[1])
    raise RuntimeError(f"Cannot infer feature dim from output shape {tuple(out.shape)}")


__all__ = ["DINOHead", "infer_in_dim_from_backbone"]
