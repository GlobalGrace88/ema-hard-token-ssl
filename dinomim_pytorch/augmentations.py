"""
Per-view augmentations for medical 1-channel data: weak (teacher) vs strong (student).
"""

from __future__ import annotations

import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MedicalWeakGlobalAug(nn.Module):
    def __init__(self, size: int, scale: Tuple[float, float] = (0.6, 1.0)):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c, h, w = x.shape
        scale = float(random.uniform(self.scale[0], self.scale[1]))
        th, tw = max(1, int(h * scale)), max(1, int(w * scale))
        i = random.randint(0, max(0, h - th))
        j = random.randint(0, max(0, w - tw))
        x = x[:, i : i + th, j : j + tw]
        x = F.interpolate(
            x.unsqueeze(0), size=(self.size, self.size), mode="bilinear", align_corners=False
        )[0]
        if random.random() < 0.5:
            x = torch.flip(x, (2,))
        return x


class MedicalStrongAug(nn.Module):
    def __init__(self, size: int, scale: Tuple[float, float] = (0.2, 0.5)):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c, h, w = x.shape
        x = x + 0.08 * torch.randn_like(x)
        x = x.clamp(0, 1)
        scale = float(random.uniform(self.scale[0], self.scale[1]))
        th, tw = max(1, int(h * scale)), max(1, int(w * scale))
        i = random.randint(0, max(0, h - th))
        j = random.randint(0, max(0, w - tw))
        x = x[:, i : i + th, j : j + tw]
        x = F.interpolate(
            x.unsqueeze(0), size=(self.size, self.size), mode="bilinear", align_corners=False
        )[0]
        if random.random() < 0.5:
            x = torch.flip(x, (2,))
        if random.random() < 0.3:
            x = torch.flip(x, (1,))
        return x


__all__ = ["MedicalWeakGlobalAug", "MedicalStrongAug"]
