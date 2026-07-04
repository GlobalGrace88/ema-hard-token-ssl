"""
3D volume augmentations for DINO SSL (teacher weak vs student strong).

2D pretrain uses ``MedicalWeakGlobalAug`` / ``MedicalStrongAug`` on [C,H,W].
Volume SSL uses anisotropic random crops on [C,D,H,W] plus flips and intensity jitter.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _is_volume_cdhw(x: torch.Tensor) -> bool:
    """``[C,D,H,W]`` with few channels (not batched 2D ``[B,C,H,W]``)."""
    return x.dim() == 4 and int(x.shape[0]) <= 8


def sample_anisotropic_crop_resize(
    x: torch.Tensor,
    target_dhw: Tuple[int, int, int],
    scale: Tuple[float, float],
    *,
    max_scale_when_fits: Optional[float] = None,
) -> Tuple[torch.Tensor, Tuple[int, int, int, int, int, int]]:
    """
    Random axis-aligned crop with independent scale per axis, trilinear resize to ``target_dhw``.

    ``scale`` applies to each spatial axis: side_axis = uniform(scale) * D/H/W.
    When the source volume is already near ``target_dhw``, pass ``max_scale_when_fits`` (e.g. 0.85)
    so crops are strict sub-windows (critical for nnFormer npz that are pre-cropped).
    """
    if x.dim() != 4:
        raise ValueError(f"Expected [C,D,H,W], got {tuple(x.shape)}")
    c, d, h, w = x.shape
    lo, hi = float(scale[0]), float(scale[1])
    td, th, tw = int(target_dhw[0]), int(target_dhw[1]), int(target_dhw[2])

    fits_target = d <= td and h <= th and w <= tw
    cap = float(max_scale_when_fits) if max_scale_when_fits is not None else None
    if fits_target and cap is not None:
        hi = min(hi, cap)

    sd = max(1, min(d, int(random.uniform(lo, hi) * d)))
    sh = max(1, min(h, int(random.uniform(lo, hi) * h)))
    sw = max(1, min(w, int(random.uniform(lo, hi) * w)))
    z0 = random.randint(0, max(0, d - sd))
    y0 = random.randint(0, max(0, h - sh))
    x0 = random.randint(0, max(0, w - sw))
    patch = x[:, z0 : z0 + sd, y0 : y0 + sh, x0 : x0 + sw]
    if tuple(patch.shape[-3:]) != (td, th, tw):
        patch = F.interpolate(
            patch.unsqueeze(0),
            size=(td, th, tw),
            mode="trilinear",
            align_corners=False,
        )[0]
    return patch, (z0, y0, x0, sd, sh, sw)


class MedicalWeakGlobalAug3D(nn.Module):
    """Teacher: moderate sub-volume crop, light noise, optional axis flips."""

    def __init__(
        self,
        target_dhw: Tuple[int, int, int],
        scale: Tuple[float, float] = (0.5, 0.9),
        noise_std: float = 0.02,
        flip_prob: float = 0.5,
        max_scale_when_fits: float = 0.92,
    ):
        super().__init__()
        self.target_dhw = tuple(int(v) for v in target_dhw)
        self.scale = scale
        self.noise_std = float(noise_std)
        self.flip_prob = float(flip_prob)
        self.max_scale_when_fits = float(max_scale_when_fits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not _is_volume_cdhw(x):
            raise ValueError(f"MedicalWeakGlobalAug3D expects [C,D,H,W], got {tuple(x.shape)}")
        v, _ = sample_anisotropic_crop_resize(
            x,
            self.target_dhw,
            self.scale,
            max_scale_when_fits=self.max_scale_when_fits,
        )
        if self.noise_std > 0:
            v = v + self.noise_std * torch.randn_like(v)
        if random.random() < self.flip_prob:
            v = torch.flip(v, (2,))
        if random.random() < self.flip_prob:
            v = torch.flip(v, (3,))
        if random.random() < self.flip_prob:
            v = torch.flip(v, (1,))
        return v.clamp(-6.0, 6.0)


class MedicalStrongAug3D(nn.Module):
    """Student: smaller crops, stronger noise, contrast jitter, more flips."""

    def __init__(
        self,
        target_dhw: Tuple[int, int, int],
        scale: Tuple[float, float] = (0.35, 0.75),
        noise_std: float = 0.08,
        flip_prob: float = 0.5,
        gamma_jitter: float = 0.15,
        max_scale_when_fits: float = 0.8,
    ):
        super().__init__()
        self.target_dhw = tuple(int(v) for v in target_dhw)
        self.scale = scale
        self.noise_std = float(noise_std)
        self.flip_prob = float(flip_prob)
        self.gamma_jitter = float(gamma_jitter)
        self.max_scale_when_fits = float(max_scale_when_fits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not _is_volume_cdhw(x):
            raise ValueError(f"MedicalStrongAug3D expects [C,D,H,W], got {tuple(x.shape)}")
        v, _ = sample_anisotropic_crop_resize(
            x,
            self.target_dhw,
            self.scale,
            max_scale_when_fits=self.max_scale_when_fits,
        )
        if self.gamma_jitter > 0:
            g = 1.0 + random.uniform(-self.gamma_jitter, self.gamma_jitter)
            v = torch.sign(v) * (v.abs() + 1e-6).pow(g)
        if self.noise_std > 0:
            v = v + self.noise_std * torch.randn_like(v)
        if random.random() < self.flip_prob:
            v = torch.flip(v, (2,))
        if random.random() < self.flip_prob:
            v = torch.flip(v, (3,))
        if random.random() < self.flip_prob:
            v = torch.flip(v, (1,))
        return v.clamp(-6.0, 6.0)


def build_volume_view_augmentor(
    section: dict,
    target_dhw: Tuple[int, int, int],
    *,
    default_strength: str = "weak",
) -> nn.Module:
    """
    Build 3D augmentor from a teacher/student branch dict.

    Honors ``augmentation_strength``: weak | strong (default weak for teacher).
    ``crop_scale`` and ``noise_std`` override defaults when set.
    """
    strength = str(section.get("augmentation_strength", default_strength)).lower()
    scale = tuple(section.get("crop_scale", (0.5, 0.9) if strength == "weak" else (0.35, 0.75)))
    noise = float(section.get("noise_std", 0.02 if strength == "weak" else 0.08))
    max_fit = float(section.get("max_scale_when_fits", 0.92 if strength == "weak" else 0.8))
    if strength in ("strong", "student"):
        return MedicalStrongAug3D(
            target_dhw,
            scale=scale,
            noise_std=noise,
            max_scale_when_fits=max_fit,
        )
    return MedicalWeakGlobalAug3D(
        target_dhw,
        scale=scale,
        noise_std=noise,
        max_scale_when_fits=max_fit,
    )


__all__ = [
    "MedicalWeakGlobalAug3D",
    "MedicalStrongAug3D",
    "sample_anisotropic_crop_resize",
    "build_volume_view_augmentor",
    "_is_volume_cdhw",
]
