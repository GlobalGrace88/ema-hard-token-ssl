"""
Masking for student views: patch-based, AAMM (Anatomy-Aware Multi-Zone), and legacy types.

AAMM is ported from ``MAE_v3/medical_mim/common/masking`` (same API as MAE/BYOL pretrain).
Set ``mask_type: aamm`` under ``student.*.masking`` and use the same keys as ``masking:`` in
``pretrain_mae_byol_vit3d_monai_aligned.yaml`` (``zone_probs``, ``boundary_width``, etc.).
"""

from __future__ import annotations

import random
from typing import Any, Dict, Literal, Optional, Tuple

import torch

from dinomim_pytorch.masking.aamm.apply import apply_aamm_mask

MaskType = Literal[
    "random_patch",
    "block_patch",
    "grid_patch",
    "aamm",
    "anatomy_aware",
    "anatomy_aware_optional",
    "none",
]

_AAMM_ALIASES = frozenset({"aamm", "anatomy_aware", "anatomy_aware_optional"})


def apply_masking(
    x: torch.Tensor,
    mask_type: str,
    mask_ratio: float,
    patch_size: int,
    mask_value: float,
    masking_cfg: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Apply mask on student views only (teacher views stay unmasked).

    For ``mask_type: aamm``, pass the full ``student.*.masking`` dict as ``masking_cfg``
    (zone_probs, use_boundary_prior, hybrid_uniform_mix, ...).
    """
    t = (mask_type or "none").lower().replace("-", "_")
    if t in ("none", ""):
        return x
    if t in _AAMM_ALIASES:
        return apply_aamm_mask(x, mask_ratio, patch_size, mask_value, masking_cfg=masking_cfg)
    if t == "random_patch":
        return random_patch_mask(x, mask_ratio, patch_size, mask_value)
    if t == "block_patch":
        return block_patch_mask(x, mask_ratio, patch_size, mask_value)
    if t == "grid_patch":
        return grid_patch_mask(x, mask_ratio, patch_size, mask_value)
    raise ValueError(f"Unknown mask_type {mask_type!r}")


def _is_volume_cdhw(x: torch.Tensor) -> bool:
    """Single 3D volume ``[C,D,H,W]`` (C<=8), not 2D batch ``[B,C,H,W]``."""
    return x.dim() == 4 and int(x.shape[0]) <= 8


def random_patch_mask_volume_cdhw(
    x: torch.Tensor, mask_ratio: float, patch_size: int, fill_value: float
) -> torch.Tensor:
    """Mask random 3D patches on ``[C,D,H,W]``."""
    if x.dim() != 4:
        return random_patch_mask(x, mask_ratio, patch_size, fill_value)
    c, d, h, w = x.shape
    out = x.clone()
    pd = min(patch_size, d)
    ph = min(patch_size, h)
    pw = min(patch_size, w)
    n = max(1, int((d * h * w) // max(1, pd * ph * pw) * mask_ratio + 0.5))
    for _ in range(n):
        z0 = random.randint(0, d - pd)
        y0 = random.randint(0, h - ph)
        x0 = random.randint(0, w - pw)
        out[:, z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw] = fill_value
    return out


def random_patch_mask(
    x: torch.Tensor, mask_ratio: float, patch_size: int, fill_value: float
) -> torch.Tensor:
    if x.dim() == 4 and _is_volume_cdhw(x):
        return random_patch_mask_volume_cdhw(x, mask_ratio, patch_size, fill_value)
    if x.dim() == 4:
        b, c, h, w = x.shape
        out = x.clone()
        n_h = max(1, h // patch_size)
        n_w = max(1, w // patch_size)
        num_p = max(1, int(n_h * n_w * mask_ratio))
        for i in range(b):
            for _ in range(num_p):
                ph = min(patch_size, h)
                pw = min(patch_size, w)
                y0 = random.randint(0, h - ph)
                x0 = random.randint(0, w - pw)
                out[i, :, y0 : y0 + ph, x0 : x0 + pw] = fill_value
        return out
    if x.dim() == 5:
        b, c, d, h, w = x.shape
        y = x.clone()
        for i in range(b):
            pd = min(patch_size, d)
            ph = min(patch_size, h)
            pw = min(patch_size, w)
            n = max(1, int((d * h * w) // (pd * ph * pw) * mask_ratio + 0.5))
            for _ in range(n):
                z0 = random.randint(0, d - pd)
                y0 = random.randint(0, h - ph)
                x0 = random.randint(0, w - pw)
                y[i, :, z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw] = fill_value
        return y
    if x.dim() == 3:
        return random_patch_mask(x.unsqueeze(0), mask_ratio, patch_size, fill_value).squeeze(0)
    return x


def block_patch_mask(
    x: torch.Tensor, mask_ratio: float, patch_size: int, fill_value: float
) -> torch.Tensor:
    if x.dim() != 4:
        return random_patch_mask(x, mask_ratio, patch_size, fill_value)
    b, c, h, w = x.shape
    out = x.clone()
    n_h = max(1, h // patch_size)
    n_w = max(1, w // patch_size)
    n_p = max(1, int(n_h * n_w * mask_ratio))
    side_h = max(1, int(n_h * (mask_ratio**0.5)))
    side_w = max(1, n_p // side_h)
    for i in range(b):
        y0p = random.randint(0, max(0, n_h - side_h))
        x0p = random.randint(0, max(0, n_w - side_w))
        for ph_i in range(side_h):
            for pw_i in range(side_w):
                y0 = (y0p + ph_i) * patch_size
                x0 = (x0p + pw_i) * patch_size
                y0 = min(y0, h - patch_size)
                x0 = min(x0, w - patch_size)
                out[i, :, y0 : y0 + patch_size, x0 : x0 + patch_size] = fill_value
    return out


def grid_patch_mask(
    x: torch.Tensor, mask_ratio: float, patch_size: int, fill_value: float
) -> torch.Tensor:
    if x.dim() != 4:
        return random_patch_mask(x, mask_ratio, patch_size, fill_value)
    b, c, h, w = x.shape
    out = x.clone()
    n_h, n_w = max(1, h // patch_size), max(1, w // patch_size)
    flat = [(i, j) for i in range(n_h) for j in range(n_w)]
    k = max(1, int(len(flat) * mask_ratio))
    for i in range(b):
        pick = random.sample(flat, min(k, len(flat)))
        for (gi, gj) in pick:
            y0, x0 = gi * patch_size, gj * patch_size
            out[i, :, y0 : y0 + patch_size, x0 : x0 + patch_size] = fill_value
    return out


def apply_view_masking(x: torch.Tensor, masking: Optional[Dict[str, Any]]) -> torch.Tensor:
    """Apply ``student.*.masking`` / ``teacher.*.masking`` block from YAML."""
    m = masking or {}
    if not m.get("enabled"):
        return x
    if x.dim() == 4 and _is_volume_cdhw(x):
        default_ps = 8
    else:
        default_ps = 16 if x.dim() == 3 else 8
    return apply_masking(
        x,
        str(m.get("mask_type", "random_patch")),
        float(m.get("mask_ratio", 0.2)),
        int(m.get("patch_size", default_ps)),
        float(m.get("mask_value", 0.0)),
        masking_cfg=m,
    )


def mark_masked_view_indices(
    num_views: int, num_masked: int, mode: str = "first"
) -> Tuple[int, ...]:
    n = int(num_views)
    m = min(int(num_masked), n)
    if m <= 0:
        return tuple()
    if mode == "first":
        return tuple(range(m))
    if mode == "last":
        return tuple(range(n - m, n))
    if mode == "random":
        return tuple(sorted(random.sample(range(n), m)))
    return tuple(range(m))


__all__ = [
    "apply_masking",
    "apply_view_masking",
    "apply_aamm_mask",
    "random_patch_mask",
    "block_patch_mask",
    "grid_patch_mask",
    "mark_masked_view_indices",
    "MaskType",
]
