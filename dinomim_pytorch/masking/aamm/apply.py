"""Apply AAMM patch masks to tensors (student views for DINO)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from dinomim_pytorch.masking.aamm.multi_zone_masking import (
    sample_aamm_mask_2d,
    sample_aamm_mask_3d,
)


def aamm_cfg_from_masking(masking_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Map DINO ``student.*.masking`` block to AAMM ``cfg`` (same keys as MAE_v3 ``masking``)."""
    m = dict(masking_cfg or {})
    return {
        "use_foreground_prior": bool(m.get("use_foreground_prior", True)),
        "use_boundary_prior": bool(m.get("use_boundary_prior", True)),
        "use_distance_map": bool(m.get("use_distance_map", True)),
        "use_vesselness": bool(m.get("use_vesselness", False)),
        "foreground_threshold": m.get("foreground_threshold"),
        "foreground_soft_sigmoid": bool(m.get("foreground_soft_sigmoid", True)),
        "foreground_sigmoid_scale": float(m.get("foreground_sigmoid_scale", 0.1)),
        "boundary_threshold": float(m.get("boundary_threshold", 0.5)),
        "boundary_smooth_sigma": float(m.get("boundary_smooth_sigma", 0.0)),
        "boundary_width": int(m.get("boundary_width", 0)),
        "distance_power": float(m.get("distance_power", 1.0)),
        "zone_probs": dict(m.get("zone_probs") or {"interior": 0.6, "boundary": 0.9, "context": 0.3}),
        "hybrid_uniform_mix": float(m.get("hybrid_uniform_mix", 0.0)),
    }


def _upsample_patch_mask_3d(mask_3d: torch.Tensor, d: int, h: int, w: int, patch_size: int) -> torch.Tensor:
    gd, gh, gw = mask_3d.shape
    m = mask_3d.float().view(1, 1, gd, gh, gw)
    m = torch.nn.functional.interpolate(m, size=(d, h, w), mode="nearest")
    return (m.squeeze(0).squeeze(0) > 0.5)


def _upsample_patch_mask_2d(mask_2d: torch.Tensor, h: int, w: int, patch_size: int) -> torch.Tensor:
    gh, gw = mask_2d.shape
    m = mask_2d.float().view(1, 1, gh, gw)
    m = torch.nn.functional.interpolate(m, size=(h, w), mode="nearest")
    return (m.squeeze(0).squeeze(0) > 0.5)


def apply_aamm_mask(
    x: torch.Tensor,
    mask_ratio: float,
    patch_size: int,
    fill_value: float,
    masking_cfg: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Zone-conditioned patch mask (MAE_v3 AAMM) on a single view.

    ``x``: ``[C,H,W]``, ``[C,D,H,W]``, or batched ``[B,C,H,W]`` / ``[B,C,D,H,W]``.
    Spatial sizes must be divisible by ``patch_size``.
    """
    cfg = aamm_cfg_from_masking(masking_cfg)
    device = x.device

    if x.dim() == 3:
        c, h, w = x.shape
        if h % patch_size or w % patch_size:
            raise ValueError(
                f"AAMM requires H,W divisible by patch_size={patch_size}, got H={h}, W={w}"
            )
        _flat, mask_2d, _ = sample_aamm_mask_2d(
            x.unsqueeze(0),
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            cfg=cfg,
            device=device,
        )
        m = _upsample_patch_mask_2d(mask_2d, h, w, patch_size)
        out = x.clone()
        out[:, m] = fill_value
        return out

    if x.dim() == 4:
        # Medical 3D view: [C, D, H, W] (channel-first)
        c, d, h, w = x.shape
        if d % patch_size or h % patch_size or w % patch_size:
            raise ValueError(
                f"AAMM requires D,H,W divisible by patch_size={patch_size}, "
                f"got D={d}, H={h}, W={w}"
            )
        _flat, mask_3d, _ = sample_aamm_mask_3d(
            x.unsqueeze(0),
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            cfg=cfg,
            device=device,
        )
        m = _upsample_patch_mask_3d(mask_3d, d, h, w, patch_size)
        out = x.clone()
        out[:, m] = fill_value
        return out

    if x.dim() == 5:
        out = x.clone()
        for i in range(x.shape[0]):
            out[i] = apply_aamm_mask(
                x[i], mask_ratio, patch_size, fill_value, masking_cfg=masking_cfg
            )
        return out

    raise ValueError(f"apply_aamm_mask: unsupported shape {tuple(x.shape)}")


__all__ = ["aamm_cfg_from_masking", "apply_aamm_mask"]
