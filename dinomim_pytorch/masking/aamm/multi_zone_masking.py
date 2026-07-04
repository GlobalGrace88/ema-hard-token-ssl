"""
Anatomy-Aware Multi-Zone Masking (AAMM).

Pipeline steps:
1) Prior computation (see `anatomy_priors.py`)
2) Zone assignment: interior/boundary/context based on priors + thresholds + dilation
3) Mask sampling: zone-conditioned mask probabilities with exact mask ratio enforcement
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from dinomim_pytorch.masking.aamm.anatomy_priors import (
    compute_pseudo_anatomy_priors_2d,
    compute_pseudo_anatomy_priors_3d,
)


def _ensure_zone_prob_order(zone_probs: Dict[str, float], eps: float = 1e-6) -> Dict[str, float]:
    p_int = float(zone_probs.get("interior", 0.8))
    p_bd = float(zone_probs.get("boundary", 1.0))
    p_ctx = float(zone_probs.get("context", 0.2))
    sorted_vals = sorted([p_bd, p_int, p_ctx], reverse=True)
    p_bd, p_int, p_ctx = sorted_vals
    if abs(p_int - p_ctx) < eps:
        p_int = p_ctx + eps
    if p_bd < p_int + eps:
        p_bd = p_int + eps
    return {"interior": p_int, "boundary": p_bd, "context": p_ctx}


def _dilate_boundary_2d(boundary_mask: torch.Tensor, boundary_width: int) -> torch.Tensor:
    if boundary_width <= 0:
        return boundary_mask
    B, gh, gw = boundary_mask.shape
    k = 2 * boundary_width + 1
    x = boundary_mask.float().unsqueeze(1)  # (B,1,gh,gw)
    x = F.max_pool2d(x, kernel_size=k, stride=1, padding=boundary_width)
    return (x.squeeze(1) > 0.5)


def _dilate_boundary_3d(boundary_mask: torch.Tensor, boundary_width: int) -> torch.Tensor:
    if boundary_width <= 0:
        return boundary_mask
    k = 2 * boundary_width + 1
    x = boundary_mask.float().unsqueeze(1)  # (B,1,gd,gh,gw)
    x = F.max_pool3d(x, kernel_size=k, stride=1, padding=boundary_width)
    return (x.squeeze(1) > 0.5)


def assign_patch_zones_2d(
    F_patches: torch.Tensor,
    B_patches: torch.Tensor,
    grid_h: int,
    grid_w: int,
    cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, N = F_patches.shape
    F_grid = F_patches.view(B, grid_h, grid_w)
    B_grid = B_patches.view(B, grid_h, grid_w)

    boundary_threshold = float(cfg.get("boundary_threshold", 0.5))
    boundary_width = int(cfg.get("boundary_width", 0))

    boundary_raw = B_grid >= boundary_threshold
    boundary_mask = _dilate_boundary_2d(boundary_raw, boundary_width)

    fg_thr = cfg.get("foreground_threshold", None)
    if fg_thr is None:
        fg_thr = F_grid.mean(dim=(1, 2), keepdim=True)
    else:
        fg_thr = float(fg_thr)
        fg_thr = torch.as_tensor(fg_thr, device=F_patches.device, dtype=F_patches.dtype).view(1, 1, 1)
    foreground_mask = F_grid >= fg_thr

    interior_mask = foreground_mask & (~boundary_mask)
    context_mask = ~foreground_mask

    zone_labels = torch.full((B, grid_h, grid_w), 2, dtype=torch.long, device=F_patches.device)
    zone_labels[interior_mask] = 0
    zone_labels[boundary_mask] = 1
    zone_labels = zone_labels.view(B, -1)
    return zone_labels, interior_mask, boundary_mask, context_mask


def assign_patch_zones_3d(
    F_patches: torch.Tensor,
    B_patches: torch.Tensor,
    grid_d: int,
    grid_h: int,
    grid_w: int,
    cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, N = F_patches.shape
    F_grid = F_patches.view(B, grid_d, grid_h, grid_w)
    B_grid = B_patches.view(B, grid_d, grid_h, grid_w)

    boundary_threshold = float(cfg.get("boundary_threshold", 0.5))
    boundary_width = int(cfg.get("boundary_width", 0))

    boundary_raw = B_grid >= boundary_threshold
    boundary_mask = _dilate_boundary_3d(boundary_raw, boundary_width)

    fg_thr = cfg.get("foreground_threshold", None)
    if fg_thr is None:
        fg_thr = F_grid.mean(dim=(1, 2, 3), keepdim=True)
    else:
        fg_thr = float(fg_thr)
        fg_thr = torch.as_tensor(fg_thr, device=F_patches.device, dtype=F_patches.dtype).view(1, 1, 1, 1)
    foreground_mask = F_grid >= fg_thr

    interior_mask = foreground_mask & (~boundary_mask)
    context_mask = ~foreground_mask

    zone_labels = torch.full((B, grid_d, grid_h, grid_w), 2, dtype=torch.long, device=F_patches.device)
    zone_labels[interior_mask] = 0
    zone_labels[boundary_mask] = 1
    zone_labels = zone_labels.view(B, -1)
    return zone_labels, interior_mask, boundary_mask, context_mask


def sample_masks_from_zone_labels(
    zone_labels: torch.Tensor,
    mask_ratio: float,
    zone_probs: Dict[str, float],
    hybrid_uniform_mix: float,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    zone_probs = _ensure_zone_prob_order(zone_probs)
    p_int = float(zone_probs["interior"])
    p_bd = float(zone_probs["boundary"])
    p_ctx = float(zone_probs["context"])

    B, N = zone_labels.shape
    k = int(round(float(mask_ratio) * N))
    k = min(max(k, 0), N)

    weights = torch.empty((B, N), dtype=torch.float32, device=zone_labels.device)
    weights[zone_labels == 0] = p_int
    weights[zone_labels == 1] = p_bd
    weights[zone_labels == 2] = p_ctx

    h = float(hybrid_uniform_mix)
    if h > 0:
        weights = (1.0 - h) * weights + h * torch.ones_like(weights)

    weights = weights.clamp_min(1e-8)

    mask_flat = torch.zeros((B, N), device=zone_labels.device, dtype=torch.float32)
    for b in range(B):
        w = weights[b]
        if w.sum().item() <= 0:
            idx = torch.randperm(N, device=zone_labels.device, generator=generator)[:k]
        else:
            idx = torch.multinomial(w, num_samples=k, replacement=False, generator=generator)
        mask_flat[b, idx] = 1.0

    stats = {"k": k, "zone_probs": zone_probs, "hybrid_uniform_mix": hybrid_uniform_mix}
    return mask_flat, stats


def sample_aamm_mask_2d(
    img: torch.Tensor,
    patch_size: int,
    mask_ratio: float,
    cfg: Dict[str, Any],
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    other_mask_2d: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if img.dim() == 3:
        img = img.unsqueeze(0)
    if img.dim() != 4:
        raise ValueError(f"Expected img (C,H,W) or (B,C,H,W), got {tuple(img.shape)}")
    B, C, H, W = img.shape
    gh, gw = H // patch_size, W // patch_size
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError("AAMM requires H,W divisible by patch_size")

    priors_cfg = {
        "use_foreground_prior": bool(cfg.get("use_foreground_prior", True)),
        "use_boundary_prior": bool(cfg.get("use_boundary_prior", True)),
        "use_distance_map": bool(cfg.get("use_distance_map", True)),
        "use_vesselness": bool(cfg.get("use_vesselness", False)),
        "foreground_threshold": cfg.get("foreground_threshold", None),
        "foreground_soft_sigmoid": cfg.get("foreground_soft_sigmoid", True),
        "foreground_sigmoid_scale": cfg.get("foreground_sigmoid_scale", 0.1),
        "boundary_threshold": cfg.get("boundary_threshold", 0.5),
        "boundary_smooth_sigma": cfg.get("boundary_smooth_sigma", 0.0),
        "distance_power": cfg.get("distance_power", 1.0),
    }
    priors = compute_pseudo_anatomy_priors_2d(img.to(device), patch_size=patch_size, cfg=priors_cfg)

    zone_labels, interior_mask, boundary_mask, context_mask = assign_patch_zones_2d(
        priors["F_patches"],
        priors["B_patches"],
        grid_h=gh,
        grid_w=gw,
        cfg=cfg,
    )

    zone_probs = cfg.get("zone_probs", {}) or {}
    hybrid_uniform_mix = float(cfg.get("hybrid_uniform_mix", 0.0))

    other_vis = None
    if other_mask_2d is not None:
        if other_mask_2d.dim() == 2:
            other_mask_2d = other_mask_2d.unsqueeze(0).expand(B, -1, -1)
        other_vis = (1.0 - other_mask_2d.float().to(device))

    mask_flat, sample_stats = sample_masks_from_zone_labels(
        zone_labels,
        mask_ratio=mask_ratio,
        zone_probs=zone_probs,
        hybrid_uniform_mix=hybrid_uniform_mix,
        generator=generator,
    )

    if other_vis is not None:
        zone_probs = _ensure_zone_prob_order(zone_probs)
        p_int = float(zone_probs["interior"])
        p_bd = float(zone_probs["boundary"])
        p_ctx = float(zone_probs["context"])
        N = gh * gw
        weights = torch.ones((B, N), device=device, dtype=torch.float32)
        weights[zone_labels == 0] = p_int
        weights[zone_labels == 1] = p_bd
        weights[zone_labels == 2] = p_ctx
        weights = weights * (0.5 + other_vis.flatten(1))
        weights = weights.clamp_min(1e-8)
        k_exact = int(round(float(mask_ratio) * N))
        k_exact = min(max(k_exact, 0), N)
        mask_flat = torch.zeros((B, N), device=device, dtype=torch.float32)
        for b in range(B):
            idx = torch.multinomial(weights[b], num_samples=k_exact, replacement=False, generator=generator)
            mask_flat[b, idx] = 1.0

    mask_2d = mask_flat.view(B, gh, gw)
    stats = {
        "zone_stats": {
            "interior_frac": interior_mask.float().mean(dim=(1, 2)).mean().item(),
            "boundary_frac": boundary_mask.float().mean(dim=(1, 2)).mean().item(),
            "context_frac": context_mask.float().mean(dim=(1, 2)).mean().item(),
        },
        **sample_stats,
    }
    if img.shape[0] == 1:
        return mask_flat.view(-1), mask_2d.squeeze(0), stats
    return mask_flat.view(-1), mask_2d, stats


def sample_aamm_mask_3d(
    img: torch.Tensor,
    patch_size: int,
    mask_ratio: float,
    cfg: Dict[str, Any],
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    other_mask_3d: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if img.dim() == 4:
        img = img.unsqueeze(0)
    if img.dim() != 5:
        raise ValueError(f"Expected img (C,D,H,W) or (B,C,D,H,W), got {tuple(img.shape)}")
    B, C, D, H, W = img.shape
    gd, gh, gw = D // patch_size, H // patch_size, W // patch_size
    if D % patch_size != 0 or H % patch_size != 0 or W % patch_size != 0:
        raise ValueError("AAMM requires D,H,W divisible by patch_size")

    priors_cfg = {
        "use_foreground_prior": bool(cfg.get("use_foreground_prior", True)),
        "use_boundary_prior": bool(cfg.get("use_boundary_prior", True)),
        "use_distance_map": bool(cfg.get("use_distance_map", True)),
        "use_vesselness": bool(cfg.get("use_vesselness", False)),
        "foreground_threshold": cfg.get("foreground_threshold", None),
        "foreground_soft_sigmoid": cfg.get("foreground_soft_sigmoid", True),
        "foreground_sigmoid_scale": cfg.get("foreground_sigmoid_scale", 0.1),
        "boundary_threshold": cfg.get("boundary_threshold", 0.5),
        "distance_power": cfg.get("distance_power", 1.0),
    }
    priors = compute_pseudo_anatomy_priors_3d(img.to(device), patch_size=patch_size, cfg=priors_cfg)

    zone_labels, interior_mask, boundary_mask, context_mask = assign_patch_zones_3d(
        priors["F_patches"],
        priors["B_patches"],
        grid_d=gd,
        grid_h=gh,
        grid_w=gw,
        cfg=cfg,
    )

    zone_probs = cfg.get("zone_probs", {}) or {}
    hybrid_uniform_mix = float(cfg.get("hybrid_uniform_mix", 0.0))

    other_vis = None
    if other_mask_3d is not None:
        if other_mask_3d.dim() == 3:
            other_mask_3d = other_mask_3d.unsqueeze(0).expand(B, -1, -1, -1)
        other_vis = (1.0 - other_mask_3d.float().to(device))

    mask_flat, sample_stats = sample_masks_from_zone_labels(
        zone_labels,
        mask_ratio=mask_ratio,
        zone_probs=zone_probs,
        hybrid_uniform_mix=hybrid_uniform_mix,
        generator=generator,
    )

    if other_vis is not None:
        zone_probs = _ensure_zone_prob_order(zone_probs)
        p_int = float(zone_probs["interior"])
        p_bd = float(zone_probs["boundary"])
        p_ctx = float(zone_probs["context"])
        N = gd * gh * gw
        weights = torch.ones((B, N), device=device, dtype=torch.float32)
        weights[zone_labels == 0] = p_int
        weights[zone_labels == 1] = p_bd
        weights[zone_labels == 2] = p_ctx
        weights = weights * (0.5 + other_vis.flatten(1))
        weights = weights.clamp_min(1e-8)
        k_exact = int(round(float(mask_ratio) * N))
        k_exact = min(max(k_exact, 0), N)
        mask_flat = torch.zeros((B, N), device=device, dtype=torch.float32)
        for b in range(B):
            idx = torch.multinomial(weights[b], num_samples=k_exact, replacement=False, generator=generator)
            mask_flat[b, idx] = 1.0

    mask_3d = mask_flat.view(B, gd, gh, gw)
    stats = {
        "zone_stats": {
            "interior_frac": interior_mask.float().mean(dim=(1, 2, 3)).mean().item(),
            "boundary_frac": boundary_mask.float().mean(dim=(1, 2, 3)).mean().item(),
            "context_frac": context_mask.float().mean(dim=(1, 2, 3)).mean().item(),
        },
        **sample_stats,
    }
    if img.shape[0] == 1:
        return mask_flat.view(-1), mask_3d.squeeze(0), stats
    return mask_flat.view(-1), mask_3d, stats

