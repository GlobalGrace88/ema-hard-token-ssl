"""
Pseudo-anatomical prior computation for Anatomy-Aware Multi-Zone Masking (AAMM).

These priors are derived from unlabeled inputs only and are used to:
- assign patch/voxel zones (interior/boundary/context) for zone-conditioned mask sampling
- supervise an auxiliary anatomical prediction branch (distance/boundary/zone targets)

The implementation uses lightweight image-structure heuristics (no ground-truth labels).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _minmax_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x_min = x.amin(dim=tuple(range(1, x.dim())), keepdim=True)
    x_max = x.amax(dim=tuple(range(1, x.dim())), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def _patch_mean_2d(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    x: (B,H,W) -> (B,grid_h,grid_w) by mean pooling over non-overlapping patch_size blocks.
    """
    B, H, W = x.shape
    p = patch_size
    if H % p != 0 or W % p != 0:
        raise ValueError(f"2D patching requires H,W divisible by patch_size={p}, got H={H},W={W}")
    gh, gw = H // p, W // p
    patches = x.unfold(1, p, p).unfold(2, p, p)
    return patches.contiguous().view(B, gh, gw, p * p).mean(dim=-1)


def _patch_mean_3d(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    x: (B,D,H,W) -> (B,grid_d,grid_h,grid_w) by mean pooling over non-overlapping patch_size blocks.
    """
    B, D, H, W = x.shape
    p = patch_size
    if D % p != 0 or H % p != 0 or W % p != 0:
        raise ValueError(f"3D patching requires D,H,W divisible by patch_size={p}, got D={D},H={H},W={W}")
    gd, gh, gw = D // p, H // p, W // p
    patches = x.unfold(1, p, p).unfold(2, p, p).unfold(3, p, p)
    return patches.contiguous().view(B, gd, gh, gw, p**3).mean(dim=-1)


def _safe_all_finite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def compute_pseudo_foreground_2d(gray: torch.Tensor, cfg: dict) -> torch.Tensor:
    gray_norm = _minmax_normalize(gray)
    use_soft = bool(cfg.get("foreground_soft_sigmoid", True))
    if use_soft:
        t = cfg.get("foreground_threshold", None)
        if t is None:
            t = gray_norm.mean(dim=(1, 2), keepdim=True)
        else:
            t = torch.as_tensor(t, device=gray.device, dtype=gray_norm.dtype).view(1, 1, 1)
        s = float(cfg.get("foreground_sigmoid_scale", 0.1))
        return torch.sigmoid((gray_norm - t) / (s + 1e-8))
    t = cfg.get("foreground_threshold", 0.5)
    return (gray_norm > float(t)).float()


def compute_pseudo_boundary_2d(gray: torch.Tensor, cfg: dict) -> torch.Tensor:
    gx = gray[:, :, 1:] - gray[:, :, :-1]
    gy = gray[:, 1:, :] - gray[:, :-1, :]
    gx = F.pad(gx, (0, 1), mode="replicate")
    gy = F.pad(gy, (0, 0, 0, 1), mode="replicate")
    mag = (gx**2 + gy**2).sqrt()
    mag = mag.clamp_min(0.0)
    mag = _minmax_normalize(mag)
    if float(cfg.get("boundary_smooth_sigma", 0.0)) > 0:
        n = max(int(round(float(cfg["boundary_smooth_sigma"]))), 1)
        for _ in range(n - 1):
            mag = F.avg_pool2d(mag.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
    return mag


def compute_pseudo_distance_from_boundary_2d(boundary: torch.Tensor, cfg: dict) -> torch.Tensor:
    eps = 1e-8
    dist = 1.0 / (boundary + eps)
    dist = _minmax_normalize(dist)
    power = float(cfg.get("distance_power", 1.0))
    if power != 1.0:
        dist = dist.clamp_min(0.0) ** power
        dist = _minmax_normalize(dist)
    return dist


def compute_pseudo_vesselness_2d(gray: torch.Tensor, cfg: dict) -> torch.Tensor:
    gx = gray[:, :, 1:] - gray[:, :, :-1]
    gy = gray[:, 1:, :] - gray[:, :-1, :]
    gx = F.pad(gx, (0, 1), mode="replicate")
    gy = F.pad(gy, (0, 0, 0, 1), mode="replicate")
    gxx = gx[:, :, 1:] - gx[:, :, :-1]
    gyy = gy[:, 1:, :] - gy[:, :-1, :]
    gxx = F.pad(gxx, (0, 1), mode="replicate")
    gyy = F.pad(gyy, (0, 0, 0, 1), mode="replicate")
    trace = (gxx + gyy).abs()
    trace = _minmax_normalize(trace)
    return trace


def compute_pseudo_anatomy_priors_2d(img: torch.Tensor, patch_size: int, cfg: dict | None = None) -> Dict[str, torch.Tensor]:
    cfg = cfg or {}
    use_foreground_prior = bool(cfg.get("use_foreground_prior", True))
    use_boundary_prior = bool(cfg.get("use_boundary_prior", True))
    use_distance_map = bool(cfg.get("use_distance_map", True))
    use_vesselness = bool(cfg.get("use_vesselness", False))

    if img.dim() == 3:
        img = img.unsqueeze(0)
    if img.dim() != 4:
        raise ValueError(f"Expected img shape (C,H,W) or (B,C,H,W), got {tuple(img.shape)}")
    B, C, H, W = img.shape
    gray = img.mean(dim=1)

    try:
        F_full = compute_pseudo_foreground_2d(gray, cfg) if use_foreground_prior else torch.ones_like(gray)
        if not _safe_all_finite(F_full):
            raise RuntimeError("Foreground prior produced non-finite values")
    except Exception:
        gray_norm = _minmax_normalize(gray)
        t = gray_norm.mean(dim=(1, 2), keepdim=True)
        F_full = (gray_norm > t).float()
        F_full = F_full.nan_to_num(0.0)

    try:
        B_full = compute_pseudo_boundary_2d(gray, cfg) if use_boundary_prior else compute_pseudo_boundary_2d(gray, cfg)
        if not _safe_all_finite(B_full):
            raise RuntimeError("Boundary prior produced non-finite values")
    except Exception:
        B_full = compute_pseudo_boundary_2d(gray, cfg)

    F_p = _patch_mean_2d(F_full, patch_size)
    B_p = _patch_mean_2d(B_full, patch_size)
    F_p = _minmax_normalize(F_p)
    B_p = _minmax_normalize(B_p)

    out: Dict[str, torch.Tensor] = {"F_patches": F_p.view(B, -1), "B_patches": B_p.view(B, -1)}
    if use_distance_map:
        D_p = compute_pseudo_distance_from_boundary_2d(B_p, cfg)
        out["D_patches"] = D_p.view(B, -1)

    if use_vesselness:
        try:
            V_full = compute_pseudo_vesselness_2d(gray, cfg)
            if not _safe_all_finite(V_full):
                raise RuntimeError("Vesselness produced non-finite values")
            V_p = _patch_mean_2d(V_full, patch_size)
            V_p = _minmax_normalize(V_p)
            out["V_patches"] = V_p.view(B, -1)
        except Exception:
            pass

    return out


def compute_pseudo_foreground_3d(gray: torch.Tensor, cfg: dict) -> torch.Tensor:
    gray_norm = _minmax_normalize(gray)
    use_soft = bool(cfg.get("foreground_soft_sigmoid", True))
    if use_soft:
        t = cfg.get("foreground_threshold", None)
        if t is None:
            t = gray_norm.mean(dim=(1, 2, 3), keepdim=True)
        else:
            t = torch.as_tensor(t, device=gray.device, dtype=gray_norm.dtype).view(1, 1, 1, 1)
        s = float(cfg.get("foreground_sigmoid_scale", 0.1))
        return torch.sigmoid((gray_norm - t) / (s + 1e-8))
    t = float(cfg.get("foreground_threshold", 0.5))
    return (gray_norm > t).float()


def compute_pseudo_boundary_3d(gray: torch.Tensor, cfg: dict) -> torch.Tensor:
    gd = gray[:, 1:, :, :] - gray[:, :-1, :, :]
    gh = gray[:, :, 1:, :] - gray[:, :, :-1, :]
    gw = gray[:, :, :, 1:] - gray[:, :, :, :-1]
    gd = F.pad(gd, (0, 0, 0, 0, 0, 1), mode="replicate")
    gh = F.pad(gh, (0, 0, 0, 1, 0, 0), mode="replicate")
    gw = F.pad(gw, (0, 1, 0, 0, 0, 0), mode="replicate")
    mag = (gd**2 + gh**2 + gw**2).sqrt().clamp_min(0.0)
    mag = _minmax_normalize(mag)
    return mag


def compute_pseudo_distance_from_boundary_3d(boundary: torch.Tensor, cfg: dict) -> torch.Tensor:
    eps = 1e-8
    dist = 1.0 / (boundary + eps)
    dist = _minmax_normalize(dist)
    power = float(cfg.get("distance_power", 1.0))
    if power != 1.0:
        dist = dist.clamp_min(0.0) ** power
        dist = _minmax_normalize(dist)
    return dist


def compute_pseudo_vesselness_3d(gray: torch.Tensor, cfg: dict) -> torch.Tensor:
    gray_pad = F.pad(gray, (1, 1, 1, 1, 1, 1), mode="replicate")
    center = gray_pad[:, 1:-1, 1:-1, 1:-1]
    lap = (
        gray_pad[:, 2:, 1:-1, 1:-1]
        + gray_pad[:, :-2, 1:-1, 1:-1]
        + gray_pad[:, 1:-1, 2:, 1:-1]
        + gray_pad[:, 1:-1, :-2, 1:-1]
        + gray_pad[:, 1:-1, 1:-1, 2:]
        + gray_pad[:, 1:-1, 1:-1, :-2]
        - 6.0 * center
    )
    lap = lap.abs()
    lap = _minmax_normalize(lap)
    return lap


def compute_pseudo_anatomy_priors_3d(img: torch.Tensor, patch_size: int, cfg: dict | None = None) -> Dict[str, torch.Tensor]:
    cfg = cfg or {}
    use_foreground_prior = bool(cfg.get("use_foreground_prior", True))
    use_boundary_prior = bool(cfg.get("use_boundary_prior", True))
    use_distance_map = bool(cfg.get("use_distance_map", True))
    use_vesselness = bool(cfg.get("use_vesselness", False))

    if img.dim() == 4:
        img = img.unsqueeze(0)
    if img.dim() != 5:
        raise ValueError(f"Expected img shape (C,D,H,W) or (B,C,D,H,W), got {tuple(img.shape)}")

    B, C, D, H, W = img.shape
    gray = img.mean(dim=1)

    try:
        F_full = compute_pseudo_foreground_3d(gray, cfg) if use_foreground_prior else torch.ones_like(gray)
        if not _safe_all_finite(F_full):
            raise RuntimeError("Foreground prior produced non-finite values")
    except Exception:
        gray_norm = _minmax_normalize(gray)
        t = gray_norm.mean(dim=(1, 2, 3), keepdim=True)
        F_full = (gray_norm > t).float()
        F_full = F_full.nan_to_num(0.0)

    try:
        B_full = compute_pseudo_boundary_3d(gray, cfg) if use_boundary_prior else compute_pseudo_boundary_3d(gray, cfg)
        if not _safe_all_finite(B_full):
            raise RuntimeError("Boundary prior produced non-finite values")
    except Exception:
        B_full = compute_pseudo_boundary_3d(gray, cfg)

    F_p = _patch_mean_3d(F_full, patch_size)
    B_p = _patch_mean_3d(B_full, patch_size)
    F_p = _minmax_normalize(F_p)
    B_p = _minmax_normalize(B_p)

    out: Dict[str, torch.Tensor] = {"F_patches": F_p.view(B, -1), "B_patches": B_p.view(B, -1)}
    if use_distance_map:
        D_p = compute_pseudo_distance_from_boundary_3d(B_p, cfg)
        out["D_patches"] = D_p.view(B, -1)

    if use_vesselness:
        try:
            V_full = compute_pseudo_vesselness_3d(gray, cfg)
            if not _safe_all_finite(V_full):
                raise RuntimeError("Vesselness produced non-finite values")
            V_p = _patch_mean_3d(V_full, patch_size)
            V_p = _minmax_normalize(V_p)
            out["V_patches"] = V_p.view(B, -1)
        except Exception:
            pass

    return out

