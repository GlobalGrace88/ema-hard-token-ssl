"""
UNETR++ inpainting + masked EMA teacher-feature reconstruction (no prototypes / DINO heads).

v1: stage-4 tokens only.
v2: multi-scale stages 2/3/4 with per-stage predictors and stricter token masking.
"""

from __future__ import annotations

import copy
import sys
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from dinomim_pytorch.segmentation_models.official_unetrpp3d import (
    OFFICIAL_VARIANT_SPATIAL,
    build_official_unetrpp3d,
)

# UNETR++ encoder hidden_states indices (synapse/tumor/acdc/lung share this layout).
# hidden_states[0]=enc1, [1]=enc2, [2]=enc3, [3]=enc4 (EPA tokens).
STAGE_TO_HIDDEN_IDX: Dict[int, int] = {1: 0, 2: 1, 3: 2, 4: 3}
STAGE_SOURCE_DESC: Dict[int, str] = {
    1: "hidden_states[0] / enc1 (stem+stage0)",
    2: "hidden_states[1] / enc2 (downsample_layers[1]+stages[1])",
    3: "hidden_states[2] / enc3 (downsample_layers[2]+stages[2])",
    4: "hidden_states[3] / enc4 (downsample_layers[3]+stages[3], EPA tokens)",
}


def resolve_feature_recon_version(feat_cfg: Mapping[str, Any]) -> str:
    version = str(feat_cfg.get("version", "") or "").strip().lower()
    if version in ("v2", "v2_multiscale", "multiscale"):
        return "v2_multiscale"
    if version in ("v2_fair_multiscale", "fair_multiscale"):
        return "v2_fair_multiscale"
    if version in ("v3_stage34_lite", "v3", "stage34_lite", "stage34"):
        return "v3_stage34_lite"
    if version in ("v4_hard_stage34", "v4_hard", "hard_stage34"):
        return "v4_hard_stage34"
    if version in ("v4_hard_stage34_smoothl1", "v4_hard_smoothl1", "hard_stage34_smoothl1"):
        return "v4_hard_stage34_smoothl1"
    if version in (
        "v4_hard_stage234_smoothl1",
        "hard_stage234_smoothl1",
    ):
        return "v4_hard_stage234_smoothl1"
    if version in (
        "v4_hard_stage234_cosine",
        "v4_hard_stage234",
        "hard_stage234_cosine",
        "hard_stage234",
    ):
        return "v4_hard_stage234_cosine"
    if version in (
        "v4_stage234_cosine_nohard",
        "stage234_cosine_nohard",
        "msfr_stage234_cosine",
    ):
        return "v4_stage234_cosine_nohard"
    if version in (
        "v4_hard_stage4_smoothl1",
        "hard_stage4_smoothl1",
    ):
        return "v4_hard_stage4_smoothl1"
    if version in (
        "v5_adaptive_boundary",
        "v5_adaptive",
        "adaptive_boundary",
    ):
        return "v5_adaptive_boundary"
    if version in ("v1", "v1_stage4", "stage4", ""):
        return "v1_stage4"
    return version


def is_multiscale_feature_recon(version: str) -> bool:
    return str(version) in (
        "v2_multiscale",
        "v2_fair_multiscale",
        "v3_stage34_lite",
        "v4_hard_stage34",
        "v4_hard_stage34_smoothl1",
        "v4_hard_stage234_smoothl1",
        "v4_hard_stage234_cosine",
        "v4_stage234_cosine_nohard",
        "v4_hard_stage4_smoothl1",
        "v5_adaptive_boundary",
    )


def resolve_feature_loss_kind(feat_cfg: Mapping[str, Any]) -> str:
    loss = str(feat_cfg.get("loss", "cosine") or "cosine").strip().lower()
    if loss in ("hard_cosine", "hard_cosine_smoothl1", "cosine"):
        return loss
    version = resolve_feature_recon_version(feat_cfg)
    if version == "v4_hard_stage34":
        return "hard_cosine"
    if version == "v4_hard_stage34_smoothl1":
        return "hard_cosine_smoothl1"
    if version == "v4_hard_stage234_smoothl1":
        return "hard_cosine_smoothl1"
    if version == "v5_adaptive_boundary":
        return "hard_cosine_smoothl1"
    if version == "v4_hard_stage234_cosine":
        return "hard_cosine"
    if version in ("v4_hard_stage4_smoothl1",):
        return "hard_cosine_smoothl1"
    if version == "v4_stage234_cosine_nohard":
        return "cosine"
    return "cosine"


SSL_OBJECTIVE_PRESETS: Dict[str, Dict[str, Any]] = {
    # Paper main: best HD95 / boundary focus (stages 2--4).
    "boundary": {"stages": [2, 3, 4], "stage_weights": {"stage2": 0.2, "stage3": 0.3, "stage4": 0.5}},
    # Best mean Dice in paper ablations (stages 3--4).
    "overlap": {"stages": [3, 4], "stage_weights": {"stage3": 0.35, "stage4": 0.65}},
    # Compromise between boundary and overlap objectives.
    "balanced": {"stages": [2, 3, 4], "stage_weights": {"stage2": 0.25, "stage3": 0.35, "stage4": 0.4}},
}


def apply_objective_preset(feat_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Merge ``objective_preset`` into ``stages`` / ``stage_weights`` when not explicitly set."""
    out = dict(feat_cfg or {})
    preset_name = str(out.get("objective_preset", "") or "").strip().lower()
    if not preset_name:
        return out
    preset = SSL_OBJECTIVE_PRESETS.get(preset_name)
    if preset is None:
        raise ValueError(
            f"Unknown objective_preset={preset_name!r}. "
            f"Choose from: {', '.join(sorted(SSL_OBJECTIVE_PRESETS))}"
        )
    if not out.get("stages"):
        out["stages"] = list(preset["stages"])
    if not out.get("stage_weights"):
        out["stage_weights"] = dict(preset["stage_weights"])
    return out


def merge_feature_reconstruction_config(feat_cfg: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Normalize feature-reconstruction config (objective preset, mining defaults)."""
    return apply_objective_preset(dict(feat_cfg or {}))


def resolve_hard_token_mining_cfg(feat_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    raw = dict(feat_cfg.get("hard_token_mining") or {})
    loss_kind = resolve_feature_loss_kind(feat_cfg)
    enabled = bool(raw.get("enabled", False))
    if loss_kind.startswith("hard"):
        enabled = True
    mode = str(raw.get("mode", "topk_error")).strip().lower()
    if mode in ("topk", "topk_ratio", "topk_error"):
        mode = "topk_error"
    elif mode in ("error_mass", "mass", "cumulative_error"):
        mode = "error_mass"
    return {
        "enabled": enabled,
        "mode": mode,
        "topk_ratio": float(raw.get("topk_ratio", 0.5)),
        "error_mass_fraction": float(raw.get("error_mass_fraction", 0.65)),
        "topk_ratio_per_stage": dict(raw.get("topk_ratio_per_stage") or {}),
        "error_mass_fraction_per_stage": dict(raw.get("error_mass_fraction_per_stage") or {}),
        "min_tokens": int(raw.get("min_tokens", 8)),
        "detach_weights": bool(raw.get("detach_weights", True)),
        "apply_per_stage": bool(raw.get("apply_per_stage", True)),
        "curriculum_epochs": int(raw.get("curriculum_epochs", 0) or 0),
        "curriculum_topk_start": float(raw.get("curriculum_topk_start", 0.75)),
        "curriculum_error_mass_start": float(raw.get("curriculum_error_mass_start", 0.85)),
    }


def resolve_hard_token_mining_for_epoch(
    feat_cfg: Mapping[str, Any],
    epoch: int,
    *,
    stage_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Epoch-aware hard-token settings (curriculum + optional per-stage overrides)."""
    hard = dict(resolve_hard_token_mining_cfg(feat_cfg))
    cur_epochs = int(hard.get("curriculum_epochs", 0) or 0)
    if cur_epochs > 0:
        t = min(1.0, float(epoch + 1) / float(cur_epochs))
        hard["topk_ratio"] = (
            hard["curriculum_topk_start"]
            + (hard["topk_ratio"] - hard["curriculum_topk_start"]) * t
        )
        hard["error_mass_fraction"] = (
            hard["curriculum_error_mass_start"]
            + (hard["error_mass_fraction"] - hard["curriculum_error_mass_start"]) * t
        )
    if stage_key and hard.get("apply_per_stage", True):
        per_topk = hard.get("topk_ratio_per_stage") or {}
        per_mass = hard.get("error_mass_fraction_per_stage") or {}
        if stage_key in per_topk:
            hard["topk_ratio"] = float(per_topk[stage_key])
        if stage_key in per_mass:
            hard["error_mass_fraction"] = float(per_mass[stage_key])
    return hard


def resolve_feature_loss_weights(feat_cfg: Mapping[str, Any]) -> Dict[str, float]:
    raw = dict(feat_cfg.get("feature_loss") or {})
    return {
        "cosine_weight": float(raw.get("cosine_weight", 1.0)),
        "smooth_l1_weight": float(raw.get("smooth_l1_weight", 0.0)),
        "normalize_before_l1": bool(raw.get("normalize_before_l1", True)),
    }


def _hard_token_selection_mask_topk(
    per_token_err: torch.Tensor,
    base_mask: torch.Tensor,
    *,
    topk_ratio: float,
    min_tokens: int,
    detach_weights: bool,
) -> Tuple[torch.Tensor, float]:
    """Select top-error masked tokens by ratio. Returns (hard_mask, total_hard_tokens)."""
    err = per_token_err.detach() if detach_weights else per_token_err
    hard_mask = torch.zeros_like(base_mask)
    total_hard = 0.0
    bsz = int(base_mask.shape[0])
    for bi in range(bsz):
        valid = base_mask[bi] > 0
        n_valid = int(valid.sum().item())
        if n_valid <= 0:
            continue
        k = max(int(min_tokens), int(float(topk_ratio) * n_valid))
        k = min(k, n_valid)
        valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        errs = err[bi, valid_idx]
        if errs.numel() == 1:
            top_local = torch.zeros(1, dtype=torch.long, device=errs.device)
        else:
            top_local = torch.topk(errs, k, largest=True).indices
        chosen = valid_idx[top_local]
        hard_mask[bi, chosen] = 1.0
        total_hard += float(chosen.numel())
    return hard_mask, total_hard


def _hard_token_selection_mask_error_mass(
    per_token_err: torch.Tensor,
    base_mask: torch.Tensor,
    *,
    error_mass_fraction: float,
    min_tokens: int,
    detach_weights: bool,
) -> Tuple[torch.Tensor, float]:
    """Select hardest tokens until cumulative error reaches ``error_mass_fraction`` of masked mass."""
    err = per_token_err.detach() if detach_weights else per_token_err
    hard_mask = torch.zeros_like(base_mask)
    total_hard = 0.0
    mass_frac = float(min(max(error_mass_fraction, 0.0), 1.0))
    bsz = int(base_mask.shape[0])
    for bi in range(bsz):
        valid = base_mask[bi] > 0
        n_valid = int(valid.sum().item())
        if n_valid <= 0:
            continue
        valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        errs = err[bi, valid_idx]
        if errs.numel() == 1:
            chosen = valid_idx
        else:
            sorted_err, order = torch.sort(errs, descending=True)
            total_err = float(sorted_err.sum().clamp_min(1e-8).item())
            target = mass_frac * total_err
            cum = torch.cumsum(sorted_err, dim=0)
            k = int((cum < target).sum().item()) + 1
            k = max(int(min_tokens), k)
            k = min(k, n_valid)
            chosen = valid_idx[order[:k]]
        hard_mask[bi, chosen] = 1.0
        total_hard += float(chosen.numel())
    return hard_mask, total_hard


def _hard_token_selection_mask(
    per_token_err: torch.Tensor,
    base_mask: torch.Tensor,
    *,
    hard_cfg: Mapping[str, Any],
) -> Tuple[torch.Tensor, float]:
    mode = str(hard_cfg.get("mode", "topk_error")).lower()
    common = {
        "min_tokens": int(hard_cfg.get("min_tokens", 8)),
        "detach_weights": bool(hard_cfg.get("detach_weights", True)),
    }
    if mode == "error_mass":
        return _hard_token_selection_mask_error_mass(
            per_token_err,
            base_mask,
            error_mass_fraction=float(hard_cfg.get("error_mass_fraction", 0.65)),
            **common,
        )
    return _hard_token_selection_mask_topk(
        per_token_err,
        base_mask,
        topk_ratio=float(hard_cfg.get("topk_ratio", 0.5)),
        **common,
    )


def resolve_feature_stages(feat_cfg: Mapping[str, Any]) -> List[int]:
    feat_cfg = apply_objective_preset(dict(feat_cfg or {}))
    version = resolve_feature_recon_version(feat_cfg)
    raw = feat_cfg.get("stages")
    if raw:
        stages = [int(s) for s in raw]
    elif is_multiscale_feature_recon(version):
        stages = [2, 3, 4]
    else:
        legacy = str(feat_cfg.get("feature_stage", "stage4")).lower().replace("stage", "")
        stages = [int(legacy) if legacy.isdigit() else 4]
    stages = sorted({int(s) for s in stages if int(s) in STAGE_TO_HIDDEN_IDX})
    if not stages:
        stages = [4]
    return stages


def resolve_stage_weights(feat_cfg: Mapping[str, Any], stages: List[int]) -> Dict[str, float]:
    feat_cfg = apply_objective_preset(dict(feat_cfg or {}))
    raw = dict(feat_cfg.get("stage_weights") or {})
    weights: Dict[str, float] = {}
    for stage in stages:
        key = f"stage{stage}"
        if key in raw:
            weights[key] = float(raw[key])
        elif str(stage) in raw:
            weights[key] = float(raw[str(stage)])
    if not weights:
        version = resolve_feature_recon_version(feat_cfg)
        if stages == [2, 3, 4]:
            if version == "v2_fair_multiscale":
                weights = {"stage2": 0.2, "stage3": 0.3, "stage4": 0.5}
            elif str(feat_cfg.get("objective_preset", "")).lower() == "boundary":
                weights = {"stage2": 0.2, "stage3": 0.3, "stage4": 0.5}
            else:
                weights = {"stage2": 0.5, "stage3": 0.3, "stage4": 0.2}
        elif stages == [3, 4]:
            weights = {"stage3": 0.35, "stage4": 0.65}
        else:
            inv = 1.0 / float(len(stages))
            weights = {f"stage{s}": inv for s in stages}
    total = sum(weights.get(f"stage{s}", 0.0) for s in stages)
    if total <= 0:
        inv = 1.0 / float(len(stages))
        return {f"stage{s}": inv for s in stages}
    return {f"stage{s}": weights.get(f"stage{s}", 0.0) / total for s in stages}


def _factor_token_grid(n_tokens: int, spatial: Tuple[int, int, int]) -> Tuple[int, int, int]:
    d, h, w = spatial
    best = (1, 1, max(1, n_tokens))
    best_score = float("inf")
    for dp in range(1, min(n_tokens, 64) + 1):
        if n_tokens % dp:
            continue
        rem = n_tokens // dp
        for hp in range(1, min(rem, 128) + 1):
            if rem % hp:
                continue
            wp = rem // hp
            score = abs(dp / max(d, 1) - hp / max(h, 1)) + abs(hp / max(h, 1) - wp / max(w, 1))
            if score < best_score:
                best_score = score
                best = (dp, hp, wp)
    return best


def tokens_from_stage_feat(
    feat: torch.Tensor,
    spatial: Tuple[int, int, int],
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """Encoder stage feature -> patch tokens ``[B, N, C]``."""
    if feat.dim() == 5:
        _b, _c, dp, hp, wp = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        return tokens, (int(dp), int(hp), int(wp))
    if feat.dim() == 3:
        _b, n, _c = feat.shape
        return feat, _factor_token_grid(int(n), spatial)
    if feat.dim() == 2:
        return feat.unsqueeze(1), (1, 1, 1)
    raise RuntimeError(f"Unexpected stage feature shape {tuple(feat.shape)}")


def tokens_from_stage4_feat(
    feat: torch.Tensor,
    spatial: Tuple[int, int, int],
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    return tokens_from_stage_feat(feat, spatial)


def voxel_mask_to_token_mask(
    voxel_mask: torch.Tensor,
    token_grid: Tuple[int, int, int],
    *,
    mode: str = "any",
    threshold: float = 0.75,
) -> torch.Tensor:
    """``[B,1,D,H,W]`` or ``[B,D,H,W]`` -> ``[B,N]`` token mask (1 = include)."""
    if voxel_mask.dim() == 4:
        voxel_mask = voxel_mask.unsqueeze(1)
    dp, hp, wp = token_grid
    mode_l = str(mode or "any").strip().lower()
    if mode_l in ("avg", "avg_threshold", "average", "mean"):
        m = F.adaptive_avg_pool3d(voxel_mask.float(), output_size=(dp, hp, wp))
        return (m.flatten(2).squeeze(1) >= float(threshold)).float()
    m = F.interpolate(voxel_mask.float(), size=(dp, hp, wp), mode="nearest")
    return m.flatten(2).squeeze(1)


def inpainting_mask_and_indicator(
    x: torch.Tensor,
    inpaint_cfg: Mapping[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample inpainting mask for SSL pretraining.

    ``inpainting.mask_strategy``: ``random`` (default) or ``aamm`` (anatomy-aware zones).
    """
    strategy = str(inpaint_cfg.get("mask_strategy", "random")).strip().lower()
    mask_ratio = float(inpaint_cfg.get("mask_ratio", 0.75))
    patch_size = int(inpaint_cfg.get("patch_size", 16))
    mask_value = float(inpaint_cfg.get("mask_value", 0.0))

    if strategy in ("random", "block", "uniform"):
        return random_block_mask_and_indicator(
            x,
            mask_ratio=mask_ratio,
            patch_size=patch_size,
            mask_value=mask_value,
        )

    if strategy in ("aamm", "anatomy", "anatomy_aware"):
        from dinomim_pytorch.masking.aamm.apply import aamm_cfg_from_masking
        from dinomim_pytorch.masking.aamm.multi_zone_masking import sample_aamm_mask_3d

        masking_cfg = aamm_cfg_from_masking(dict(inpaint_cfg.get("masking") or {}))
        hybrid = float(inpaint_cfg.get("hybrid_uniform_mix", masking_cfg.get("hybrid_uniform_mix", 0.0)))
        if hybrid > 0.0 and torch.rand(()) < hybrid:
            return random_block_mask_and_indicator(
                x,
                mask_ratio=mask_ratio,
                patch_size=patch_size,
                mask_value=mask_value,
            )

        if x.dim() != 5:
            raise ValueError(f"AAMM inpainting mask expects [B,C,D,H,W], got {tuple(x.shape)}")
        b, c, d, h, w = x.shape
        indicator = torch.zeros(b, 1, d, h, w, device=x.device, dtype=x.dtype)
        for bi in range(b):
            _flat, mask_3d, _ = sample_aamm_mask_3d(
                x[bi : bi + 1],
                patch_size=patch_size,
                mask_ratio=mask_ratio,
                cfg=masking_cfg,
                device=x.device,
            )
            m = torch.nn.functional.interpolate(
                mask_3d.float().view(1, 1, *mask_3d.shape[-3:]),
                size=(d, h, w),
                mode="nearest",
            )
            indicator[bi] = m
        x_masked = x * (1.0 - indicator)
        if mask_value != 0.0:
            x_masked = torch.where(
                indicator.bool().expand_as(x),
                torch.full_like(x, mask_value),
                x_masked,
            )
        return x_masked, indicator

    raise ValueError(f"Unknown inpainting.mask_strategy={strategy!r} (use random or aamm).")


def random_block_mask_and_indicator(
    x: torch.Tensor,
    *,
    mask_ratio: float,
    patch_size: int,
    mask_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block inpainting mask; returns ``(x_masked, voxel_mask)`` with mask ``1`` = masked."""
    import random

    import numpy as np

    y = x.clone()
    b = y.shape[0]
    spatial = tuple(y.shape[-3:])
    dims = len(spatial)
    ps = max(1, int(patch_size))
    n_cells = [max(1, s // ps) for s in spatial]
    n_total = int(np.prod(n_cells))
    n_mask = max(1, int(mask_ratio * n_total))
    indicator = torch.zeros(b, 1, *spatial, device=x.device, dtype=x.dtype)
    for bi in range(b):
        picks = random.sample(range(n_total), k=min(n_mask, n_total))
        for idx in picks:
            if dims == 3:
                n_d, n_h, n_w = n_cells
                zi = idx // (n_h * n_w)
                yi = (idx % (n_h * n_w)) // n_w
                xi = idx % n_w
                z0, y0, x0 = zi * ps, yi * ps, xi * ps
                indicator[bi, :, z0 : z0 + ps, y0 : y0 + ps, x0 : x0 + ps] = 1.0
            else:
                n_h, n_w = n_cells
                yi = idx // n_w
                xi = idx % n_w
                y0, x0 = yi * ps, xi * ps
                indicator[bi, :, y0 : y0 + ps, x0 : x0 + ps] = 1.0
    x_masked = x * (1.0 - indicator)
    if float(mask_value) != 0.0:
        x_masked = torch.where(indicator.bool().expand_as(x), torch.full_like(x, mask_value), x)
    return x_masked, indicator


class FeaturePredictor(nn.Module):
    def __init__(self, dim: int = 256, hidden_dim: int = 512) -> None:
        super().__init__()
        hid = max(int(hidden_dim), int(dim))
        self.net = nn.Sequential(
            nn.Linear(dim, hid),
            nn.GELU(),
            nn.LayerNorm(hid),
            nn.Linear(hid, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiStageFeaturePredictors(nn.Module):
    def __init__(self, stage_dims: Dict[str, int], *, hidden_dim: int = 512) -> None:
        super().__init__()
        self.predictors = nn.ModuleDict(
            {name: FeaturePredictor(int(dim), hidden_dim) for name, dim in stage_dims.items()}
        )

    def forward_stage(self, stage_key: str, tokens: torch.Tensor) -> torch.Tensor:
        return self.predictors[stage_key](tokens)


def masked_cosine_feature_loss(
    student_pred: torch.Tensor,
    teacher_target: torch.Tensor,
    *,
    token_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Normalized cosine regression: ``2 - 2 * cos(pred, target)`` per token."""
    pred = F.normalize(student_pred, dim=-1)
    target = F.normalize(teacher_target.detach(), dim=-1)
    per_token = 2.0 - 2.0 * (pred * target).sum(dim=-1)
    cosine_sim = (pred * target).sum(dim=-1)

    if token_mask is not None:
        m = token_mask.float()
        denom = m.sum().clamp_min(1.0)
        loss = (per_token * m).sum() / denom
        mask_frac = float(m.mean().detach())
        mean_cos = float((cosine_sim * m).sum() / denom)
    else:
        loss = per_token.mean()
        mask_frac = 1.0
        mean_cos = float(cosine_sim.mean().detach())

    meta = {
        "mean_cosine_similarity": mean_cos,
        "masked_token_fraction": mask_frac,
        "student_pred_std": float(pred.float().std(dim=-1).mean().detach()),
        "teacher_target_std": float(target.float().std(dim=-1).mean().detach()),
    }
    return loss, meta


def masked_feature_reconstruction_loss(
    student_pred: torch.Tensor,
    teacher_target: torch.Tensor,
    *,
    token_mask: Optional[torch.Tensor] = None,
    feat_cfg: Optional[Mapping[str, Any]] = None,
    epoch: int = 0,
    stage_key: Optional[str] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Cosine / hard-cosine / hard-cosine+SmoothL1 feature loss with optional hard-token mining."""
    feat_cfg = feat_cfg or {}
    loss_kind = resolve_feature_loss_kind(feat_cfg)
    hard_cfg = resolve_hard_token_mining_for_epoch(feat_cfg, epoch, stage_key=stage_key)
    loss_w = resolve_feature_loss_weights(feat_cfg)

    pred_norm = F.normalize(student_pred, dim=-1)
    target_norm = F.normalize(teacher_target.detach(), dim=-1)
    cosine_sim = (pred_norm * target_norm).sum(dim=-1)
    per_token_cos_err = 2.0 - 2.0 * cosine_sim

    if token_mask is not None:
        base_mask = token_mask.float()
    else:
        base_mask = torch.ones_like(per_token_cos_err)

    hard_frac = 0.0
    hard_token_count = 0.0
    if hard_cfg["enabled"] and loss_kind.startswith("hard"):
        hard_mask, hard_token_count = _hard_token_selection_mask(
            per_token_cos_err,
            base_mask,
            hard_cfg=hard_cfg,
        )
        eff_mask = hard_mask
        hard_frac = float(hard_mask.sum() / base_mask.sum().clamp_min(1.0).detach())
    else:
        eff_mask = base_mask
        hard_token_count = float(base_mask.sum().detach())

    denom = eff_mask.sum().clamp_min(1.0)
    cos_loss = (per_token_cos_err * eff_mask).sum() / denom
    mean_cos = float((cosine_sim * eff_mask).sum() / denom)

    meta: Dict[str, float] = {
        "mean_cosine_similarity": mean_cos,
        "masked_token_fraction": float(base_mask.mean().detach()),
        "hard_token_fraction": hard_frac,
        "hard_token_count": hard_token_count,
        "student_pred_std": float(pred_norm.float().std(dim=-1).mean().detach()),
        "teacher_target_std": float(target_norm.float().std(dim=-1).mean().detach()),
        "smooth_l1_mean": 0.0,
    }

    if loss_kind == "hard_cosine_smoothl1":
        if loss_w["normalize_before_l1"]:
            p_l1, t_l1 = pred_norm, target_norm
        else:
            p_l1, t_l1 = student_pred, teacher_target.detach()
        per_token_smooth = F.smooth_l1_loss(p_l1, t_l1, reduction="none", beta=1.0).sum(dim=-1)
        smooth_loss = (per_token_smooth * eff_mask).sum() / denom
        total = loss_w["cosine_weight"] * cos_loss + loss_w["smooth_l1_weight"] * smooth_loss
        meta["smooth_l1_mean"] = float(smooth_loss.detach())
        return total, meta

    return cos_loss, meta


def multiscale_feature_loss(
    student_preds: Dict[str, torch.Tensor],
    teacher_targets: Dict[str, torch.Tensor],
    *,
    token_masks: Dict[str, Optional[torch.Tensor]],
    stage_weights: Dict[str, float],
    feat_cfg: Optional[Mapping[str, Any]] = None,
    epoch: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    total = torch.zeros((), device=next(iter(student_preds.values())).device)
    meta: Dict[str, float] = {}
    used_weight = 0.0
    for stage_key, pred in student_preds.items():
        target = teacher_targets[stage_key]
        mask = token_masks.get(stage_key)
        if mask is not None and float(mask.sum()) <= 0:
            meta[f"feature_{stage_key}"] = 0.0
            meta[f"cos_{stage_key}"] = 0.0
            meta[f"mask_frac_{stage_key}"] = 0.0
            meta[f"pred_std_{stage_key}"] = 0.0
            meta[f"target_std_{stage_key}"] = 0.0
            meta[f"hard_frac_{stage_key}"] = 0.0
            meta[f"hard_tokens_{stage_key}"] = 0.0
            meta[f"smooth_l1_{stage_key}"] = 0.0
            continue
        loss_s, loss_meta = masked_feature_reconstruction_loss(
            pred, target, token_mask=mask, feat_cfg=feat_cfg, epoch=epoch, stage_key=stage_key,
        )
        w = float(stage_weights.get(stage_key, 0.0))
        total = total + w * loss_s
        used_weight += w
        meta[f"feature_{stage_key}"] = float(loss_s.detach())
        meta[f"cos_{stage_key}"] = float(loss_meta["mean_cosine_similarity"])
        meta[f"mask_frac_{stage_key}"] = float(loss_meta["masked_token_fraction"])
        meta[f"pred_std_{stage_key}"] = float(loss_meta["student_pred_std"])
        meta[f"target_std_{stage_key}"] = float(loss_meta["teacher_target_std"])
        meta[f"hard_frac_{stage_key}"] = float(loss_meta.get("hard_token_fraction", 0.0))
        meta[f"hard_tokens_{stage_key}"] = float(loss_meta.get("hard_token_count", 0.0))
        meta[f"smooth_l1_{stage_key}"] = float(loss_meta.get("smooth_l1_mean", 0.0))
    if used_weight > 0:
        total = total / used_weight
    meta["mean_feature"] = float(total.detach())
    return total, meta


def _primary_logits(raw: Union[torch.Tensor, list, tuple]) -> torch.Tensor:
    if isinstance(raw, (list, tuple)):
        return raw[0]
    return raw


def _stage_keys(stages: List[int]) -> List[str]:
    return [f"stage{s}" for s in stages]


def _extract_stage_tokens(
    hidden_states: List[torch.Tensor],
    stage: int,
    spatial: Tuple[int, int, int],
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    idx = STAGE_TO_HIDDEN_IDX[int(stage)]
    if idx >= len(hidden_states):
        raise RuntimeError(f"Stage {stage} index {idx} out of range (n={len(hidden_states)})")
    return tokens_from_stage_feat(hidden_states[idx], spatial)


class UNETRPPInpaintingFeatureReconstruction(nn.Module):
    """Inpainting + masked teacher-feature reconstruction."""

    def __init__(
        self,
        student_net: nn.Module,
        *,
        stages: List[int],
        stage_dims: Dict[str, int],
        predictor_hidden_dim: int = 512,
        version: str = "v1_stage4",
    ) -> None:
        super().__init__()
        if not hasattr(student_net, "unetr_pp_encoder"):
            raise AttributeError("student_net must expose unetr_pp_encoder (official UNETR++)")
        self.student_net = student_net
        self.version = str(version)
        self.stages = list(stages)
        self.stage_keys = _stage_keys(self.stages)
        self.feature_predictors = MultiStageFeaturePredictors(stage_dims, hidden_dim=predictor_hidden_dim)
        self.teacher_encoder = copy.deepcopy(student_net.unetr_pp_encoder)
        self.teacher_encoder.load_state_dict(student_net.unetr_pp_encoder.state_dict())
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False

    @property
    def feature_predictor(self) -> FeaturePredictor:
        """v1 compatibility alias (stage4 predictor)."""
        if "stage4" in self.feature_predictors.predictors:
            return self.feature_predictors.predictors["stage4"]
        return next(iter(self.feature_predictors.predictors.values()))

    def _student_encoder_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        captured: Dict[str, Any] = {}
        student_net = self.student_net.module if hasattr(self.student_net, "module") else self.student_net

        def _hook(_module, _inp, out) -> None:
            captured["hidden_states"] = out[1]

        handle = student_net.unetr_pp_encoder.register_forward_hook(_hook)
        try:
            recon = _primary_logits(self.student_net(x))
        finally:
            handle.remove()
        hidden_states = captured.get("hidden_states")
        if not hidden_states:
            raise RuntimeError("Student encoder hook did not capture hidden_states")
        return recon, list(hidden_states)

    def forward_student(
        self,
        x_masked: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Tuple[int, int, int]]]:
        recon, hidden_states = self._student_encoder_forward(x_masked)
        spatial = tuple(x_masked.shape[-3:])
        tokens: Dict[str, torch.Tensor] = {}
        grids: Dict[str, Tuple[int, int, int]] = {}
        for stage in self.stages:
            key = f"stage{stage}"
            tok, grid = _extract_stage_tokens(hidden_states, stage, spatial)
            tokens[key] = tok
            grids[key] = grid
        return recon, tokens, grids

    def forward_student_v1(
        self,
        x_masked: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
        recon, tokens, grids = self.forward_student(x_masked)
        key = self.stage_keys[-1]
        return recon, tokens[key], grids[key]

    @torch.no_grad()
    def encode_teacher_tokens(self, x_clean: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, Tuple[int, int, int]]]:
        _x_out, hidden_states = self.teacher_encoder(x_clean)
        if not hidden_states:
            raise RuntimeError("teacher_encoder returned no hidden_states")
        spatial = tuple(x_clean.shape[-3:])
        tokens: Dict[str, torch.Tensor] = {}
        grids: Dict[str, Tuple[int, int, int]] = {}
        for stage in self.stages:
            key = f"stage{stage}"
            tok, grid = _extract_stage_tokens(hidden_states, stage, spatial)
            tokens[key] = tok
            grids[key] = grid
        return tokens, grids

    @torch.no_grad()
    def encode_teacher_tokens_v1(self, x_clean: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        tokens, grids = self.encode_teacher_tokens(x_clean)
        key = self.stage_keys[-1]
        return tokens[key], grids[key]

    @torch.no_grad()
    def update_teacher_ema(self, momentum: float) -> None:
        m = float(momentum)
        student = self.student_net.module if hasattr(self.student_net, "module") else self.student_net
        for ps, pt in zip(
            student.unetr_pp_encoder.parameters(),
            self.teacher_encoder.parameters(),
        ):
            pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)


def build_unetrpp_inpainting_feature_reconstruction(cfg: Dict[str, Any]) -> UNETRPPInpaintingFeatureReconstruction:
    mcfg = dict((cfg or {}).get("model") or {})
    fcfg = (cfg or {}).get("feature_reconstruction") or {}
    version = resolve_feature_recon_version(fcfg)
    stages = resolve_feature_stages(fcfg)
    variant = str(
        mcfg.get("unetrpp_official_variant")
        or (mcfg.get("unetrpp") or {}).get("official_variant")
        or "synapse"
    ).lower()
    ut: Dict[str, Any] = {
        "in_channels": int(mcfg.get("in_channels", 1)),
        "out_channels": int(mcfg.get("out_channels", 1)),
        "preferred_source": "official",
        "unetrpp_official_variant": variant,
        "is_3d": True,
        "feature_size": int(mcfg.get("feature_size", 16)),
        "unetrpp": dict(mcfg.get("unetrpp") or {}),
    }
    ut["unetrpp"]["do_ds"] = False
    sp = mcfg.get("spatial_size") or mcfg.get("img_size")
    if sp is None:
        sp = list(OFFICIAL_VARIANT_SPATIAL.get(variant, (96, 96, 96)))
    ut["spatial_size"] = list(sp)
    if variant == "synapse":
        ut["img_size"] = list(sp)

    net = build_official_unetrpp3d(ut)
    dims = list((mcfg.get("unetrpp") or {}).get("dims") or [32, 64, 128, 256])
    stage_dims = {f"stage{s}": int(dims[STAGE_TO_HIDDEN_IDX[s]]) for s in stages}
    for stage in stages:
        print(
            f"[feature-recon-v2] stage{stage} source: {STAGE_SOURCE_DESC[stage]} "
            f"(C={stage_dims[f'stage{stage}']})",
            file=sys.stderr,
            flush=True,
        )
    print(f"[feature-recon] version={version} stages={stages}", file=sys.stderr, flush=True)

    return UNETRPPInpaintingFeatureReconstruction(
        net,
        stages=stages,
        stage_dims=stage_dims,
        predictor_hidden_dim=int(fcfg.get("predictor_hidden_dim", 512)),
        version=version,
    )


def format_feature_recon_config_log(feat_cfg: Mapping[str, Any]) -> str:
    feat_cfg = apply_objective_preset(dict(feat_cfg or {}))
    version = resolve_feature_recon_version(feat_cfg)
    stages = resolve_feature_stages(feat_cfg)
    stage_weights = resolve_stage_weights(feat_cfg, stages)
    w_parts = ",".join(f"{k}:{stage_weights[k]:g}" for k in sorted(stage_weights))
    token_mask_mode = str(feat_cfg.get("token_mask_mode", "any"))
    lam = float(feat_cfg.get("lambda_feature", 0.1))
    warmup = int(
        feat_cfg.get("warmup_epochs")
        or feat_cfg.get("lambda_feature_warmup_epochs")
        or 0
    )
    t_base = float(feat_cfg.get("teacher_momentum_base", 0.996))
    t_final = float(feat_cfg.get("teacher_momentum_final", 1.0))
    stages_str = ",".join(str(s) for s in stages)
    loss_kind = resolve_feature_loss_kind(feat_cfg)
    hard_cfg = resolve_hard_token_mining_cfg(feat_cfg)
    preset = str(feat_cfg.get("objective_preset", "") or "")
    lam_sched = str(feat_cfg.get("lambda_feature_schedule", "linear_warmup"))
    if version in ("v4_hard_stage34_smoothl1", "v4_hard_stage234_smoothl1", "v5_adaptive_boundary"):
        loss_w = resolve_feature_loss_weights(feat_cfg)
        mining_desc = (
            f"error_mass_fraction={hard_cfg['error_mass_fraction']:g}"
            if hard_cfg["mode"] == "error_mass"
            else f"hard_topk_ratio={hard_cfg['topk_ratio']:g}"
        )
        return (
            f"[feature-recon-config] version={version} preset={preset or 'none'} stages=[{stages_str}] "
            f"stage_weights={{{w_parts}}} token_mask_mode={token_mask_mode} "
            f"loss={loss_kind} lambda={lam:g} lambda_schedule={lam_sched} warmup_epochs={warmup} "
            f"mining_mode={hard_cfg['mode']} {mining_desc} "
            f"curriculum_epochs={hard_cfg['curriculum_epochs']} "
            f"cos_w={loss_w['cosine_weight']:g} smooth_w={loss_w['smooth_l1_weight']:g} "
            f"teacher_momentum={t_base:g}->{t_final:g}"
        )
    if version == "v4_hard_stage34":
        return (
            f"[feature-recon-config] version={version} stages=[{stages_str}] "
            f"stage_weights={{{w_parts}}} token_mask_mode={token_mask_mode} "
            f"loss={loss_kind} lambda={lam:g} warmup_epochs={warmup} "
            f"hard_topk_ratio={hard_cfg['topk_ratio']:g} "
            f"teacher_momentum_base={t_base:g} teacher_momentum_final={t_final:g}"
        )
    return (
        f"[feature-recon-config] version={version} stages=[{stages_str}] "
        f"stage_weights={{{w_parts}}} token_mask_mode={token_mask_mode} "
        f"lambda={lam:g} warmup_epochs={warmup} "
        f"teacher_momentum_base={t_base:g} teacher_momentum_final={t_final:g}"
    )


def lambda_feature_for_epoch(
    feat_cfg: dict,
    epoch: int,
    *,
    total_epochs: Optional[int] = None,
) -> Tuple[float, float]:
    base = float(feat_cfg.get("lambda_feature", 0.1))
    schedule = str(feat_cfg.get("lambda_feature_schedule", "linear_warmup")).strip().lower()
    warmup = int(
        feat_cfg.get("warmup_epochs")
        or feat_cfg.get("lambda_feature_warmup_epochs")
        or 0
    )
    if schedule in ("cosine", "cosine_warmup", "cosine_decay"):
        from dinomim_pytorch.training_schedules import cosine_schedule

        te = int(total_epochs or feat_cfg.get("total_epochs") or 0)
        if te <= 0:
            te = max(warmup, epoch + 1)
        if warmup > 0 and epoch < warmup:
            now = base * float(epoch + 1) / float(warmup)
        else:
            start = base if warmup <= 0 else base
            end = float(feat_cfg.get("lambda_feature_final", base * 0.5))
            rem = max(1, te - max(warmup, 0))
            step = max(0, epoch - max(warmup - 1, 0))
            now = cosine_schedule(step, rem, start, end)
    elif warmup > 0:
        now = base * min(1.0, float(epoch + 1) / float(warmup))
    else:
        now = base
    return base, now


def token_std_raw(tokens: torch.Tensor) -> float:
    if tokens.numel() == 0:
        return 0.0
    return float(tokens.float().std(dim=-1).mean().detach())


def predictor_grad_norm(predictor: Union[FeaturePredictor, MultiStageFeaturePredictors, nn.Module]) -> float:
    sq = 0.0
    for p in predictor.parameters():
        if p.grad is not None:
            sq += float(p.grad.data.pow(2).sum())
    return sq ** 0.5 if sq > 0 else 0.0


def predictor_grad_norm_per_stage(predictors: MultiStageFeaturePredictors) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, pred in predictors.predictors.items():
        out[f"pred_grad_{key}"] = predictor_grad_norm(pred)
    return out


def feature_collapse_warning(epoch: int, meta: Dict[str, float]) -> None:
    if epoch + 1 < 3:
        return
    for key, val in meta.items():
        if (key.startswith("s_std_") or key.startswith("t_std_") or key in (
            "student_token_std_raw", "teacher_token_std_raw"
        )) and float(val) < 0.05:
            print(
                f"WARNING: feature representation may be collapsing ({key}={float(val):.4f})",
                file=sys.stderr,
                flush=True,
            )


def save_feature_recon_checkpoint(
    path,
    *,
    model: UNETRPPInpaintingFeatureReconstruction,
    scheme: str,
    epoch: int,
    best_loss: Optional[float],
    cfg: dict,
    optimizer=None,
    global_step: Optional[int] = None,
) -> None:
    student = model.student_net.module if hasattr(model.student_net, "module") else model.student_net
    predictors = (
        model.feature_predictors.module
        if hasattr(model.feature_predictors, "module")
        else model.feature_predictors
    )
    pred_state = {k: v.state_dict() for k, v in predictors.predictors.items()}
    payload: Dict[str, Any] = {
        "student_backbone": student.state_dict(),
        "student_feature_predictors": pred_state,
        "teacher_encoder": model.teacher_encoder.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
        "cfg": cfg,
        "scheme": scheme,
        "feature_recon_version": model.version,
        "feature_recon_stages": model.stages,
    }
    if "stage4" in pred_state:
        payload["student_feature_predictor"] = pred_state["stage4"]
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if global_step is not None:
        payload["global_step"] = global_step
    torch.save(payload, str(path))


__all__ = [
    "FeaturePredictor",
    "MultiStageFeaturePredictors",
    "UNETRPPInpaintingFeatureReconstruction",
    "build_unetrpp_inpainting_feature_reconstruction",
    "masked_cosine_feature_loss",
    "masked_feature_reconstruction_loss",
    "multiscale_feature_loss",
    "resolve_feature_loss_kind",
    "resolve_hard_token_mining_cfg",
    "resolve_feature_loss_weights",
    "random_block_mask_and_indicator",
    "voxel_mask_to_token_mask",
    "tokens_from_stage_feat",
    "tokens_from_stage4_feat",
    "lambda_feature_for_epoch",
    "token_std_raw",
    "predictor_grad_norm",
    "predictor_grad_norm_per_stage",
    "feature_collapse_warning",
    "save_feature_recon_checkpoint",
    "resolve_feature_recon_version",
    "resolve_feature_stages",
    "resolve_stage_weights",
    "format_feature_recon_config_log",
    "is_multiscale_feature_recon",
    "STAGE_TO_HIDDEN_IDX",
    "STAGE_SOURCE_DESC",
]
