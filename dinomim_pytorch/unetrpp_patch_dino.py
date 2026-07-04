"""
UNETR++ patch-token DINO / iBOT-style SSL (separate from global-pooled DINO + inpainting).

Modes:
  - patch_dino_only: encoder patch tokens + PatchDINOHead, no reconstruction
  - inpainting_patch_dino: full UNETR++ reconstruction + patch-token DINO branch
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from dinomim_pytorch.dino_heads import DINOHead
from dinomim_pytorch.segmentation_models.official_unetrpp3d import (
    OFFICIAL_VARIANT_SPATIAL,
    build_official_unetrpp3d,
)
from dinomim_pytorch.unetrpp_dino_inpainting import inpainting_recon_loss


def _factor_token_grid(n_tokens: int, spatial: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Infer (Dp, Hp, Wp) with Dp*Hp*Wp == n_tokens, aspect ratio ~ spatial."""
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


def extract_unetrpp_patch_tokens(
    encoder: nn.Module,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    Last EPA stage → patch tokens ``[B, N, C]`` (no global pooling).

    Returns ``(tokens, (Dp, Hp, Wp))``.
    """
    _x_out, hidden_states = encoder(x)
    if not hidden_states:
        raise RuntimeError("unetr_pp_encoder returned no hidden_states")
    feat = hidden_states[-1]
    if feat.dim() == 5:
        _b, _c, dp, hp, wp = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        return tokens, (int(dp), int(hp), int(wp))
    if feat.dim() == 3:
        b, n, c = feat.shape
        grid = _factor_token_grid(int(n), tuple(x.shape[-3:]))
        return feat, grid
    if feat.dim() == 2:
        return feat.unsqueeze(1), (1, 1, 1)
    raise RuntimeError(f"Unexpected encoder feature shape {tuple(feat.shape)}")


class PatchDINOHead(nn.Module):
    """DINO head on patch tokens ``[B, N, C]`` → ``[B, N, K]``."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 1024,
        bottleneck_dim: int = 256,
        use_bn_in_head: bool = False,
        norm_last_layer: bool = True,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.head = DINOHead(
            in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            n_layers=n_layers,
            use_bn_in_head=use_bn_in_head,
            norm_last_layer=norm_last_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"PatchDINOHead expects [B,N,C], got {tuple(x.shape)}")
        b, n, c = x.shape
        flat = x.reshape(b * n, c)
        logits = self.head(flat)
        return logits.reshape(b, n, -1)

    @property
    def last_layer(self) -> nn.Module:
        return self.head.last_layer


def patch_dino_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    student_temp: float = 0.1,
    teacher_temp: float = 0.04,
    center: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Patch-level DINO cross-entropy between teacher softmax and student log-softmax.

    Shapes: ``[B, N, K]``; optional ``mask`` ``[B, N]`` (1 = include).
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"student/teacher patch logits shape mismatch: "
            f"{tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}"
        )
    c = center
    if c is not None:
        if c.dim() == 2:
            c = c.unsqueeze(1)
        teacher_logits = teacher_logits - c

    t_probs = F.softmax(teacher_logits / float(teacher_temp), dim=-1).detach()
    s_log = F.log_softmax(student_logits / float(student_temp), dim=-1)
    per_token = (-t_probs * s_log).sum(dim=-1)

    if mask is not None:
        m = mask.float()
        loss = (per_token * m).sum() / m.sum().clamp_min(1e-6)
        mask_frac = float(m.mean().detach())
    else:
        loss = per_token.mean()
        mask_frac = 1.0

    with torch.no_grad():
        t_ent = (-t_probs * (t_probs.clamp_min(1e-8).log())).sum(dim=-1).mean()
        s_probs = F.softmax(student_logits / float(student_temp), dim=-1)
        s_ent = (-s_probs * (s_probs.clamp_min(1e-8).log())).sum(dim=-1).mean()

    meta = {
        "teacher_patch_entropy": float(t_ent.detach()),
        "student_patch_entropy": float(s_ent.detach()),
        "mask_token_fraction": mask_frac,
    }
    return loss, meta


class PatchDINOLoss(nn.Module):
    """Patch DINO loss with EMA teacher center ``[1, 1, K]``."""

    def __init__(
        self,
        out_dim: int,
        *,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
        warmup_teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 0,
    ) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.student_temp = float(student_temp)
        self.teacher_temp = float(teacher_temp)
        self.center_momentum = float(center_momentum)
        self.warmup_teacher_temp = float(warmup_teacher_temp)
        self.warmup_teacher_temp_epochs = int(warmup_teacher_temp_epochs)
        self.register_buffer("center", torch.zeros(1, 1, self.out_dim))
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def current_teacher_temp(self) -> float:
        return get_teacher_temp(self._epoch, self._patch_cfg_dict())

    def _patch_cfg_dict(self) -> Dict[str, Any]:
        return {
            "warmup_teacher_temp": self.warmup_teacher_temp,
            "teacher_temp": self.teacher_temp,
            "warmup_teacher_temp_epochs": self.warmup_teacher_temp_epochs,
        }

    @torch.no_grad()
    def update_center(self, teacher_logits: torch.Tensor) -> None:
        batch_center = teacher_logits.mean(dim=(0, 1), keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1.0 - self.center_momentum)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss, meta = patch_dino_loss(
            student_logits,
            teacher_logits,
            student_temp=self.student_temp,
            teacher_temp=self.current_teacher_temp(),
            center=self.center,
            mask=mask,
        )
        self.update_center(teacher_logits)
        meta["teacher_temp"] = self.current_teacher_temp()
        meta["center_norm"] = float(self.center.norm().detach())
        return loss, meta


def voxel_mask_to_patch_mask(
    voxel_mask: torch.Tensor,
    token_grid: Tuple[int, int, int],
) -> torch.Tensor:
    """Downsample voxel mask ``[B,1,D,H,W]`` or ``[B,D,H,W]`` → ``[B,N]``."""
    if voxel_mask.dim() == 4:
        voxel_mask = voxel_mask.unsqueeze(1)
    dp, hp, wp = token_grid
    m = F.interpolate(voxel_mask.float(), size=(dp, hp, wp), mode="nearest")
    return m.flatten(2).squeeze(1)


def _primary_logits(raw: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]) -> torch.Tensor:
    if isinstance(raw, (list, tuple)):
        return raw[0]
    return raw


class UNETRPPPatchDINO(nn.Module):
    """
    Patch-token DINO on official UNETR++ EPA encoder.

    ``enable_reconstruction=True`` adds full student UNETR++ decoder path for inpainting.
    """

    def __init__(
        self,
        student_net: nn.Module,
        *,
        out_dim: int,
        hidden_dim: int = 1024,
        bottleneck_dim: int = 256,
        use_bn_in_head: bool = False,
        norm_last_layer: bool = True,
        n_head_layers: int = 3,
        embed_dim: Optional[int] = None,
        enable_reconstruction: bool = False,
    ) -> None:
        super().__init__()
        if not hasattr(student_net, "unetr_pp_encoder"):
            raise AttributeError("student_net must expose unetr_pp_encoder (official UNETR++)")
        self.student_net = student_net
        self.enable_reconstruction = bool(enable_reconstruction)
        self.embed_dim = int(embed_dim or getattr(student_net, "hidden_size", 256))
        self.out_dim = int(out_dim)

        self.student_patch_head = PatchDINOHead(
            self.embed_dim,
            out_dim=self.out_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            use_bn_in_head=use_bn_in_head,
            norm_last_layer=norm_last_layer,
            n_layers=n_head_layers,
        )
        self.teacher_encoder = copy.deepcopy(student_net.unetr_pp_encoder)
        self.teacher_patch_head = PatchDINOHead(
            self.embed_dim,
            out_dim=self.out_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            use_bn_in_head=use_bn_in_head,
            norm_last_layer=norm_last_layer,
            n_layers=n_head_layers,
        )
        self.teacher_encoder.load_state_dict(student_net.unetr_pp_encoder.state_dict())
        self.teacher_patch_head.load_state_dict(self.student_patch_head.state_dict())
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False
        for p in self.teacher_patch_head.parameters():
            p.requires_grad = False

    def encode_student_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        return extract_unetrpp_patch_tokens(self.student_net.unetr_pp_encoder, x)

    @torch.no_grad()
    def encode_teacher_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        return extract_unetrpp_patch_tokens(self.teacher_encoder, x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enable_reconstruction:
            raise RuntimeError("reconstruction disabled (patch_dino_only mode)")
        return _primary_logits(self.student_net(x))

    def forward_patch_dino(
        self,
        student_views: List[torch.Tensor],
        teacher_views: List[torch.Tensor],
        loss_mod: PatchDINOLoss,
        *,
        loss_on: str = "all_tokens",
        student_voxel_masks: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if len(student_views) != 1 or len(teacher_views) != 1:
            raise ValueError("forward_patch_dino (v1) expects one student and one teacher view")
        s_x, t_x = student_views[0], teacher_views[0]
        s_tokens, s_grid = self.encode_student_tokens(s_x)
        with torch.no_grad():
            t_tokens, t_grid = self.encode_teacher_tokens(t_x)
        if s_grid != t_grid:
            raise ValueError(f"Teacher/student token grids differ: {s_grid} vs {t_grid}")

        s_logits = self.student_patch_head(s_tokens)
        with torch.no_grad():
            t_logits = self.teacher_patch_head(t_tokens)

        patch_mask = None
        if loss_on == "masked_tokens":
            if student_voxel_masks is None or not student_voxel_masks[0].numel():
                raise ValueError("loss_on=masked_tokens requires student_voxel_masks")
            patch_mask = voxel_mask_to_patch_mask(student_voxel_masks[0], s_grid)

        loss, meta = loss_mod(s_logits, t_logits, mask=patch_mask)
        meta.update(
            {
                "student_tokens_shape": tuple(s_tokens.shape),
                "teacher_tokens_shape": tuple(t_tokens.shape),
                "student_patch_logits_shape": tuple(s_logits.shape),
                "teacher_patch_logits_shape": tuple(t_logits.shape),
                "patch_mask_shape": tuple(patch_mask.shape) if patch_mask is not None else (),
                "token_grid": s_grid,
            }
        )
        return loss, meta

    @torch.no_grad()
    def update_teacher_ema(self, momentum: float) -> None:
        m = float(momentum)
        for ps, pt in zip(
            self.student_net.unetr_pp_encoder.parameters(),
            self.teacher_encoder.parameters(),
        ):
            pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)
        for ps, pt in zip(self.student_patch_head.parameters(), self.teacher_patch_head.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)


def _build_student_net(cfg: Dict[str, Any], *, enable_reconstruction: bool) -> nn.Module:
    mcfg = dict((cfg or {}).get("model") or {})
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
    if not enable_reconstruction:
        ut["out_channels"] = 1
    return build_official_unetrpp3d(ut)


def build_unetrpp_patch_dino(cfg: Dict[str, Any], *, enable_reconstruction: bool = False) -> UNETRPPPatchDINO:
    mcfg = (cfg or {}).get("model") or {}
    net = _build_student_net(cfg, enable_reconstruction=enable_reconstruction)
    return UNETRPPPatchDINO(
        net,
        out_dim=int(mcfg.get("out_dim", 128)),
        hidden_dim=int(mcfg.get("hidden_dim", 1024)),
        bottleneck_dim=int(mcfg.get("bottleneck_dim", 256)),
        use_bn_in_head=bool(mcfg.get("use_bn_in_head", False)),
        norm_last_layer=bool(mcfg.get("norm_last_layer", True)),
        embed_dim=int(mcfg.get("hidden_size", getattr(net, "hidden_size", 256))),
        enable_reconstruction=enable_reconstruction,
    )


def build_unetrpp_patch_dino_only(cfg: Dict[str, Any]) -> UNETRPPPatchDINO:
    return build_unetrpp_patch_dino(cfg, enable_reconstruction=False)


def build_unetrpp_inpainting_patch_dino(cfg: Dict[str, Any]) -> UNETRPPPatchDINO:
    return build_unetrpp_patch_dino(cfg, enable_reconstruction=True)


def _aug_no_crop(x: torch.Tensor, *, noise_std: float, flip_prob: float, gamma_jitter: float = 0.0) -> torch.Tensor:
    """Intensity/spatial flip aug without re-cropping (paired-crop mode)."""
    v = x
    if gamma_jitter > 0:
        g = 1.0 + (torch.rand(1, device=v.device, dtype=v.dtype) * 2 - 1) * gamma_jitter
        v = torch.sign(v) * (v.abs() + 1e-6).pow(g.item())
    if noise_std > 0:
        v = v + noise_std * torch.randn_like(v)
    if torch.rand(1).item() < flip_prob:
        v = torch.flip(v, (-1,))
    if torch.rand(1).item() < flip_prob:
        v = torch.flip(v, (-2,))
    if torch.rand(1).item() < flip_prob:
        v = torch.flip(v, (-3,))
    return v.clamp(-6.0, 6.0)


def prepare_patch_dino_views(
    batch: Dict[str, Any],
    *,
    device: torch.device,
    spatial: Tuple[int, int, int],
    patch_cfg: dict,
    teacher_cfg: dict,
    student_cfg: dict,
    patch_size: int = 16,
    mask_value: float = 0.0,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], Optional[List[torch.Tensor]]]:
    """
    Build aligned teacher/student views for patch-DINO.

    Returns ``(teacher_views, student_views, student_voxel_masks)``.
    """
    from dinomim_pytorch.dense_ssl import random_block_mask_nd

    d, h, w = spatial

    def _to_model(t: torch.Tensor) -> torch.Tensor:
        t = t.to(device, non_blocking=True)
        if t.dim() != 5:
            raise ValueError(f"Expected 5D [B,C,D,H,W], got {tuple(t.shape)}")
        if tuple(t.shape[-3:]) == (d, h, w):
            return t
        return F.interpolate(t, size=(d, h, w), mode="trilinear", align_corners=False)

    same_crop = bool(patch_cfg.get("same_crop_for_teacher_student", True))
    t_mr = float(patch_cfg.get("teacher_mask_ratio", 0.0))
    s_mr = float(patch_cfg.get("student_mask_ratio", 0.3))

    if same_crop:
        x_base = batch.get("volume")
        if x_base is None:
            x_base = batch["teacher_glob"][0]
        x_base = _to_model(x_base)
        tg = teacher_cfg.get("global") or {}
        sg = student_cfg.get("global") or {}
        x_t = _aug_no_crop(x_base, noise_std=0.02, flip_prob=0.5)
        x_s = _aug_no_crop(
            x_base,
            noise_std=0.08,
            flip_prob=0.5,
            gamma_jitter=0.15 if str(sg.get("augmentation_strength", "strong")).lower() == "strong" else 0.0,
        )
        if t_mr > 0:
            x_t = random_block_mask_nd(x_t, mask_ratio=t_mr, patch_size=patch_size, mask_value=mask_value)
        student_voxel_masks = []
        if s_mr > 0:
            x_s, m = _mask_with_indicator(x_s, mask_ratio=s_mr, patch_size=patch_size, mask_value=mask_value)
            student_voxel_masks.append(m)
        else:
            student_voxel_masks.append(torch.zeros(x_s.shape[0], 1, d, h, w, device=x_s.device))
        return [x_t], [x_s], student_voxel_masks

    teacher_views = [_to_model(batch["teacher_glob"][0])]
    if t_mr > 0:
        teacher_views = [
            random_block_mask_nd(teacher_views[0], mask_ratio=t_mr, patch_size=patch_size, mask_value=mask_value)
        ]
    x_s = _to_model(batch["student_glob"][0])
    student_voxel_masks = []
    if s_mr > 0:
        x_s, m = _mask_with_indicator(x_s, mask_ratio=s_mr, patch_size=patch_size, mask_value=mask_value)
        student_voxel_masks.append(m)
    else:
        student_voxel_masks.append(torch.zeros(x_s.shape[0], 1, d, h, w, device=x_s.device))
    return teacher_views, [x_s], student_voxel_masks


def _mask_with_indicator(
    x: torch.Tensor,
    *,
    mask_ratio: float,
    patch_size: int,
    mask_value: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from dinomim_pytorch.dense_ssl import random_block_mask_nd

    masked = random_block_mask_nd(x, mask_ratio=mask_ratio, patch_size=patch_size, mask_value=mask_value)
    m = (masked == float(mask_value)).float()
    if m.dim() == 5:
        return masked, m
    return masked, m.unsqueeze(1)


def get_teacher_temp(epoch: int, patch_cfg: dict) -> float:
    """Linear warmup of teacher softmax temperature over ``warmup_teacher_temp_epochs``."""
    warmup_temp = float(patch_cfg.get("warmup_teacher_temp", 0.04))
    final_temp = float(patch_cfg.get("teacher_temp", 0.07))
    warmup_epochs = int(patch_cfg.get("warmup_teacher_temp_epochs", 0) or 0)
    if warmup_epochs <= 0:
        return final_temp
    alpha = min(1.0, float(epoch + 1) / float(warmup_epochs))
    return warmup_temp + alpha * (final_temp - warmup_temp)


def lambda_patch_dino_for_epoch(patch_cfg: dict, epoch: int) -> Tuple[float, float]:
    base = float(patch_cfg.get("lambda_patch_dino", 1.0))
    warmup = int(patch_cfg.get("lambda_patch_dino_warmup_epochs", 0) or 0)
    if warmup > 0:
        now = base * min(1.0, float(epoch + 1) / float(warmup))
    else:
        now = base
    return base, now


def token_std(tokens: torch.Tensor) -> float:
    if tokens.numel() == 0:
        return 0.0
    return float(tokens.float().std(dim=-1).mean().detach())


def logits_std(logits: torch.Tensor) -> float:
    if logits.numel() == 0:
        return 0.0
    return float(logits.float().std(dim=-1).mean().detach())


def patch_head_grad_norm(head: PatchDINOHead) -> float:
    sq = 0.0
    for p in head.parameters():
        if p.grad is not None:
            sq += float(p.grad.data.pow(2).sum())
    return math.sqrt(sq) if sq > 0 else 0.0


def set_patch_head_last_layer_frozen(head: PatchDINOHead, frozen: bool) -> None:
    for p in head.last_layer.parameters():
        p.requires_grad = not frozen


def patch_head_last_layer_frozen(head: PatchDINOHead) -> bool:
    params = list(head.last_layer.parameters())
    return bool(params) and not params[0].requires_grad


@torch.no_grad()
def patch_proto_stats(logits: torch.Tensor, *, temp: float = 1.0) -> Dict[str, float]:
    """Prototype usage diagnostics from patch logits ``[B, N, K]``."""
    probs = F.softmax(logits / float(temp), dim=-1)
    entropy = (-probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
    perplexity = torch.exp(entropy)
    top1 = probs.argmax(dim=-1).reshape(-1)
    k = int(logits.shape[-1])
    unique_proto = int(top1.unique().numel())
    max_prob_mean = probs.max(dim=-1).values.mean()
    counts = torch.bincount(top1, minlength=k).float()
    top1_frac = float((counts.max() / counts.sum().clamp_min(1.0)).item())
    return {
        "entropy": float(entropy.item()),
        "perplexity": float(perplexity.item()),
        "unique_proto": float(unique_proto),
        "max_prob_mean": float(max_prob_mean.item()),
        "top1_frac": top1_frac,
    }


def patch_dino_step_diagnostics(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    *,
    student_temp: float,
    teacher_temp: float,
) -> Dict[str, float]:
    s = patch_proto_stats(s_logits, temp=student_temp)
    t = patch_proto_stats(t_logits, temp=teacher_temp)
    return {
        "student_unique_proto": s["unique_proto"],
        "teacher_unique_proto": t["unique_proto"],
        "student_proto_perplexity": s["perplexity"],
        "teacher_proto_perplexity": t["perplexity"],
        "student_max_prob_mean": s["max_prob_mean"],
        "teacher_max_prob_mean": t["max_prob_mean"],
        "student_top1_frac": s["top1_frac"],
        "teacher_top1_frac": t["top1_frac"],
    }


PATCH_DINO_EXTRA_METRICS = [
    "teacher_temp_now",
    "student_unique_proto",
    "teacher_unique_proto",
    "student_proto_perplexity",
    "teacher_proto_perplexity",
    "student_max_prob_mean",
    "teacher_max_prob_mean",
    "student_top1_frac",
    "teacher_top1_frac",
    "last_layer_frozen",
]


def save_patch_dino_checkpoint(
    path,
    *,
    model: UNETRPPPatchDINO,
    loss_mod: PatchDINOLoss,
    scheme: str,
    epoch: int,
    best_loss: Optional[float],
    cfg: dict,
    optimizer=None,
    global_step: Optional[int] = None,
    extra: Optional[dict] = None,
) -> None:
    if model.enable_reconstruction:
        student_backbone_sd = model.student_net.state_dict()
    else:
        student_backbone_sd = model.student_net.unetr_pp_encoder.state_dict()
    payload: Dict[str, Any] = {
        "student_backbone": student_backbone_sd,
        "student_patch_head": model.student_patch_head.state_dict(),
        "teacher_encoder": model.teacher_encoder.state_dict(),
        "teacher_patch_head": model.teacher_patch_head.state_dict(),
        "patch_dino_loss": loss_mod.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
        "cfg": cfg,
        "scheme": scheme,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if global_step is not None:
        payload["global_step"] = global_step
    if extra:
        payload.update(extra)
    torch.save(payload, str(path))


def print_patch_dino_startup(
    *,
    mode: str,
    spatial: Tuple[int, int, int],
    out_dim: int,
    patch_cfg: dict,
    reconstruction_enabled: bool,
) -> None:
    import sys

    print(
        f"[patch-dino] mode: {mode}\n"
        f"[patch-dino] token shape: [B, N, C] (spatial grid from EPA stage-4)\n"
        f"[patch-dino] out_dim: {out_dim}\n"
        f"[patch-dino] loss_on: {patch_cfg.get('loss_on', 'all_tokens')}\n"
        f"[patch-dino] same_crop_for_teacher_student: {patch_cfg.get('same_crop_for_teacher_student', True)}\n"
        f"[patch-dino] reconstruction enabled: {reconstruction_enabled}\n"
        f"[patch-dino] spatial: {spatial}",
        file=sys.stderr,
        flush=True,
    )


def sharpness_warning(epoch: int, meta: Dict[str, float]) -> None:
    import sys

    if epoch + 1 < 2:
        return
    t_ent = float(meta.get("teacher_patch_entropy", meta.get("teacher_entropy", 1.0)))
    t_max = float(meta.get("teacher_max_prob_mean", 0.0))
    if t_ent < 0.2 or t_max > 0.95:
        print(
            "WARNING: Patch-DINO teacher targets are too sharp. "
            "Consider increasing teacher_temp, increasing warmup_teacher_temp_epochs, "
            "or freezing the last layer longer.",
            file=sys.stderr,
            flush=True,
        )


def collapse_warning(epoch: int, meta: Dict[str, float]) -> None:
    import sys

    if epoch + 1 < 3:
        return
    s_std = float(meta.get("student_token_std", 0.0))
    t_std = float(meta.get("teacher_token_std", 0.0))
    if s_std < 0.05 and t_std < 0.05:
        print(
            "WARNING: Patch-DINO token features appear collapsed.",
            file=sys.stderr,
            flush=True,
        )


__all__ = [
    "UNETRPPPatchDINO",
    "PatchDINOHead",
    "PatchDINOLoss",
    "patch_dino_loss",
    "extract_unetrpp_patch_tokens",
    "build_unetrpp_patch_dino_only",
    "build_unetrpp_inpainting_patch_dino",
    "prepare_patch_dino_views",
    "voxel_mask_to_patch_mask",
    "lambda_patch_dino_for_epoch",
    "get_teacher_temp",
    "save_patch_dino_checkpoint",
    "print_patch_dino_startup",
    "collapse_warning",
    "sharpness_warning",
    "token_std",
    "logits_std",
    "patch_head_grad_norm",
    "set_patch_head_last_layer_frozen",
    "patch_head_last_layer_frozen",
    "patch_proto_stats",
    "patch_dino_step_diagnostics",
    "PATCH_DINO_EXTRA_METRICS",
]
