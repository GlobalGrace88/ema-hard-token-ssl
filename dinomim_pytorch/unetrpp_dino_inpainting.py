"""
UNETR++ masked reconstruction + DINO encoder consistency (student–teacher).

Student: full official UNETR++ (encoder + decoder) for inpainting.
Teacher: EMA copy of ``unetr_pp_encoder`` + DINO head for prototype targets.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from dinomim_pytorch.dino_heads import DINOHead
from dinomim_pytorch.segmentation_models.official_unetrpp3d import (
    OFFICIAL_VARIANT_SPATIAL,
    build_official_unetrpp3d,
)


def pool_unetrpp_encoder_features(encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Global-pool last EPA encoder stage → ``[B, hidden_size]``."""
    _x_out, hidden_states = encoder(x)
    if not hidden_states:
        raise RuntimeError("unetr_pp_encoder returned no hidden_states")
    h = hidden_states[-1]
    if h.dim() == 3:
        return h.mean(dim=1)
    if h.dim() == 5:
        return h.mean(dim=(2, 3, 4))
    if h.dim() == 2:
        return h
    raise RuntimeError(f"Unexpected encoder feature shape {tuple(h.shape)}")


def _primary_logits(raw: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]) -> torch.Tensor:
    if isinstance(raw, (list, tuple)):
        return raw[0]
    return raw


class UNETRPPDINOInpainting(nn.Module):
    """
    Composite SSL model for ``L_total = L_recon + λ * L_DINO``.

    - ``student_net``: full UNETR++ (``out_channels=1``, ``do_ds=False``)
    - ``teacher_encoder``: EMA ``unetr_pp_encoder`` only
    - ``student_head`` / ``teacher_head``: standard DINO MLP heads on pooled encoder features
    """

    def __init__(
        self,
        student_net: nn.Module,
        *,
        out_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        use_bn_in_head: bool = False,
        norm_last_layer: bool = True,
        n_head_layers: int = 3,
        embed_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not hasattr(student_net, "unetr_pp_encoder"):
            raise AttributeError("student_net must expose unetr_pp_encoder (official UNETR++)")
        self.student_net = student_net
        self.embed_dim = int(embed_dim or getattr(student_net, "hidden_size", 256))
        self.out_dim = int(out_dim)

        self.student_head = DINOHead(
            self.embed_dim,
            out_dim=self.out_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            n_layers=n_head_layers,
            use_bn_in_head=use_bn_in_head,
            norm_last_layer=norm_last_layer,
        )
        self.teacher_encoder = copy.deepcopy(student_net.unetr_pp_encoder)
        self.teacher_head = DINOHead(
            self.embed_dim,
            out_dim=self.out_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            n_layers=n_head_layers,
            use_bn_in_head=use_bn_in_head,
            norm_last_layer=norm_last_layer,
        )
        self.teacher_encoder.load_state_dict(student_net.unetr_pp_encoder.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

    def encode_student(self, x: torch.Tensor) -> torch.Tensor:
        return pool_unetrpp_encoder_features(self.student_net.unetr_pp_encoder, x)

    @torch.no_grad()
    def encode_teacher(self, x: torch.Tensor) -> torch.Tensor:
        return pool_unetrpp_encoder_features(self.teacher_encoder, x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return _primary_logits(self.student_net(x))

    def forward_student_logits(self, views: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.student_head(self.encode_student(v)) for v in views]

    @torch.no_grad()
    def forward_teacher_logits(self, views: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.teacher_head(self.encode_teacher(v)) for v in views]

    @torch.no_grad()
    def update_teacher_ema(self, momentum: float) -> None:
        m = float(momentum)
        for ps, pt in zip(
            self.student_net.unetr_pp_encoder.parameters(),
            self.teacher_encoder.parameters(),
        ):
            pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)
        for ps, pt in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)


def build_unetrpp_dino_inpainting(cfg: Dict[str, Any]) -> UNETRPPDINOInpainting:
    """Build official UNETR++ inpainting student + DINO heads from config."""
    mcfg = dict((cfg or {}).get("model") or {})
    variant = str(
        mcfg.get("unetrpp_official_variant")
        or (mcfg.get("unetrpp") or {}).get("official_variant")
        or "synapse"
    ).lower()
    ut: Dict[str, Any] = {
        "in_channels": int(mcfg.get("in_channels", 1)),
        "out_channels": 1,
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
    return UNETRPPDINOInpainting(
        net,
        out_dim=int(mcfg.get("out_dim", 128)),
        hidden_dim=int(mcfg.get("hidden_dim", 2048)),
        bottleneck_dim=int(mcfg.get("bottleneck_dim", 256)),
        use_bn_in_head=bool(mcfg.get("use_bn_in_head", False)),
        norm_last_layer=bool(mcfg.get("norm_last_layer", True)),
        embed_dim=int(mcfg.get("hidden_size", getattr(net, "hidden_size", 256))),
    )


def inpainting_recon_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    masked_input: torch.Tensor,
    *,
    mask_value: float = 0.0,
    only_masked: bool = True,
) -> torch.Tensor:
    """L1 reconstruction; optionally only on voxels masked in ``masked_input``."""
    if pred.shape[-3:] != target.shape[-3:]:
        pred = F.interpolate(
            pred,
            size=target.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
    if only_masked:
        m = (masked_input == float(mask_value)).float()
        if m.sum() < 1:
            return F.l1_loss(pred, target)
        return (torch.abs(pred - target) * m).sum() / m.sum().clamp_min(1.0)
    return F.l1_loss(pred, target)


__all__ = [
    "UNETRPPDINOInpainting",
    "build_unetrpp_dino_inpainting",
    "pool_unetrpp_encoder_features",
    "inpainting_recon_loss",
]
