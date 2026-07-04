"""
Multi-view masked DINO: student / teacher encoders, DINO heads, EMA teacher (no grad).
Local DINOMiM reference: see repo-root ``utils.py`` (Head, Loss) and ``train.py`` (momentum teacher).
"""

from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from dinomim_pytorch.dino_heads import DINOHead
from dinomim_pytorch.medical_backbones import _guess_first_conv_in_from_patch_embedding


def _module_device(m: nn.Module) -> torch.device:
    return next(m.parameters()).device


class MultiViewMaskedDINO(nn.Module):
    """
    Student and teacher each consist of ``backbone`` + DINO MLP head.
    Teacher is updated by EMA from student outside (see training script). Teacher has ``requires_grad=False``.
    """

    def __init__(
        self,
        student_backbone: nn.Module,
        out_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        teacher_momentum: float = 0.996,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
        use_bn_in_head: bool = False,
        norm_last_layer: bool = True,
        n_head_layers: int = 3,
        feature_dim: Optional[int] = None,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.teacher_momentum = teacher_momentum
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self._head_hidden = int(hidden_dim)
        self._bottleneck = int(bottleneck_dim)
        self._n_head_layers = n_head_layers
        self._use_bn = use_bn_in_head
        self._norm_last = norm_last_layer
        self._fd_override = feature_dim

        self.student_backbone = student_backbone
        dev = _module_device(student_backbone)
        dt = next(student_backbone.parameters()).dtype
        c_in = self._infer_in_chans(student_backbone)
        in_dim = self._infer_in_dim(
            student_backbone, dev, dt, c_in, feature_dim=feature_dim
        )

        self.student_head = DINOHead(
            in_dim,
            out_dim=int(out_dim),
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            n_layers=n_head_layers,
            use_bn_in_head=use_bn_in_head,
            norm_last_layer=norm_last_layer,
        )
        self.teacher_backbone = copy.deepcopy(student_backbone)
        self.teacher_head = DINOHead(
            in_dim,
            out_dim=int(out_dim),
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            n_layers=n_head_layers,
            use_bn_in_head=use_bn_in_head,
            norm_last_layer=norm_last_layer,
        )
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

    @staticmethod
    def _infer_in_chans(m: nn.Module) -> int:
        for step in (
            m,
            getattr(m, "m", None),
            getattr(m, "unet", None),
            getattr(m, "net", None),
        ):
            if step is None:
                continue
            for k in ("in_chans", "in_channels", "input_channels"):
                if hasattr(step, k):
                    v = getattr(step, k)
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        return int(v)
            ic = _guess_first_conv_in_from_patch_embedding(step)
            if ic is not None:
                return ic
        inner = getattr(m, "m", None)
        if inner is not None:
            for sub in (inner, getattr(inner, "unetr", None), getattr(inner, "vit", None)):
                if sub is None:
                    continue
                ic = _guess_first_conv_in_from_patch_embedding(sub)
                if ic is not None:
                    return ic
        return 1

    def _infer_in_dim(
        self,
        backbone: nn.Module,
        dev: torch.device,
        dtype: torch.dtype,
        in_ch: int,
        feature_dim: Optional[int] = None,
    ) -> int:
        if feature_dim and feature_dim > 0:
            return int(feature_dim)
        if self._fd_override:
            return int(self._fd_override)
        mcfg = getattr(self, "_init_image_size", None) or 224
        s = int(mcfg)
        sp = getattr(backbone, "_dinomim_spatial_probe", None)
        if getattr(backbone, "_dinomim_is3d", False):
            if isinstance(sp, (tuple, list)) and len(sp) == 3:
                d, h, w = int(sp[0]), int(sp[1]), int(sp[2])
            else:
                d = h = w = 96
        else:
            d, h, w = 1, s, s
        if getattr(backbone, "_dinomim_is3d", False):
            x = torch.zeros(1, in_ch, d, h, w, device=dev, dtype=dtype)
        else:
            x = torch.zeros(1, in_ch, s, s, device=dev, dtype=dtype)
        with torch.no_grad():
            o = backbone(x)
        while isinstance(o, (list, tuple)) and len(o) > 0:
            o = o[-1]
        if isinstance(o, (list, tuple)) or not isinstance(o, torch.Tensor):
            raise RuntimeError(f"DINO backbone probe expected a Tensor, got {type(o)!r}")
        if o.dim() == 2:
            return int(o.shape[1])
        if o.dim() == 4:
            return int(o.shape[1])
        if o.dim() == 5:
            return int(o.shape[1])
        raise RuntimeError(f"Cannot infer feature dim from shape {tuple(o.shape)}")

    def set_init_spatial(self, size: int) -> None:
        self._init_image_size = int(size)

    def embed_student(self, x: torch.Tensor) -> torch.Tensor:
        return self.student_backbone(x)

    def forward_student_logits(self, views: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.student_head(self.student_backbone(v)) for v in views]

    @torch.no_grad()
    def forward_teacher_logits(self, views: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.teacher_head(self.teacher_backbone(v)) for v in views]

    @torch.no_grad()
    def update_teacher_ema(self, m: float) -> None:
        m = float(m)
        with torch.no_grad():
            for ps, pt in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
                pt.data = pt.data * m + ps.data * (1.0 - m)
            for ps, pt in zip(self.student_head.parameters(), self.teacher_head.parameters()):
                pt.data = pt.data * m + ps.data * (1.0 - m)


__all__ = ["MultiViewMaskedDINO"]
