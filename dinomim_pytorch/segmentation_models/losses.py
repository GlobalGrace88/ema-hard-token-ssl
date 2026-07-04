"""
Segmentation losses (MONAI Dice, Dice+CE, Dice+Focal) and factory from YAML-style config.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from monai.losses import DiceCELoss, DiceFocalLoss, DiceLoss
except Exception:  # noqa: BLE001
    DiceLoss = None  # type: ignore[misc, assignment]
    DiceCELoss = None  # type: ignore[misc, assignment]
    DiceFocalLoss = None  # type: ignore[misc, assignment]

__all__ = [
    "dice_loss",
    "dice_ce_loss",
    "dice_focal_loss",
    "dice_ce_boundary_loss",
    "boundary_map_from_labels",
    "build_segmentation_loss",
    "classification_ce",
    "bce_multilabel",
    "as_logits_list",
    "primary_segmentation_logits",
    "align_logits_to_labels",
    "compute_segmentation_loss",
]


def _require_monai(what: str) -> None:
    if DiceLoss is None:
        raise ImportError("MONAI is required for " + what)


def as_logits_list(out: Union[torch.Tensor, List[torch.Tensor], tuple]) -> List[torch.Tensor]:
    """UNETR++ with ``do_ds: true`` returns ``[full_res, mid, coarse]`` logits."""
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (list, tuple)):
        parts = [o for o in out if isinstance(o, torch.Tensor)]
        if parts:
            return parts
    raise TypeError(f"Expected Tensor or list of Tensors from segmentation model, got {type(out)!r}")


def primary_segmentation_logits(out: Union[torch.Tensor, List[torch.Tensor], tuple]) -> torch.Tensor:
    """Full-resolution head (``out[0]`` for official UNETR++ deep supervision)."""
    return as_logits_list(out)[0]


def align_logits_to_labels(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    mode: str = "trilinear",
) -> torch.Tensor:
    if logits.shape[-3:] == y.shape[-3:]:
        return logits
    return F.interpolate(logits, size=y.shape[-3:], mode=mode, align_corners=False)


def compute_segmentation_loss(
    loss_fn: nn.Module,
    out: Union[torch.Tensor, List[torch.Tensor], tuple],
    y: torch.Tensor,
    *,
    interp_mode: str = "trilinear",
) -> torch.Tensor:
    """
    Dice/CE on a single logits tensor or mean loss over deep-supervision heads
    (labels downsampled with nearest-neighbor per scale).
    """
    heads = as_logits_list(out)
    if len(heads) == 1:
        return loss_fn(align_logits_to_labels(heads[0], y, mode=interp_mode), y)
    total = y.new_tensor(0.0)
    for lo in heads:
        yt = y
        if lo.shape[-3:] != y.shape[-3:]:
            yt = F.interpolate(
                y.float(),
                size=lo.shape[-3:],
                mode="nearest",
            ).long()
        total = total + loss_fn(lo, yt)
    return total / len(heads)


def boundary_map_from_labels(
    y: torch.Tensor,
    *,
    num_classes: int,
    kernel: int = 3,
) -> torch.Tensor:
    """Approximate voxel boundary map from integer labels ``[B,1,D,H,W]`` -> ``[B,1,D,H,W]`` float."""
    if y.dim() == 4:
        y = y.unsqueeze(1)
    fg = (y > 0).float()
    if int(num_classes) > 1:
        one_hot = torch.zeros(
            y.shape[0], int(num_classes), *y.shape[-3:], device=y.device, dtype=torch.float32,
        )
        yi = y.long().clamp_min(0).clamp_max(int(num_classes) - 1)
        one_hot.scatter_(1, yi, 1.0)
        fg = one_hot[:, 1:].sum(dim=1, keepdim=True).clamp_max(1.0) if one_hot.shape[1] > 1 else fg
    pad = max(0, int(kernel) // 2)
    dil = F.max_pool3d(fg, kernel_size=kernel, stride=1, padding=pad)
    ero = -F.max_pool3d(-fg, kernel_size=kernel, stride=1, padding=pad)
    return (dil - ero).clamp_min(0.0)


class DiceCEBoundaryLoss(nn.Module):
    """Dice+CE with an auxiliary boundary L1 term on foreground edges."""

    def __init__(self, dice_ce: nn.Module, *, boundary_weight: float = 0.1, num_classes: int = 14) -> None:
        super().__init__()
        self.dice_ce = dice_ce
        self.boundary_weight = float(boundary_weight)
        self.num_classes = int(num_classes)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base = self.dice_ce(pred, target)
        if self.boundary_weight <= 0:
            return base
        if pred.dim() != 5:
            return base
        probs = torch.softmax(pred, dim=1)
        pred_labels = probs.argmax(dim=1, keepdim=True)
        pred_b = boundary_map_from_labels(pred_labels, num_classes=self.num_classes)
        tgt_b = boundary_map_from_labels(target, num_classes=self.num_classes)
        b_loss = F.l1_loss(pred_b, tgt_b)
        return base + self.boundary_weight * b_loss


def build_segmentation_loss(
    config: Optional[Dict[str, Any]] = None,
) -> "nn.Module":
    """
    Build loss from a ``loss`` or ``losses`` config block, e.g.::

        loss:
          name: dice_ce
          include_background: false
          to_onehot_y: true
          softmax: true
    """
    c = (config or {}) if isinstance((config or {}), dict) else {}
    c = c.get("loss", c)
    if not c:
        c = {"name": "dice_ce", "include_background": False, "to_onehot_y": True, "softmax": True}
    name = str(c.get("name", "dice_ce")).lower().replace("-", "_")
    _require_monai("losses")
    if name in ("dice", "dice_loss"):
        return DiceLoss(  # type: ignore[operator, misc]
            softmax=bool(c.get("softmax", True)),
            to_onehot_y=bool(c.get("to_onehot_y", True)),
        )
    if name in ("dice_ce", "dicece", "dice_with_ce", "dicece_loss"):
        dkw = {k: v for k, v in c.items() if k in ("include_background", "to_onehot_y", "softmax", "sigmoid", "lambda_dice", "lambda_ce", "squared_pred", "jaccard")}
        dkw.setdefault("softmax", bool(c.get("softmax", True)))
        dkw.setdefault("to_onehot_y", bool(c.get("to_onehot_y", True)))
        dkw.setdefault("include_background", bool(c.get("include_background", True)))
        return DiceCELoss(**dkw)  # type: ignore[operator, misc, arg-type]
    if name in ("dice_focal", "dice_focal_loss"):
        return DiceFocalLoss(  # type: ignore[operator, misc]
            softmax=bool(c.get("softmax", True)),
            to_onehot_y=bool(c.get("to_onehot_y", True)),
        )
    if name in ("dice_ce_boundary", "dice_ce_with_boundary", "dicece_boundary"):
        dkw = {k: v for k, v in c.items() if k in ("include_background", "to_onehot_y", "softmax", "sigmoid", "lambda_dice", "lambda_ce", "squared_pred", "jaccard")}
        dkw.setdefault("softmax", bool(c.get("softmax", True)))
        dkw.setdefault("to_onehot_y", bool(c.get("to_onehot_y", True)))
        dkw.setdefault("include_background", bool(c.get("include_background", False)))
        base = DiceCELoss(**dkw)  # type: ignore[operator, misc, arg-type]
        return DiceCEBoundaryLoss(
            base,
            boundary_weight=float(c.get("boundary_weight", 0.1)),
            num_classes=int(c.get("num_classes", 14)),
        )
    raise NotImplementedError(f"Unknown loss name: {name!r}")


def dice_loss() -> "nn.Module":
    _require_monai("DiceLoss")
    return DiceLoss(softmax=True)  # type: ignore[operator]


def dice_ce_loss() -> "nn.Module":
    _require_monai("DiceCELoss")
    return DiceCELoss(softmax=True)  # type: ignore[operator]


def dice_focal_loss() -> "nn.Module":
    _require_monai("DiceFocalLoss")
    return DiceFocalLoss(softmax=True)  # type: ignore[operator]


def dice_ce_boundary_loss(**kwargs: Any) -> "nn.Module":
    return build_segmentation_loss({"name": "dice_ce_boundary", **kwargs})


def classification_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target)


def bce_multilabel(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target)
