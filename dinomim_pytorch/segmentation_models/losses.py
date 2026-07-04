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


def classification_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target)


def bce_multilabel(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target)
