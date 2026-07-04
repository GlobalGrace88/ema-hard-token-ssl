"""
Segmentation metrics: Dice, IoU, HD95 if MONAI, sensitivity, specificity.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

try:
    from monai.metrics import compute_hausdorff_distance
except Exception:  # noqa: BLE001
    compute_hausdorff_distance = None  # type: ignore[assignment]


def binary_dice_bool(pred_mask: torch.Tensor, true_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Scalar Dice for boolean masks ``pred_mask``, ``true_mask`` (same shape, bool or 0/1 float)."""
    p = pred_mask.float().reshape(pred_mask.size(0), -1)
    t = true_mask.float().reshape(true_mask.size(0), -1)
    inter = (p * t).sum(dim=1)
    denom = p.sum(dim=1) + t.sum(dim=1) + eps
    return (2.0 * inter / denom).mean()


def brats_wt_tc_et_dice(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    softmax: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    BraTS-style region Dice from **discrete** argmax predictions vs labels.

    Assumes labels ``{0,1,2,3}``: 0=background, 1=NCR/NET, 2=ED, 3=ET (common BraTS convention).

    - **WT** (whole tumor): any of {1,2,3}
    - **TC** (tumor core): {1, 3} (necrotic / non-enhancing + enhancing; edema excluded)
    - **ET** (enhancing tumor): {3}
    """
    if softmax:
        pred = logits.argmax(dim=1)
    else:
        pred = logits
    if y.dim() == 5 and y.size(1) == 1:
        y = y[:, 0]
    y = y.long()
    pred = pred.long()

    wt_p = pred > 0
    wt_t = y > 0
    tc_p = (pred == 1) | (pred == 3)
    tc_t = (y == 1) | (y == 3)
    et_p = pred == 3
    et_t = y == 3

    d_wt = binary_dice_bool(wt_p, wt_t)
    d_tc = binary_dice_bool(tc_p, tc_t)
    d_et = binary_dice_bool(et_p, et_t)
    return d_wt, d_tc, d_et


def _pred_and_label_maps(
    y_pred: torch.Tensor, y: torch.Tensor, n_classes: int, softmax: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    if softmax:
        yp = y_pred.argmax(dim=1)
    else:
        yp = y_pred.long()
    y_o = y.long()
    if y_o.dim() == 5 and y_o.size(1) == 1:
        y_o = y_o[:, 0]
    return yp, y_o


def dice_iou_foreground_per_class(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    n_classes: int,
    softmax: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per foreground class Dice and IoU for classes ``1 .. n_classes-1``.

    Returns ``(dice_vec, iou_vec)`` each shape ``(n_classes - 1,)`` on ``y_pred.device``.
    Empty foreground uses scalar ``0`` vectors.
    """
    yp, y_o = _pred_and_label_maps(y_pred, y, n_classes, softmax)
    n_fg = max(0, n_classes - 1)
    if n_fg == 0:
        z = torch.zeros(0, device=y_pred.device, dtype=torch.float32)
        return z, z
    dice_v = torch.zeros(n_fg, device=y_pred.device, dtype=torch.float32)
    iou_v = torch.zeros(n_fg, device=y_pred.device, dtype=torch.float32)
    for i, c in enumerate(range(1, n_classes)):
        dice_v[i] = binary_dice_bool(yp == c, y_o == c)
        p = (yp == c).float()
        t = (y_o == c).float()
        inter = (p * t).sum()
        union_iou = (p + t - p * t).sum() + 1e-6
        iou_v[i] = inter / union_iou
    return dice_v, iou_v


def dice_per_class(
    y_pred: torch.Tensor, y: torch.Tensor, n_classes: int, softmax: bool = True
) -> torch.Tensor:
    """
    Macro mean Dice over foreground classes ``1 .. n_classes-1``.

    Uses **argmax** class predictions vs hard labels (same spirit as ``iou_per_class``),
    not soft Dice on one-hot probabilities (which can read ~0 while IoU / region Dice
    look healthy if the model is confident on the wrong class).
    """
    dice_v, _ = dice_iou_foreground_per_class(y_pred, y, n_classes, softmax=softmax)
    n_fg = max(1, n_classes - 1)
    if dice_v.numel() == 0:
        return torch.zeros((), device=y_pred.device, dtype=torch.float32)
    return dice_v.mean()


def iou_per_class(
    y_pred: torch.Tensor, y: torch.Tensor, n_classes: int, softmax: bool = True
) -> torch.Tensor:
    """Macro mean IoU over foreground classes ``1 .. n_classes-1``."""
    _, iou_v = dice_iou_foreground_per_class(y_pred, y, n_classes, softmax=softmax)
    if iou_v.numel() == 0:
        return torch.zeros((), device=y_pred.device, dtype=torch.float32)
    return iou_v.mean()


def _scatter_one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """``labels`` long ``[B, *spatial]`` -> float one-hot ``[B, C, *spatial]``."""
    labels = labels.long().clamp(min=0, max=num_classes - 1)
    b = labels.shape[0]
    spatial = labels.shape[1:]
    out = torch.zeros((b, num_classes) + spatial, device=labels.device, dtype=torch.float32)
    out.scatter_(1, labels.unsqueeze(1), 1.0)
    return out


def mean_hd95(
    logits: torch.Tensor,
    y: torch.Tensor,
    n_classes: int,
    *,
    softmax: bool = True,
    spacing: Optional[Union[tuple[float, float, float], tuple[float, ...]]] = None,
) -> torch.Tensor:
    """
    Mean 95% Hausdorff distance over foreground channels (background excluded per MONAI).

    Uses argmax predictions vs ``y``. If ``spacing`` is ``(sx, sy, sz)`` voxel spacing in mm,
    distances are in mm; otherwise voxel units (MONAI default spacing 1).

    Returns a scalar tensor; ``nan`` if MONAI is unavailable or no valid surface distances.
    """
    if compute_hausdorff_distance is None or n_classes < 2:
        return torch.tensor(float("nan"), device=logits.device, dtype=torch.float32)
    y = y.long()
    if y.dim() == 5 and y.size(1) == 1:
        y = y[:, 0]
    if softmax:
        pred = logits.argmax(dim=1).long()
    else:
        pred = logits.long()
        if pred.shape != y.shape:
            raise ValueError(f"Hard pred shape {tuple(pred.shape)} must match y {tuple(y.shape)}")
    pred_oh = _scatter_one_hot(pred, n_classes)
    y_oh = _scatter_one_hot(y, n_classes)
    sp = None
    if spacing is not None and len(spacing) >= 3:
        sp = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
    hd = compute_hausdorff_distance(  # type: ignore[misc]
        pred_oh,
        y_oh,
        include_background=False,
        percentile=95.0,
        directed=False,
        spacing=sp,
    )
    return torch.nanmean(hd.detach().float().reshape(-1))


def hausdorff95_if_available(yp: torch.Tensor, y: torch.Tensor) -> Optional[float]:
    """If ``yp`` is logits ``[B, C, ...]``, returns mean HD95; else ``None``."""
    if compute_hausdorff_distance is None or yp.dim() != y.dim() + 1:
        return None
    nc = int(yp.shape[1])
    t = mean_hd95(yp, y, nc, softmax=True)
    v = float(t.detach().cpu())
    return v if v == v else None  # filter nan


def sensitivity_specificity(
    y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5, binary: bool = True
) -> Tuple[float, float]:
    with torch.no_grad():
        if binary:
            p = (y_pred.sigmoid() if y_pred.min() < 0 else y_pred) > threshold
            t = y_true > 0.5
        else:
            p = y_pred.argmax(1) > 0
            t = y_true > 0
        tp = (p & t).float().sum()
        fn = ((~p) & t).float().sum()
        fp = (p & (~t)).float().sum()
        tn = ((~p) & (~t)).float().sum()
        se = (tp + 1e-6) / (tp + fn + 1e-6)
        sp = (tn + 1e-6) / (tn + fp + 1e-6)
    return float(se), float(sp)


__all__ = [
    "binary_dice_bool",
    "brats_wt_tc_et_dice",
    "dice_iou_foreground_per_class",
    "dice_per_class",
    "iou_per_class",
    "mean_hd95",
    "hausdorff95_if_available",
    "sensitivity_specificity",
]
