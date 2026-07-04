"""Validation Dice via global TP/FP/FN (MAE_BYOL ``train_seg`` parity)."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch


def val_confusion_update(
    pred: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    tp: torch.Tensor,
    fp: torch.Tensor,
    fn: torch.Tensor,
) -> None:
    lc = labels.long().clamp(0, num_classes - 1)
    for c in range(num_classes):
        p = pred == c
        t = lc == c
        tp[c] = tp[c] + (p & t).sum().double()
        fp[c] = fp[c] + (p & (~t)).sum().double()
        fn[c] = fn[c] + ((~p) & t).sum().double()


def val_dice_from_accumulated(
    tp: torch.Tensor,
    fp: torch.Tensor,
    fn: torch.Tensor,
    num_classes: int,
    *,
    smooth: float = 1e-5,
    class_names: Optional[Sequence[str]] = None,
) -> Tuple[float, dict[str, float], str]:
    """
    Macro Dice = nan-mean over foreground classes with any GT (TP+FN>0).

    Returns ``(macro_dice, per_class_slug_dict, note_suffix)``.
    """
    fg: List[float] = []
    per_class: dict[str, float] = {}
    n_with_gt = 0
    n_fg = num_classes - 1
    names = list(class_names) if class_names is not None else [f"class_{c}" for c in range(num_classes)]
    if len(names) != num_classes:
        names = [f"class_{c}" for c in range(num_classes)]

    for c in range(1, num_classes):
        tpc, fpc, fnc = tp[c].item(), fp[c].item(), fn[c].item()
        slug = names[c].replace(" ", "_")
        key = f"val_dice_c{c:02d}_{slug}"
        if tpc + fnc <= 0:
            if fpc <= 0:
                fg.append(float("nan"))
            else:
                fg.append(float("nan"))
        else:
            n_with_gt += 1
            d = (2 * tpc + smooth) / (2 * tpc + fpc + fnc + smooth)
            fg.append(float(d))
            per_class[key] = float(d)

    macro = float(np.nanmean(np.asarray(fg, dtype=np.float64))) if fg else 0.0
    if not np.isfinite(macro):
        macro = 0.0
    note = f" | val_macro_over={n_with_gt}/{n_fg}_fg"
    return macro, per_class, note


__all__ = ["val_confusion_update", "val_dice_from_accumulated"]
