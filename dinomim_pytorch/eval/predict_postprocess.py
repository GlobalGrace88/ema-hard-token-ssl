"""Apply MAE-style CC post-processing to model logits (argmax → filter → label map)."""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import torch

from dinomim_pytorch.eval.seg_postprocess import apply_seg_postprocess, postprocess_enabled


def logits_to_label_map(
    logits: torch.Tensor,
    num_classes: int,
    postprocess_cfg: Optional[Mapping[str, Any]] = None,
) -> torch.Tensor:
    """
    Argmax logits ``[B,C,*spatial]`` → integer labels ``[B,*spatial]``.

    If ``postprocess_cfg.enabled`` is true, run connected-component filtering
    (MAE ``unetr_brats.yaml`` behavior).
    """
    if logits.dim() < 4 or int(logits.shape[1]) < 2:
        raise ValueError(f"Expected logits [B,C,D,H,W], got {tuple(logits.shape)}")
    pred = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
    if not postprocess_enabled(postprocess_cfg):
        return torch.from_numpy(pred).to(device=logits.device, dtype=torch.long)
    pp = dict(postprocess_cfg or {})
    out = np.empty_like(pred)
    for b in range(pred.shape[0]):
        out[b] = apply_seg_postprocess(pred[b], num_classes=num_classes, cfg=pp)
    return torch.from_numpy(out).to(device=logits.device, dtype=torch.long)


__all__ = ["logits_to_label_map"]
