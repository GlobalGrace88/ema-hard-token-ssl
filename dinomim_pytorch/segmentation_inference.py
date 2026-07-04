"""
3D validation / test inference with ``monai.inferers.sliding_window_inference``.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn

try:
    from monai.inferers import sliding_window_inference
except Exception:  # noqa: BLE001
    sliding_window_inference = None  # type: ignore[assignment]


def sliding_window_predict(
    model: nn.Module,
    inputs: torch.Tensor,
    roi_size: Union[Tuple[int, int, int], list],
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    mode: str = "gaussian",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    ``inputs`` shape ``[B, C, D, H, W]`` (B often 1 for full volume).
    """
    if sliding_window_inference is None:
        raise ImportError("MONAI is required for sliding window inference")
    if device is None:
        device = next(model.parameters()).device
    with torch.set_grad_enabled(model.training):
        return sliding_window_inference(  # type: ignore[no-any-return]
            inputs, roi_size, sw_batch_size, model, overlap, mode=mode, device=device, sw_device=device
        )


def predict_from_config(
    model: nn.Module,
    inputs: torch.Tensor,
    val_cfg: dict,
) -> torch.Tensor:
    c = (val_cfg or {}).get("validation", val_cfg) or val_cfg
    if not c.get("sliding_window_inference", True):
        return model(inputs)
    return sliding_window_predict(
        model,
        inputs,
        roi_size=tuple(c.get("roi_size", (96, 96, 96))),  # type: ignore[arg-type]
        sw_batch_size=int(c.get("sw_batch_size", 4)),
        overlap=float(c.get("overlap", 0.5)),
    )
