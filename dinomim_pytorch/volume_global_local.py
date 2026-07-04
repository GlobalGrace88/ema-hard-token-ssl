"""3D global vs local voxel crops for multi-scale DINO (nested local inside global FOV)."""

from __future__ import annotations

import random
from typing import Tuple

import torch
import torch.nn.functional as F

__all__ = ["resize_volume_to_spatial", "sample_global_local_crops_3d"]


def resize_volume_to_spatial(x: torch.Tensor, spatial: Tuple[int, int, int]) -> torch.Tensor:
    """Resize [B,C,D,H,W] to ``spatial`` (D,H,W) with trilinear interpolation."""
    if x.dim() != 5:
        raise ValueError(f"Expected 5D [B,C,D,H,W], got {tuple(x.shape)}")
    d, h, w = spatial
    if tuple(x.shape[-3:]) == (d, h, w):
        return x
    return F.interpolate(x, size=(d, h, w), mode="trilinear", align_corners=False)


def sample_global_local_crops_3d(
    x: torch.Tensor,
    global_size: Tuple[int, int, int],
    local_size: Tuple[int, int, int],
    *,
    nested_local_in_global: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Crop one global window and one local window from each batch element.

    * ``nested_local_in_global`` (default): local crop lies fully inside the global crop
      (same anatomy at two scales).
    * Otherwise: independent uniform random crops (each must fit in ``x``).
    """
    if x.dim() != 5:
        raise ValueError(f"Expected [B,C,D,H,W], got {tuple(x.shape)}")
    _, _, D, H, W = x.shape
    dg, dh, dw = global_size
    dl, dhl, dwl = local_size
    if dg > D or dh > H or dw > W:
        raise ValueError(
            f"global_crop_size {global_size} does not fit volume spatial {D,H,W}"
        )
    if dl > D or dhl > H or dwl > W:
        raise ValueError(
            f"local_crop_size {local_size} does not fit volume spatial {D,H,W}"
        )
    if nested_local_in_global:
        if dl > dg or dhl > dh or dwl > dw:
            raise ValueError(
                f"nested local {local_size} must be <= global {global_size} on each axis"
            )
    z0 = random.randint(0, D - dg)
    y0 = random.randint(0, H - dh)
    x0 = random.randint(0, W - dw)
    xg = x[:, :, z0 : z0 + dg, y0 : y0 + dh, x0 : x0 + dw]
    if nested_local_in_global:
        rz = random.randint(0, dg - dl)
        ry = random.randint(0, dh - dhl)
        rx = random.randint(0, dw - dwl)
        z1 = z0 + rz
        y1 = y0 + ry
        x1 = x0 + rx
    else:
        z1 = random.randint(0, D - dl)
        y1 = random.randint(0, H - dhl)
        x1 = random.randint(0, W - dwl)
    xl = x[:, :, z1 : z1 + dl, y1 : y1 + dhl, x1 : x1 + dwl]
    return xg, xl
