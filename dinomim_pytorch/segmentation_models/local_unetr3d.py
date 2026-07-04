"""Local 3D encoder–decoder (CNN; not ViT). Fallback when MONAI UNETR unavailable. [B,C,D,H,W] -> [B,K,D,H,W]."""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from dinomim_pytorch.segmentation_models.local_unet3d import LocalUNet3D


def build_local_unetr3d(c: Dict[str, Any]) -> nn.Module:
    """UNETR-shaped contract: same I/O as MONAI UNETR; architecture is a strong 3D U-Net fallback."""
    return LocalUNet3D(
        int(c.get("in_channels", 1)),
        int(c.get("out_channels", 2)),
        feature_size=int(c.get("feature_size", 16)),
        levels=int(c.get("levels", 5)),
    )


class LocalUNETR3D(LocalUNet3D):
    """Alias for registry class name."""
