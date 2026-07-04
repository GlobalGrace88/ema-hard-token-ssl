# Backward compatibility: local 3D Swin-UNet.

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from dinomim_pytorch.segmentation_models.local_swinunet3d import (
    LocalSwinUnet3D,
    build_local_swinunet3d,
)

SwinUnet3D = LocalSwinUnet3D


def build_swin_unet(c: Dict[str, Any]) -> nn.Module:
    return build_local_swinunet3d(c)
