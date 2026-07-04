"""Local 3D SwinUNETR-style fallback (3D U-Net + attention) when MONAI SwinUNETR is unavailable."""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from dinomim_pytorch.segmentation_models.local_swinunet3d import LocalSwinUnet3D


class LocalSwinUNETR3D(LocalSwinUnet3D):
    """Same capacity as ``LocalSwinUnet3D``; distinct class name for the SwinUNETR registry slot."""


def build_local_swinunetr3d(c: Dict[str, Any]) -> nn.Module:
    return LocalSwinUNETR3D(
        int(c.get("in_channels", 1)),
        int(c.get("out_channels", 2)),
        int(c.get("feature_size", 32)),
    )
