"""Local 3D UNETR++-style model without MONAI (deeper 3D U-Net with dense skip contract). [B,C,D,H,W] -> [B,K,D,H,W]."""

from __future__ import annotations

import warnings
from typing import Any, Dict

import torch
import torch.nn as nn

from dinomim_pytorch.segmentation_models.local_unet3d import LocalUNet3D


class LocalUNETRPP3D(nn.Module):
    """
    When MONAI has no UNETR++ class, this provides a true 3D local fallback:
    a deeper 3D U-Net (UNet++-inspired multi-scale capacity; not the official code).
    """

    def __init__(self, in_ch: int, out_ch: int, f0: int = 32, levels: int = 5) -> None:
        super().__init__()
        self.net = LocalUNet3D(in_ch, out_ch, feature_size=f0, levels=int(levels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_local_unetrpp3d(c: Dict[str, Any]) -> nn.Module:
    """
    Optional ``model.unetrpp`` / ``model.convvit`` — same semantics as BYOL ``build_local_unetrpp3d``.
    """
    in_ch = int(c.get("in_channels", 1))
    out_ch = int(c.get("out_channels", 2))
    f0 = int(c.get("feature_size", 32))
    levels = int(c.get("levels", 5))

    ut = c.get("unetrpp")
    if isinstance(ut, dict):
        if ut.get("levels") is not None:
            levels = int(ut["levels"])
        if ut.get("feature_size") is not None:
            f0 = int(ut["feature_size"])

    ut_has_fs = isinstance(ut, dict) and ut.get("feature_size") is not None
    ut_has_lv = isinstance(ut, dict) and ut.get("levels") is not None

    cv = c.get("convvit")
    if isinstance(cv, dict):
        if cv.get("stem_channels") is not None:
            f0 = int(cv["stem_channels"])
        elif cv.get("embed_dim") is not None and not ut_has_fs:
            f0 = int(cv["embed_dim"])
        dep = cv.get("depth")
        if dep is not None and not ut_has_lv:
            levels = max(2, min(8, int(dep)))
        unused = [k for k in ("num_heads", "mlp_ratio") if cv.get(k) is not None]
        if unused:
            warnings.warn(
                "model.convvit.%s not used by local UNETR++ CNN stub (MONAI-free); "
                "only stem_channels/embed_dim/depth affect width/levels." % ", ".join(unused),
                UserWarning,
                stacklevel=2,
            )

    return LocalUNETRPP3D(in_ch, out_ch, f0=f0, levels=levels)
