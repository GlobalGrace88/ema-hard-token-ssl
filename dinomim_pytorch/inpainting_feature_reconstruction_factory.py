"""Factory dispatch for inpainting + feature-reconstruction SSL (UNETR++ or Swin UNETR)."""

from __future__ import annotations

from typing import Any, Dict, Union

import torch.nn as nn

from dinomim_pytorch.swin_unetr_feature_reconstruction import (
    SwinUNETRInpaintingFeatureReconstruction,
    build_swin_unetr_inpainting_feature_reconstruction,
)
from dinomim_pytorch.unetrpp_feature_reconstruction import (
    UNETRPPInpaintingFeatureReconstruction,
    build_unetrpp_inpainting_feature_reconstruction,
)

InpaintingFeatureReconModel = Union[
    UNETRPPInpaintingFeatureReconstruction,
    SwinUNETRInpaintingFeatureReconstruction,
]


def resolve_ssl_architecture(cfg: Dict[str, Any]) -> str:
    mcfg = dict((cfg or {}).get("model") or {})
    for key in ("architecture", "name", "backbone_name", "ssl_method"):
        raw = str(mcfg.get(key, "") or "").strip().lower().replace("-", "_")
        compact = raw.replace("_", "")
        if compact in ("swinunetr", "swinunet") or raw in ("swin_unetr", "swin_unet_r"):
            return "swinunetr"
    return "unetrpp"


def build_inpainting_feature_reconstruction(cfg: Dict[str, Any]) -> InpaintingFeatureReconModel:
    arch = resolve_ssl_architecture(cfg)
    if arch == "swinunetr":
        return build_swin_unetr_inpainting_feature_reconstruction(cfg)
    return build_unetrpp_inpainting_feature_reconstruction(cfg)


def ssl_pretrain_scheme(cfg: Dict[str, Any]) -> str:
    arch = resolve_ssl_architecture(cfg)
    if arch == "swinunetr":
        return "swin_unetr_inpainting_feature_reconstruction"
    return "unetrpp_inpainting_feature_reconstruction"


__all__ = [
    "InpaintingFeatureReconModel",
    "build_inpainting_feature_reconstruction",
    "resolve_ssl_architecture",
    "ssl_pretrain_scheme",
]
