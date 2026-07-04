"""
CXR downstream classification with TIMM backbones and optional DINO encoder init.
"""

from __future__ import annotations

from typing import Any, Dict

import timm
import torch.nn as nn

from dinomim_pytorch.checkpointing import load_dino_weights_into_downstream_model


def _model_dict(config: Any) -> Dict[str, Any]:
    if isinstance(config, dict) and "model" in config:
        return dict(config["model"])
    if isinstance(config, dict):
        return config
    return {}


def build_classification_model(config: Any) -> nn.Module:
    c = _model_dict(config)
    if str(c.get("backbone_source", "timm")) != "timm":
        raise NotImplementedError("Only backbone_source=timm is supported for CXR classification.")
    name = c.get("backbone_name", "resnet50")
    in_ch = int(c.get("in_channels", 1))
    ncls = int(c.get("num_classes", 2))
    pretrained = bool(c.get("pretrained", False))
    model = timm.create_model(
        str(name), pretrained=pretrained, in_chans=in_ch, num_classes=ncls, global_pool="avg"
    )
    if c.get("ssl_init") and c.get("ssl_checkpoint"):
        load_dino_weights_into_downstream_model(
            model,
            c["ssl_checkpoint"],
            load_encoder_only=bool(c.get("load_encoder_only", True)),
            strict_load=bool(c.get("strict_load", False)),
            from_teacher_encoder=bool(c.get("load_from_teacher_encoder", False)),
        )
    return model


def load_dino_encoder_weights(
    model: nn.Module, checkpoint: str, strict: bool = False
) -> Dict[str, Any]:
    return load_dino_weights_into_downstream_model(
        model,
        checkpoint,
        load_encoder_only=True,
        strict_load=not strict,
    )


__all__ = ["build_classification_model", "load_dino_encoder_weights"]
