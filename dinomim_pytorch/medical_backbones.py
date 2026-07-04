"""
2D and 3D SSL backbones: TIMM (2D) and MONAI 3D encoders for volumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

# MAE_BYOL MAE_v3 naming: optional ssl_encoder fills monai_3d_model when omitted.
# convvit_mae → UNETR++ (``Pooled3DFullModelEncoder`` + local UNETR++), matching MAE_v3 conv SSL → UNETR++.
SSL_ENCODER_TO_MONAI: Dict[str, str] = {
    "vit_mae": "unetr",
    "vit_mae3d": "unetr",
    "vit3d": "unetr",
    "convvit_mae": "unetrpp",
    "convvit3d": "unetrpp",
    "cnn_unet": "unet",
    "monai_unet": "unet",
    "unet_trunk": "unet",
    "cnn3d": "unet",
    "swin_mim": "swinunetr",
    "swin_mae": "swinunetr",
    "swin3d": "swinunetr",
}


def _apply_ssl_encoder_monai_default(mcfg: Dict) -> Dict:
    m = dict(mcfg)
    enc = str(m.get("ssl_encoder", "") or "").strip().lower()
    if enc and not str(m.get("monai_3d_model", "") or "").strip():
        tgt = SSL_ENCODER_TO_MONAI.get(enc)
        if tgt:
            m["monai_3d_model"] = tgt
    return m


@dataclass
class BackboneConfig:
    backbone_source: str
    backbone_name: str
    pretrained: bool = False
    in_channels: int = 1
    num_classes: int = 0
    global_pool: str = "avg"
    features_only: bool = False
    is_3d: bool = False
    monai_3d_model: str = "swinunetr"
    spatial_size: Optional[tuple] = None
    feature_size: int = 32


def get_model_config(config: Any) -> Dict:
    if config is None:
        return {}
    if isinstance(config, dict):
        if "model" in config:
            return dict(config["model"])
        if "backbone" in config:
            return dict(config["backbone"])
    return {}


def _as_cfg(d: Dict) -> BackboneConfig:
    return BackboneConfig(
        backbone_source=str(d.get("backbone_source", "timm")).lower(),
        backbone_name=str(d.get("backbone_name", "resnet50")),
        pretrained=bool(d.get("pretrained", False)),
        in_channels=int(d.get("in_channels", 1)),
        num_classes=int(d.get("num_classes", 0)),
        global_pool=str(d.get("global_pool", "avg")),
        features_only=bool(d.get("features_only", False)),
        is_3d=bool(d.get("is_3d", False)),
        monai_3d_model=str(d.get("monai_3d_model", "swinunetr")).lower(),
        spatial_size=d.get("spatial_size") or d.get("volume_size") or d.get("img_size")
        or d.get("image_size"),
        feature_size=int(d.get("feature_size", 32)),
    )


def _spatial_tuple(d: Dict) -> tuple:
    s = d.get("spatial_size") or d.get("volume_size") or d.get("img_size")
    if s is None:
        return (96, 96, 96)
    if isinstance(s, (list, tuple)) and len(s) == 3:
        return (int(s[0]), int(s[1]), int(s[2]))
    v = int(s) if s is not None else 96
    return (v, v, v)


def _ssl_unetr_representation(d: Dict) -> str:
    """``vit``: ViT submodule only (default). ``full``: MONAI UNETR ``forward`` + pool."""
    raw = str(d.get("ssl_unetr_representation", "vit") or "vit").strip().lower()
    if raw in ("vit", "vit_only", "transformer"):
        return "vit"
    if raw in ("full", "unetr_full", "whole", "all"):
        return "full"
    raise ValueError(
        f"model.ssl_unetr_representation={raw!r} invalid; use 'vit' or 'full' "
        f"(aliases: vit_only, transformer / unetr_full, whole, all)."
    )


def get_timm_forward_embedding_layer_name() -> int:
    """Use full forward pass (pooled embedding) for TIMM when num_classes=0."""
    return -1


# Backward alias
get_dino_hidden_layer_for_timm_model = get_timm_forward_embedding_layer_name


def _build_timm(d: Dict) -> nn.Module:
    import timm

    c = _as_cfg(d)
    kw: Dict[str, Any] = {
        "pretrained": c.pretrained,
        "in_chans": c.in_channels,
        "num_classes": c.num_classes,
        "global_pool": c.global_pool,
    }
    if c.features_only:
        kw["features_only"] = True
    return timm.create_model(c.backbone_name, **kw)


def _pool_to_vec(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 5:
        return t.mean(dim=(2, 3, 4))
    if t.dim() == 4:
        return t.mean(dim=(2, 3))
    if t.dim() == 3:
        return t.mean(dim=1)
    return t


def _guess_first_conv_in_from_patch_embedding(mod: nn.Module) -> Optional[int]:
    """MONAI ViT stacks often tuck input C in PatchEmbedding/Conv*; attrs may omit ``in_channels``."""
    peg = getattr(mod, "patch_embedding", None) or getattr(mod, "patch_embeddings", None)
    if peg is None:
        return None
    for sm in peg.modules():
        if isinstance(sm, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            try:
                return int(sm.weight.shape[1])
            except Exception:
                continue
    return None


def _expose_in_channels(wrapper: nn.Module, child: nn.Module) -> None:
    """SSL wrappers must set ``in_channels`` so DINO can build probe tensors."""
    for k in ("in_channels", "in_chans"):
        if hasattr(child, k):
            try:
                v = getattr(child, k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    wrapper.in_channels = int(v)
                    return
            except (TypeError, ValueError):
                pass
    for sub in (
        child,
        getattr(child, "unetr", None),
        getattr(child, "vit", None),
        getattr(child, "encoder", None),
    ):
        if sub is None:
            continue
        ic = _guess_first_conv_in_from_patch_embedding(sub)
        if ic is not None:
            wrapper.in_channels = ic
            return
    wrapper.in_channels = 1


class MonaiUNetEncoder3D(nn.Module):
    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet
        _expose_in_channels(self, unet)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y: torch.Tensor = self.unet(x)
        return _pool_to_vec(y)


def _nested_last_tensor(z: Any) -> torch.Tensor:
    """MONAI ViTUNETR ViT encoder may return tensors or nested list/tuple hierarchies."""
    while isinstance(z, (list, tuple)):
        if not z:
            raise RuntimeError("UNETR encoder returned an empty seq.")
        z = z[-1]
    if z is None or not isinstance(z, torch.Tensor):
        raise RuntimeError(f"UNETR encoder expected a Tensor at leaf, got {type(z)!r}")
    return z


class MonaiUNETREncoder3D(nn.Module):
    def __init__(self, unetr: Any):
        super().__init__()
        self.m = unetr
        _expose_in_channels(self, unetr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.m
        enc = getattr(u, "unetr", None) or getattr(u, "vit", None)
        if enc is not None:
            z = enc(x)  # type: ignore[operator]
        else:
            z = u(x)  # type: ignore
        z = _nested_last_tensor(z)
        if z.dim() == 2:
            return z
        return _pool_to_vec(z)


class MonaiUNETRFullEncoder3D(nn.Module):
    """
    UNETR full ``forward`` for SSL: CNN encoder + ViT + decoder head, then spatial pool to a vector.

    Enable via ``model.ssl_unetr_representation: full`` (default ``vit`` = ViT branch only).
    """

    def __init__(self, unetr: Any):
        super().__init__()
        self.m = unetr
        _expose_in_channels(self, unetr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y: torch.Tensor = self.m(x)
        return _pool_to_vec(y)


class MonaiSwinUNETREncoder3D(nn.Module):
    def __init__(self, m: Any):
        super().__init__()
        self.swin = getattr(m, "swinViT", m)
        _expose_in_channels(self, m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.swin(x)  # type: ignore[operator]
        if isinstance(y, (list, tuple)):
            y = y[-1]
        return _pool_to_vec(y)


class Pooled3DFullModelEncoder(nn.Module):
    """
    Run a full 3D segmentation U-Net / UNETR++ / Swin-UNet / nnFormer forward pass, then
    global-average the logits for DINO. Matches ``MonaiUNetEncoder3D``; used for **local** nets
    where there is no separate ``encoder`` API.
    """

    def __init__(self, full_model: nn.Module):
        super().__init__()
        self.net = full_model
        _expose_in_channels(self, full_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if isinstance(y, (list, tuple)):
            y = y[0]
        if not isinstance(y, torch.Tensor):
            raise RuntimeError(f"Pooled3DFullModelEncoder expected Tensor, got {type(y)!r}")
        return _pool_to_vec(y)


def _unetrpp_use_official(d: Dict) -> bool:
    """True when config requests paper UNETR++ from ``unetr_plus_plus-main``."""
    pref = str(d.get("preferred_source", "local")).lower().replace("-", "")
    src = str(d.get("backbone_source", "")).lower().replace("-", "")
    if pref in ("official", "unetrpp", "unetr_plus_plus", "unetrplusplus", "paper"):
        return True
    if src in ("official_unetrpp", "official", "unetr_plus_plus", "unetrplusplus"):
        return True
    return bool(d.get("use_official_unetrpp", False))


def _kind_from_3d_name(name: str) -> str:
    """Map ``backbone_name`` / ``monai_3d_model`` to a discrete encoder family (order matters)."""
    s = (name or "").lower().replace("-", "_")
    s_compact = s.replace("_", "")
    if "unetrpp" in s_compact or "unetrplusplus" in s_compact:
        return "unetrpp"
    if "swinunetr" in s_compact:
        return "swinunetr"
    if "swinunet" in s_compact:  # after swinunetr
        return "swinunet"
    if "nnformer" in s_compact:
        return "nnformer"
    if s_compact.startswith("unetr") or "unet_r" in s:
        return "unetr"
    if "unet" in s:
        return "unet"
    return s


def _build_monai_3d(d: Dict) -> nn.Module:
    from dinomim_pytorch.segmentation_models.monai_models import build_swinunetr, build_unet, build_unetr

    c = _as_cfg(d)
    img = _spatial_tuple(d)
    c_in, f = c.in_channels, c.feature_size
    kind_raw = d.get("monai_3d_model") or d.get("backbone_name") or c.monai_3d_model
    kind = _kind_from_3d_name(str(kind_raw)).lower()

    if kind in ("unet", "u-net", "u_net", "3dunet", "3d_unet", "unet3d", "u-net-3d", "u_net_3d"):
        unet_cfg: Dict[str, Any] = {
            "spatial_dims": 3,
            "in_channels": c_in,
            "out_channels": 1,
            "feature_size": f,
        }
        for key in (
            "unet",
            "channels",
            "strides",
            "num_res_units",
            "kernel_size",
            "up_kernel_size",
            "dropout",
            "bias",
            "act",
            "norm",
            "adn_ordering",
        ):
            if d.get(key) is not None:
                unet_cfg[key] = d[key]
        u = build_unet(unet_cfg)
        return MonaiUNetEncoder3D(u)
    if kind in ("unetr", "unet_r"):
        unetr_cfg: Dict[str, Any] = {
            "spatial_dims": 3,
            "in_channels": c_in,
            "out_channels": 1,
            "feature_size": f,
            "img_size": img,
        }
        for key in (
            "vit",
            "hidden_size",
            "mlp_dim",
            "num_heads",
            "dropout_rate",
            "qkv_bias",
            "proj_type",
            "norm_name",
            "conv_block",
            "res_block",
        ):
            if d.get(key) is not None:
                unetr_cfg[key] = d[key]
        u = build_unetr(unetr_cfg)
        if _ssl_unetr_representation(d) == "full":
            return MonaiUNETRFullEncoder3D(u)
        return MonaiUNETREncoder3D(u)
    if kind in ("swinunetr", "swin_unetr", "swin_unet_r"):
        sw_cfg: Dict[str, Any] = {
            "spatial_dims": 3,
            "in_channels": c_in,
            "out_channels": 1,
            "feature_size": f,
            "img_size": img,
            "use_checkpoint": bool(d.get("use_checkpoint", False)),
        }
        for key in (
            "swin",
            "patch_size",
            "depths",
            "num_heads",
            "window_size",
            "mlp_ratio",
            "qkv_bias",
            "drop_rate",
            "attn_drop_rate",
            "dropout_path_rate",
            "normalize",
            "norm_name",
            "patch_norm",
            "downsample",
            "use_v2",
            "use_checkpoint",
        ):
            if d.get(key) is not None:
                sw_cfg[key] = d[key]
        u = build_swinunetr(sw_cfg)
        return MonaiSwinUNETREncoder3D(u)
    if kind == "unetrpp":
        ut_cfg: Dict[str, Any] = {
            "in_channels": c_in,
            "out_channels": 1,
            "feature_size": f,
            "spatial_size": img,
        }
        if isinstance(d.get("unetrpp"), dict):
            ut_cfg["unetrpp"] = dict(d["unetrpp"])
        if isinstance(d.get("convvit"), dict):
            ut_cfg["convvit"] = dict(d["convvit"])
        for key in (
            "preferred_source",
            "unetrpp_official_variant",
            "unetrpp_repo_root",
            "official_variant",
        ):
            if d.get(key) is not None:
                ut_cfg[key] = d[key]
        if _unetrpp_use_official(d):
            from dinomim_pytorch.segmentation_models.official_unetrpp3d import (
                build_official_unetrpp_ssl_encoder,
            )

            return build_official_unetrpp_ssl_encoder(ut_cfg)
        else:
            from dinomim_pytorch.segmentation_models.local_unetrpp3d import (
                build_local_unetrpp3d,
            )

            full = build_local_unetrpp3d(ut_cfg)
            return Pooled3DFullModelEncoder(full)
    if kind in ("nnformer", "nn_former"):
        from dinomim_pytorch.segmentation_models.local_nnformer3d import build_local_nnformer3d

        full = build_local_nnformer3d(
            {
                "in_channels": c_in,
                "out_channels": 1,
                "feature_size": f,
            }
        )
        return Pooled3DFullModelEncoder(full)
    if kind in ("swinunet", "swin_unet"):
        from dinomim_pytorch.segmentation_models.local_swinunet3d import build_local_swinunet3d

        full = build_local_swinunet3d(
            {
                "in_channels": c_in,
                "out_channels": 1,
                "feature_size": f,
            }
        )
        return Pooled3DFullModelEncoder(full)
    raise NotImplementedError(
        f"Unknown 3D encoder kind {kind!r} (from {kind_raw!r}). "
        f"Use: unet, unetr, swinunetr, unetrpp, nnformer, swinunet."
    )


def build_ssl_backbone(config: Any) -> nn.Module:
    """
    Volume SSL: ``model.ssl_encoder`` may be set to MAE_v3-style ``vit_mae`` | ``convvit_mae`` | ``swin_mim``
    to imply ``monai_3d_model`` when that field is omitted (ViT→UNETR, ConViT→UNETR++, Swin→SwinUNETR).
    Pure MONAI U-Net trunk: ``monai_unet`` / ``cnn_unet`` / ``cnn3d``.

    MONAI UNETR: ``model.ssl_unetr_representation`` — ``vit`` (default, ViT branch only) or ``full``
    (full UNETR forward + pool).
    """
    mcfg = get_model_config(config) if not (
        isinstance(config, dict) and "backbone_name" in config and "is_3d" in config
    ) else config
    if not mcfg and isinstance(config, dict):
        mcfg = config
    mcfg = _apply_ssl_encoder_monai_default(dict(mcfg))
    c = _as_cfg(mcfg)
    if c.is_3d:
        return _build_monai_3d(mcfg)
    if c.backbone_source == "timm":
        return _build_timm(mcfg)
    raise NotImplementedError(
        f"backbone_source={c.backbone_source!r} is_3d={c.is_3d} not supported in build_ssl_backbone"
    )


__all__ = [
    "BackboneConfig",
    "build_ssl_backbone",
    "SSL_ENCODER_TO_MONAI",
    "get_timm_forward_embedding_layer_name",
    "get_dino_hidden_layer_for_timm_model",
    "get_model_config",
    "MonaiUNetEncoder3D",
    "MonaiUNETREncoder3D",
    "MonaiUNETRFullEncoder3D",
    "MonaiSwinUNETREncoder3D",
    "Pooled3DFullModelEncoder",
    "_kind_from_3d_name",
]
