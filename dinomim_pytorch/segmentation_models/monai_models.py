"""
MONAI 3D segmentation backbones: UNet, UNETR, SegResNet, SwinUNETR.
``try_build_*_monai`` return ``None`` if MONAI is unavailable or construction fails.
"""

from __future__ import annotations

import inspect
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch.nn as nn

_LOG = logging.getLogger(__name__)


def _img_size(c: Dict[str, Any]) -> Union[Tuple[int, int, int], List[int]]:
    s = c.get("img_size") or c.get("spatial_size") or (96, 96, 96)
    if isinstance(s, (int, float)):
        v = int(s)
        return (v, v, v)
    if isinstance(s, (list, tuple)) and len(s) == 3:
        return (int(s[0]), int(s[1]), int(s[2]))
    return (96, 96, 96)


def _dims(c: Dict[str, Any]) -> int:
    return int(c.get("spatial_dims", 3))


def build_unet(c: Dict[str, Any]) -> nn.Module:
    from monai.networks.nets import UNet

    d = _dims(c)
    sc = int(c.get("feature_size", 32))

    merged: Dict[str, Any] = {}
    un = c.get("unet")
    if isinstance(un, dict):
        for key in (
            "channels",
            "strides",
            "num_res_units",
            "spatial_dims",
            "kernel_size",
            "up_kernel_size",
            "dropout",
            "bias",
            "act",
            "norm",
            "adn_ordering",
        ):
            if un.get(key) is not None:
                merged[key] = un[key]

    for key in (
        "channels",
        "strides",
        "num_res_units",
        "spatial_dims",
        "kernel_size",
        "up_kernel_size",
        "dropout",
        "bias",
        "act",
        "norm",
        "adn_ordering",
    ):
        if c.get(key) is not None:
            merged[key] = c[key]

    spatial_dims = int(merged.get("spatial_dims", d))
    num_res_units = int(merged.get("num_res_units", c.get("num_res_units", 2)))

    ch = merged.get("channels")
    if ch is None:
        ch = (sc, sc * 2, sc * 4, sc * 8, sc * 8)
    else:
        ch = tuple(int(x) for x in ch)

    nl = max(0, len(ch) - 1)
    st = merged.get("strides")
    if st is None:
        st = (2,) * nl
    else:
        st = tuple(int(x) for x in st)

    kw: Dict[str, Any] = {
        "spatial_dims": spatial_dims,
        "in_channels": int(c.get("in_channels", 1)),
        "out_channels": int(c.get("out_channels", 2)),
        "channels": ch,
        "strides": st,
        "num_res_units": num_res_units,
    }
    for opt in ("kernel_size", "up_kernel_size", "dropout", "bias", "act", "norm", "adn_ordering"):
        if opt in merged:
            kw[opt] = merged[opt]

    sig = inspect.signature(UNet.__init__).parameters
    valid = {k: v for k, v in kw.items() if k in sig}
    return UNet(**valid)


def build_unetr(c: Dict[str, Any]) -> nn.Module:
    d = int(c.get("spatial_dims", 3))
    if d != 3:
        raise NotImplementedError(
            f"UNETR in this project is for 3D only; got spatial_dims={d}. Use 2D slice pipeline separately."
        )
    from monai.networks.nets import UNETR

    img = _img_size(c)
    kw: Dict[str, Any] = {
        "in_channels": int(c.get("in_channels", 1)),
        "out_channels": int(c.get("out_channels", 2)),
        "feature_size": int(c.get("feature_size", 16)),
    }
    vit = c.get("vit")
    if isinstance(vit, dict):
        if vit.get("embed_dim") is not None:
            kw["hidden_size"] = int(vit["embed_dim"])
        if vit.get("num_heads") is not None:
            kw["num_heads"] = int(vit["num_heads"])
        mr = vit.get("mlp_ratio")
        if mr is not None:
            hs = kw.get("hidden_size")
            if hs is None:
                hs = int(c.get("hidden_size", 768))
                kw["hidden_size"] = hs
            kw["mlp_dim"] = int(round(float(hs) * float(mr)))
        dep = vit.get("depth")
        if dep is not None and int(dep) != 12:
            warnings.warn(
                f"model.vit.depth={dep}: MONAI UNETR uses a fixed 12-layer ViT internally; "
                "depth is not configurable. Match MAE_v3 ``monai_aligned`` with depth: 12, or ignore this field.",
                UserWarning,
                stacklevel=2,
            )
    for k in (
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
        if c.get(k) is not None:
            kw[k] = c[k]
    p = inspect.signature(UNETR.__init__).parameters
    if "img_size" in p:
        kw["img_size"] = img
    elif "image_size" in p:
        kw["image_size"] = img
    valid = {k: v for k, v in kw.items() if k in p}
    return UNETR(**valid)


def build_segresnet(c: Dict[str, Any]) -> nn.Module:
    from monai.networks.nets import SegResNet

    d = _dims(c)
    return SegResNet(
        spatial_dims=d,
        in_channels=int(c.get("in_channels", 1)),
        out_channels=int(c.get("out_channels", 2)),
        init_filters=int(c.get("feature_size", 32)),
    )


def build_swinunetr(c: Dict[str, Any]) -> nn.Module:
    from monai.networks.nets import SwinUNETR

    img = _img_size(c)
    base: Dict[str, Any] = {
        "in_channels": int(c.get("in_channels", 1)),
        "out_channels": int(c.get("out_channels", 2)),
        "feature_size": int(c.get("feature_size", 32)),
        "use_checkpoint": bool(c.get("use_checkpoint", False)),
    }
    sw = c.get("swin")
    if isinstance(sw, dict):
        if sw.get("embed_dim") is not None:
            base["feature_size"] = int(sw["embed_dim"])
        if sw.get("depths") is not None:
            base["depths"] = tuple(int(x) for x in sw["depths"])
        if sw.get("num_heads") is not None:
            base["num_heads"] = tuple(int(x) for x in sw["num_heads"])
        if sw.get("window_size") is not None:
            ws = sw["window_size"]
            base["window_size"] = (
                int(ws) if isinstance(ws, (int, float)) else tuple(int(x) for x in ws)
            )
        if sw.get("mlp_ratio") is not None:
            base["mlp_ratio"] = float(sw["mlp_ratio"])
        if sw.get("patch_size") is not None:
            base["patch_size"] = int(sw["patch_size"])
    if c.get("feature_size") is not None:
        base["feature_size"] = int(c["feature_size"])
    for k in (
        "patch_size",
        "depths",
        "num_heads",
        "window_size",
        "qkv_bias",
        "mlp_ratio",
        "drop_rate",
        "attn_drop_rate",
        "dropout_path_rate",
        "normalize",
        "norm_name",
        "patch_norm",
        "spatial_dims",
        "downsample",
        "use_v2",
        "use_checkpoint",
    ):
        if c.get(k) is not None:
            v = c[k]
            if k in ("depths", "num_heads") and isinstance(v, (list, tuple)):
                base[k] = tuple(int(x) for x in v)
            else:
                base[k] = v
    sig = inspect.signature(SwinUNETR.__init__).parameters
    if "img_size" in sig:
        base["img_size"] = img
    elif "image_size" in sig:
        base["image_size"] = img
    valid = {k: v for k, v in base.items() if k in sig}
    return SwinUNETR(**valid)


def try_build_unet_monai(c: Dict[str, Any]) -> Optional[nn.Module]:
    try:
        return build_unet(c)
    except Exception as e:  # noqa: BLE001
        _LOG.debug("MONAI UNet build failed: %s", e)
        return None


def try_build_unetr_monai(c: Dict[str, Any]) -> Optional[nn.Module]:
    try:
        return build_unetr(c)
    except Exception as e:  # noqa: BLE001
        _LOG.debug("MONAI UNETR build failed: %s", e)
        return None


def try_build_segresnet_monai(c: Dict[str, Any]) -> Optional[nn.Module]:
    try:
        return build_segresnet(c)
    except Exception as e:  # noqa: BLE001
        _LOG.debug("MONAI SegResNet build failed: %s", e)
        return None


def try_build_swinunetr_monai(c: Dict[str, Any]) -> Optional[nn.Module]:
    try:
        return build_swinunetr(c)
    except Exception as e:  # noqa: BLE001
        _LOG.debug("MONAI SwinUNETR build failed: %s", e)
        return None


__all__ = [
    "build_unet",
    "build_unetr",
    "build_segresnet",
    "build_swinunetr",
    "try_build_unet_monai",
    "try_build_unetr_monai",
    "try_build_segresnet_monai",
    "try_build_swinunetr_monai",
]
