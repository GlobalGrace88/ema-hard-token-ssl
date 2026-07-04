"""
Official UNETR++ from vendored ``unetr_plus_plus-main`` (EPA architecture, paper implementation).

Opt-in via ``model.preferred_source: official`` (or ``backbone_source: official_unetrpp`` for SSL).
Default ``local`` / ``auto`` keeps ``local_unetrpp3d`` unchanged.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn

# Paper / trainer defaults (Synapse trainer in unetr_plus_plus-main).
_DEFAULT_DEPTHS = [3, 3, 3, 3]
_DEFAULT_DIMS = [32, 64, 128, 256]

# Recommended ``model.spatial_size`` / ``img_size`` per variant (decoder geometry is fixed in-repo).
OFFICIAL_VARIANT_SPATIAL: Dict[str, Tuple[int, int, int]] = {
    "synapse": (64, 128, 128),
    "tumor": (128, 128, 128),
    "acdc": (16, 160, 160),
    "lung": (32, 192, 192),
}


def resolve_unetrpp_repo_root(explicit: Optional[str] = None) -> Path:
    """Locate ``unetr_plus_plus-main`` via ``UNETR_PP_ROOT`` or ``paths.yaml``."""
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if p.is_dir():
            return p
        raise FileNotFoundError(f"UNETR_PP_ROOT not found: {p}")
    env = os.environ.get("UNETR_PP_ROOT", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p
    try:
        from dinomim_pytorch.paths import unetr_pp_root

        p = unetr_pp_root()
        if p.is_dir():
            return p
    except (FileNotFoundError, ImportError):
        pass
    raise FileNotFoundError(
        "Official UNETR++ not found. Clone unetr_plus_plus-main and set UNETR_PP_ROOT "
        "or unetr_pp_root in paths.yaml (see docs/setup_unetrpp.md)."
    )


def _ensure_unetrpp_import_path(repo_root: Path) -> None:
    root = str(repo_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _spatial_from_cfg(c: Dict[str, Any], variant: str) -> Tuple[int, int, int]:
    sp = c.get("spatial_size") or c.get("img_size") or c.get("volume_size")
    if isinstance(sp, (list, tuple)) and len(sp) == 3:
        return (int(sp[0]), int(sp[1]), int(sp[2]))
    if isinstance(sp, (int, float)):
        v = int(sp)
        return (v, v, v)
    return OFFICIAL_VARIANT_SPATIAL.get(variant, (96, 96, 96))


def _import_unetrpp_class(variant: str, repo_root: Path) -> Type[nn.Module]:
    _ensure_unetrpp_import_path(repo_root)
    v = (variant or "synapse").lower().strip()
    if v == "synapse":
        from unetr_pp.network_architecture.synapse.unetr_pp_synapse import (  # type: ignore
            UNETR_PP,
        )

        return UNETR_PP
    if v == "tumor":
        from unetr_pp.network_architecture.tumor.unetr_pp_tumor import (  # type: ignore
            UNETR_PP,
        )

        return UNETR_PP
    if v == "acdc":
        from unetr_pp.network_architecture.acdc.unetr_pp_acdc import (  # type: ignore  # noqa: E501
            UNETR_PP,
        )

        return UNETR_PP
    if v == "lung":
        from unetr_pp.network_architecture.lung.unetr_pp_lung import (  # type: ignore
            UNETR_PP,
        )

        return UNETR_PP
    raise ValueError(
        f"Unknown model.unetrpp_official_variant={variant!r}. "
        f"Use: {', '.join(sorted(OFFICIAL_VARIANT_SPATIAL))}"
    )


def build_official_unetrpp3d(c: Dict[str, Any]) -> nn.Module:
    """
    Build paper UNETR++ for SSL or downstream segmentation.

    Config keys:
    - ``unetrpp_official_variant`` / ``unetrpp.official_variant``: synapse | tumor | acdc | lung
    - ``unetrpp_repo_root``: override repo path
    - ``spatial_size`` / ``img_size``: must match variant geometry (see OFFICIAL_VARIANT_SPATIAL)
    - ``unetrpp`` sub-dict: feature_size, num_heads, depths, dims, do_ds, hidden_size, ...
    """
    ut = c.get("unetrpp") if isinstance(c.get("unetrpp"), dict) else {}
    variant = str(
        c.get("unetrpp_official_variant")
        or ut.get("official_variant")
        or c.get("official_variant")
        or "synapse"
    ).lower()
    repo = resolve_unetrpp_repo_root(
        str(c.get("unetrpp_repo_root") or ut.get("repo_root") or "") or None
    )
    UNETR_PP = _import_unetrpp_class(variant, repo)

    in_ch = int(c.get("in_channels", 1))
    out_ch = int(c.get("out_channels", 2))
    fsize = int(ut.get("feature_size", c.get("feature_size", 16)))
    num_heads = int(ut.get("num_heads", c.get("num_heads", 4)))
    depths = list(ut.get("depths", c.get("depths", _DEFAULT_DEPTHS)))
    dims = list(ut.get("dims", c.get("dims", _DEFAULT_DIMS)))
    hidden_size = int(ut.get("hidden_size", c.get("hidden_size", 256)))
    do_ds = bool(ut.get("do_ds", c.get("do_ds", False)))
    norm_name = ut.get("norm_name", c.get("norm_name", "instance"))
    dropout_rate = float(ut.get("dropout_rate", c.get("dropout_rate", 0.0)))

    kw: Dict[str, Any] = dict(
        in_channels=in_ch,
        out_channels=out_ch,
        feature_size=fsize,
        num_heads=num_heads,
        depths=depths,
        dims=dims,
        hidden_size=hidden_size,
        do_ds=do_ds,
        norm_name=norm_name,
        dropout_rate=dropout_rate,
    )
    if variant == "synapse":
        kw["img_size"] = list(_spatial_from_cfg(c, variant))

    net = UNETR_PP(**kw)
    net._dino_official_variant = variant  # type: ignore[attr-defined]
    net._dino_official_repo = str(repo)  # type: ignore[attr-defined]
    net._dino_in_channels = in_ch  # type: ignore[attr-defined]
    return net


class OfficialUNETRPPSSLEncoder(nn.Module):
    """
    DINO SSL backbone: global-pool EPA encoder features (``unetr_pp_encoder``), not 1-ch seg logits.

    Without this, ``Pooled3DFullModelEncoder`` on ``out_channels=1`` yields a **1-D** vector per view,
    and DINO loss stalls at ``log(out_dim)`` (e.g. ``log(4096) ≈ 8.32``).
    """

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net
        self.embed_dim = int(getattr(net, "hidden_size", 256))
        self.in_channels = int(getattr(net, "_dino_in_channels", 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.net, "unetr_pp_encoder"):
            raise AttributeError("Official UNETR++ model missing unetr_pp_encoder")
        _x_out, hidden_states = self.net.unetr_pp_encoder(x)
        if not hidden_states:
            raise RuntimeError("unetr_pp_encoder returned no hidden_states")
        h = hidden_states[-1]
        if h.dim() == 3:
            return h.mean(dim=1)
        if h.dim() == 5:
            return h.mean(dim=(2, 3, 4))
        if h.dim() == 2:
            return h
        raise RuntimeError(f"Unexpected encoder feature shape {tuple(h.shape)}")


def build_official_unetrpp_ssl_encoder(c: Dict[str, Any]) -> nn.Module:
    """Official UNETR++ for DINO pretrain (encoder features only)."""
    return OfficialUNETRPPSSLEncoder(build_official_unetrpp3d(c))


def recommended_spatial_size(variant: str) -> Tuple[int, int, int]:
    return OFFICIAL_VARIANT_SPATIAL.get((variant or "synapse").lower(), (96, 96, 96))


__all__ = [
    "build_official_unetrpp3d",
    "build_official_unetrpp_ssl_encoder",
    "OfficialUNETRPPSSLEncoder",
    "resolve_unetrpp_repo_root",
    "OFFICIAL_VARIANT_SPATIAL",
    "recommended_spatial_size",
]
