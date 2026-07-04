"""
``build_3d_segmentation_model`` — MONAI first (when requested), local 3D fallback, no silent architecture swap.
``build_segmentation_model`` — backward-compatible wrapper around ``build_3d_segmentation_model``.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Optional, Tuple

import torch.nn as nn

from dinomim_pytorch.checkpointing import load_dino_weights_into_downstream_model
from dinomim_pytorch.segmentation_models.model_source_report import (
    append_model_source_report,
    log_monai_unavailable,
)

_LOG = logging.getLogger(__name__)

# Registry: monai key is for documentation; resolution uses try_build_* and local builders.
SEGMENTATION_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "unet3d": {
        "monai": "UNet",
        "local": "LocalUNet3D",
        "prefer": "monai",
    },
    "unetr": {
        "monai": "UNETR",
        "local": "LocalUNETR3D",
        "prefer": "monai",
    },
    "unetrpp": {
        "monai": None,
        "local": "LocalUNETRPP3D",
        "official": "UNETR_PP",
        "prefer": "local",
    },
    "segresnet": {
        "monai": "SegResNet",
        "local": "LocalSegResNet3D",
        "prefer": "monai",
    },
    "swinunetr": {
        "monai": "SwinUNETR",
        "local": "LocalSwinUNETR3D",
        "prefer": "monai",
    },
    "swinunet3d": {
        "monai": None,
        "local": "LocalSwinUnet3D",
        "prefer": "local",
    },
    "nnformer": {
        "monai": None,
        "local": "LocalNNFormer3D",
        "prefer": "local",
    },
}

# Kept for imports; also high-level source map (unet3d in code → unet key here).
MODEL_REGISTRY: Dict[str, str] = {
    "unet3d": "monai",
    "unetr": "monai",
    "segresnet": "monai",
    "swinunetr": "monai",
    "unetrpp": "local",
    "swinunet3d": "local_or_monai_if_available",
    "nnformer": "local",
}


def _get_model_cfg(config: Any) -> Dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    m = dict(config.get("model") or {})
    sm = config.get("segmentation_model")
    if isinstance(sm, dict):
        merged = {**sm, **m}
        return merged
    return m


def get_merged_model_config(config: Any) -> Dict[str, Any]:
    """``model`` merged with ``segmentation_model`` (``model`` wins). Same basis as the segmentation factory."""
    return _get_model_cfg(config)


def _task(config: Dict[str, Any]) -> Optional[str]:
    t = (config.get("experiment") or {}).get("task")
    if t:
        return str(t).lower()
    t2 = config.get("task")
    if t2:
        return str(t2).lower()
    return None


def normalize_architecture(name: str) -> str:
    n = str(name or "unet3d").lower().strip().replace("_", "").replace("-", "")
    alias = {
        "unet": "unet3d",
        "3dunet": "unet3d",
        "u3dunet": "unet3d",
        "unet3d": "unet3d",
        "unetr": "unetr",
        "unet_r": "unetr",
        "unetrpp": "unetrpp",
        "unetrplusplus": "unetrpp",
        "segresnet": "segresnet",
        "seg_resnet": "segresnet",
        "swinunetr": "swinunetr",
        "swinunet": "swinunet3d",
        "swinunet3d": "swinunet3d",
        "swin3dunet": "swinunet3d",
        "nnformer": "nnformer",
        "nnformers": "nnformer",
    }
    return alias.get(n, n)


def _validate_3d_segmentation_config(config: Dict[str, Any]) -> None:
    m = _get_model_cfg(config)
    if int(m.get("spatial_dims", 3)) != 3:
        raise ValueError("Only 3D segmentation is supported. Set model.spatial_dims=3.")
    if config.get("_skip_segmentation_data_validation"):
        return
    t = _task(config)
    if t == "segmentation":
        d = (config.get("data") or {})
        it = d.get("input_type", "volume_3d")
        if it != "volume_3d":
            raise ValueError("2D segmentation is disabled. Use data.input_type=volume_3d.")


def _try_monai(arch: str, c: Dict[str, Any]) -> Optional[nn.Module]:
    from dinomim_pytorch.segmentation_models import monai_models as mm

    if arch == "unet3d":
        return mm.try_build_unet_monai(c)
    if arch == "unetr":
        return mm.try_build_unetr_monai(c)
    if arch == "segresnet":
        return mm.try_build_segresnet_monai(c)
    if arch == "swinunetr":
        return mm.try_build_swinunetr_monai(c)
    if arch in ("unetrpp", "swinunet3d", "nnformer"):
        return None
    return None


def _build_local(arch: str, c: Dict[str, Any]) -> nn.Module:
    if arch == "unet3d":
        from dinomim_pytorch.segmentation_models.local_unet3d import build_local_unet3d

        return build_local_unet3d(c)
    if arch == "unetr":
        from dinomim_pytorch.segmentation_models.local_unetr3d import build_local_unetr3d

        return build_local_unetr3d(c)
    if arch == "unetrpp":
        from dinomim_pytorch.segmentation_models.local_unetrpp3d import build_local_unetrpp3d

        return build_local_unetrpp3d(c)
    if arch == "segresnet":
        from dinomim_pytorch.segmentation_models.local_segresnet3d import build_local_segresnet3d

        return build_local_segresnet3d(c)
    if arch == "swinunetr":
        from dinomim_pytorch.segmentation_models.local_swinunetr3d import build_local_swinunetr3d

        return build_local_swinunetr3d(c)
    if arch == "swinunet3d":
        from dinomim_pytorch.segmentation_models.local_swinunet3d import build_local_swinunet3d

        return build_local_swinunet3d(c)
    if arch == "nnformer":
        from dinomim_pytorch.segmentation_models.local_nnformer3d import build_local_nnformer3d

        return build_local_nnformer3d(c)
    raise NotImplementedError(
        f"Requested 3D segmentation architecture {arch!r} is unavailable in both MONAI and local implementations."
    )


def _resolve_model(
    arch: str,
    c: Dict[str, Any],
    preferred: str,
) -> Tuple[nn.Module, str]:
    """
    Returns (module, source) where source is 'monai' or 'local'.
    """
    pref = (preferred or "auto").lower()
    if arch not in SEGMENTATION_MODEL_REGISTRY:
        valid = ", ".join(sorted(SEGMENTATION_MODEL_REGISTRY.keys()))
        raise NotImplementedError(
            f"Unknown 3D segmentation architecture {arch!r}. Supported: {valid}"
        )

    if pref == "local":
        mod = _build_local(arch, c)
        return mod, "local"

    if pref in ("official", "unetr_pp", "unetr_plus_plus", "unetrplusplus", "paper"):
        if arch != "unetrpp":
            raise NotImplementedError(
                f"model.preferred_source={preferred!r} is only supported for architecture unetrpp, "
                f"got {arch!r}."
            )
        from dinomim_pytorch.segmentation_models.official_unetrpp3d import (
            build_official_unetrpp3d,
        )

        return build_official_unetrpp3d(c), "official"

    if pref in ("monai", "auto"):
        mon = _try_monai(arch, c)
        if mon is not None:
            return mon, "monai"
        if pref == "monai" or pref == "auto":
            if SEGMENTATION_MODEL_REGISTRY[arch].get("monai") is not None:
                log_monai_unavailable(arch)
            try:
                mod = _build_local(arch, c)
                return mod, "local"
            except NotImplementedError:
                pass
            raise NotImplementedError(
                f"Requested 3D segmentation architecture {arch} is unavailable in both MONAI and local implementations."
            )

    raise ValueError(
        f"Invalid model.preferred_source: {preferred!r}. "
        "Use: monai, local, official, or auto."
    )


def build_3d_segmentation_model(config: Any) -> nn.Module:
    """
    Build a true 3D segmentation model from ``config`` (``model`` / ``segmentation_model`` / ``data`` / ``experiment``).

    Enforces ``spatial_dims==3``; for ``experiment.task==segmentation``, enforces ``data.input_type==volume_3d``.
    Writes ``outputs/logs/model_source_report.txt`` and may load SSL weights when ``model.ssl_init`` is set.
    """
    if not isinstance(config, dict):
        config = {}
    else:
        config = copy.deepcopy(config)

    _validate_3d_segmentation_config(config)

    c = _get_model_cfg(config)
    arch_raw = c.get("architecture", c.get("name", "unet3d"))
    arch = normalize_architecture(str(arch_raw))
    if arch not in SEGMENTATION_MODEL_REGISTRY:
        valid = ", ".join(sorted(SEGMENTATION_MODEL_REGISTRY.keys()))
        raise NotImplementedError(
            f"Unknown 3D segmentation architecture {arch_raw!r} (normalized: {arch!r}). Supported: {valid}"
        )

    preferred = str(c.get("preferred_source", "auto")).lower()
    model, source = _resolve_model(arch, c, preferred)
    model._dino_source_used = source  # type: ignore[attr-defined]
    model._dino_architecture = arch  # type: ignore[attr-defined]

    ssl_init = bool(c.get("ssl_init", False))
    missing: Optional[list] = None
    unexpected: Optional[list] = None
    if ssl_init and c.get("ssl_checkpoint"):
        r = load_dino_weights_into_downstream_model(
            model,
            c["ssl_checkpoint"],
            load_encoder_only=bool(c.get("load_encoder_only", True)),
            strict_load=bool(c.get("strict_load", False)) is True,
            adapt_input_channels=bool(c.get("adapt_input_channels", False)),
        )
        missing, unexpected = list(r.get("missing", [])) or [], list(r.get("unexpected", [])) or []
    else:
        missing, unexpected = [], []

    append_model_source_report(
        requested_architecture=str(arch_raw),
        preferred_source=preferred,
        actual_source=source,
        ssl_init=ssl_init,
        missing_keys=missing,
        unexpected_keys=unexpected,
        extra={"normalized_architecture": arch},
    )
    return model


def build_segmentation_model(config: Any) -> nn.Module:
    """
    Backward-compatible: same as :func:`build_3d_segmentation_model` for volume configs.
    Accepts ``segmentation_model.name`` or ``model.architecture`` etc.
    """
    return build_3d_segmentation_model(config)


__all__ = [
    "build_3d_segmentation_model",
    "build_segmentation_model",
    "get_merged_model_config",
    "SEGMENTATION_MODEL_REGISTRY",
    "MODEL_REGISTRY",
    "normalize_architecture",
]
