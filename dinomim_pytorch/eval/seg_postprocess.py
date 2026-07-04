"""
Dataset-agnostic post-processing for 3D segmentation predictions.

Ported from MAE_BYOL ``MAE_v3/medical_mim/common/eval/seg_postprocess.py`` (same
behavior as ``eval_vis.postprocess`` in ``unetr_brats.yaml``).
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

try:
    from scipy import ndimage as _nd
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "seg_postprocess requires scipy. Install with `pip install scipy>=1.7`."
    ) from exc


def _connectivity_structure(ndim: int, connectivity: int) -> np.ndarray:
    if ndim == 3:
        if connectivity == 6:
            return _nd.generate_binary_structure(3, 1)
        if connectivity == 18:
            return _nd.generate_binary_structure(3, 2)
        if connectivity == 26:
            return _nd.generate_binary_structure(3, 3)
        raise ValueError(f"Unsupported 3D connectivity: {connectivity}")
    if ndim == 2:
        if connectivity in (4, 6):
            return _nd.generate_binary_structure(2, 1)
        if connectivity in (8, 18, 26):
            return _nd.generate_binary_structure(2, 2)
        raise ValueError(f"Unsupported 2D connectivity: {connectivity}")
    return _nd.generate_binary_structure(ndim, 1)


def filter_components(
    binary_mask: np.ndarray,
    *,
    keep_largest_cc: bool = True,
    keep_top_k: int = 1,
    min_component_voxels: int = 0,
    connectivity: int = 26,
) -> np.ndarray:
    m = np.asarray(binary_mask, dtype=bool)
    if m.size == 0 or not m.any():
        return m
    structure = _connectivity_structure(m.ndim, connectivity)
    labels, n = _nd.label(m, structure=structure)
    if n == 0:
        return np.zeros_like(m, dtype=bool)

    sizes = np.bincount(labels.ravel())
    comp_sizes = sizes[1:]
    comp_ids = np.arange(1, n + 1, dtype=np.int64)

    if min_component_voxels and min_component_voxels > 0:
        keep_mask = comp_sizes >= int(min_component_voxels)
        comp_ids = comp_ids[keep_mask]
        comp_sizes = comp_sizes[keep_mask]

    if keep_largest_cc and comp_ids.size > 0 and keep_top_k is not None and int(keep_top_k) > 0:
        order = np.argsort(-comp_sizes, kind="mergesort")
        keep = comp_ids[order[: int(keep_top_k)]]
    else:
        keep = comp_ids

    if keep.size == 0:
        return np.zeros_like(m, dtype=bool)
    keep_set = set(int(k) for k in keep.tolist())
    max_id = int(labels.max())
    lut = np.zeros(max_id + 1, dtype=bool)
    for k in keep_set:
        if 0 <= k <= max_id:
            lut[k] = True
    return lut[labels]


def _pp_params(cfg: Optional[Mapping]) -> Dict[str, object]:
    cfg = dict(cfg or {})
    return {
        "keep_largest_cc": bool(cfg.get("keep_largest_cc", True)),
        "keep_top_k": int(cfg.get("keep_top_k", 1)),
        "min_component_voxels": int(cfg.get("min_component_voxels", 0)),
        "connectivity": int(cfg.get("connectivity", 26)),
    }


def postprocess_brats_nested(
    pred_nested: np.ndarray,
    cfg: Optional[Mapping] = None,
) -> np.ndarray:
    pred = np.asarray(pred_nested, dtype=np.int64)
    params = _pp_params(cfg)
    wt = pred >= 1
    tc = np.logical_or(pred == 2, pred == 3)
    et = pred == 3

    wt_f = filter_components(wt, **params)
    tc_f = filter_components(tc, **params)
    et_f = filter_components(et, **params)

    tc_f = np.logical_and(tc_f, wt_f)
    et_f = np.logical_and(et_f, tc_f)

    out = np.zeros_like(pred, dtype=np.int64)
    out[wt_f] = 1
    out[tc_f] = 2
    out[et_f] = 3
    return out


def postprocess_per_class(
    pred: np.ndarray,
    num_classes: int,
    cfg: Optional[Mapping] = None,
    background_class: int = 0,
) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.int64)
    params = _pp_params(cfg)
    out = np.full_like(pred, fill_value=int(background_class), dtype=np.int64)
    class_ids = [c for c in range(int(num_classes)) if c != int(background_class)]
    for cid in class_ids:
        mask = pred == cid
        if not mask.any():
            continue
        mask_f = filter_components(mask, **params)
        out[mask_f] = cid
    return out


def apply_seg_postprocess(
    pred: np.ndarray,
    *,
    num_classes: int,
    cfg: Optional[Mapping] = None,
) -> np.ndarray:
    c = dict(cfg or {})
    if not c.get("enabled", False):
        return np.asarray(pred)
    mode = str(c.get("mode", "auto")).lower()
    if mode == "off" or mode == "none":
        return np.asarray(pred)
    if mode == "auto":
        mode = "per_region" if int(num_classes) == 4 else "per_class"
    if mode == "per_region":
        if int(num_classes) != 4:
            return postprocess_per_class(pred, num_classes=num_classes, cfg=c)
        return postprocess_brats_nested(pred, cfg=c)
    if mode == "per_class":
        return postprocess_per_class(pred, num_classes=num_classes, cfg=c)
    raise ValueError(f"Unknown postprocess mode: {mode!r}")


def resolve_postprocess_cfg(cfg: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Merge ``postprocess`` from root config and ``eval_vis.postprocess`` (eval_vis wins)."""
    root = dict((cfg or {}).get("postprocess") or {}) if isinstance(cfg, dict) else {}
    ev = (cfg or {}).get("eval_vis") if isinstance(cfg, dict) else None
    if isinstance(ev, dict) and ev.get("postprocess"):
        root = {**root, **dict(ev.get("postprocess") or {})}
    return root


def postprocess_enabled(cfg: Optional[Mapping[str, Any]]) -> bool:
    return bool(resolve_postprocess_cfg(cfg).get("enabled", False))


__all__ = [
    "apply_seg_postprocess",
    "filter_components",
    "postprocess_brats_nested",
    "postprocess_per_class",
    "postprocess_enabled",
    "resolve_postprocess_cfg",
]
