"""Organ-wise post-processing for Synapse 8-organ label maps."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import yaml

from dinomim_pytorch.eval.seg_postprocess import filter_components

try:
    from scipy import ndimage as ndi
except ImportError as exc:  # pragma: no cover
    raise ImportError("synapse_postprocess requires scipy") from exc


def load_postprocess_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Postprocess config must be a mapping: {path}")
    return cfg


def _organ_params(cfg: Mapping[str, Any], organ_name: str) -> Dict[str, Any]:
    pp = dict(cfg.get("postprocess") or {})
    return {
        "keep_largest_cc": bool((pp.get("keep_largest_connected_component") or {}).get(organ_name, True)),
        "keep_top_k": int((pp.get("keep_top_k_components") or {}).get(organ_name, 1)),
        "min_component_voxels": int((pp.get("remove_small_components_voxels") or {}).get(organ_name, 0)),
        "fill_holes": bool((pp.get("fill_holes") or {}).get(organ_name, False)),
        "connectivity": int(pp.get("connectivity", 26)),
    }


def postprocess_organ_mask(mask: np.ndarray, params: Mapping[str, Any]) -> Tuple[np.ndarray, Dict[str, int]]:
    """Connected-component filtering (+ optional hole fill) for one binary organ mask."""
    m = np.asarray(mask, dtype=bool)
    stats = {"num_components_before": 0, "num_components_after": 0, "voxels_removed": 0}
    if m.size == 0 or not m.any():
        return m, stats

    structure = ndi.generate_binary_structure(m.ndim, 1 if int(params.get("connectivity", 26)) == 6 else 3)
    labels, n_before = ndi.label(m, structure=structure)
    stats["num_components_before"] = int(n_before)
    vox_before = int(m.sum())

    proc = m.copy()
    if bool(params.get("fill_holes", False)):
        proc = ndi.binary_fill_holes(proc)

    proc = filter_components(
        proc,
        keep_largest_cc=bool(params.get("keep_largest_cc", True)),
        keep_top_k=int(params.get("keep_top_k", 1)),
        min_component_voxels=int(params.get("min_component_voxels", 0)),
        connectivity=int(params.get("connectivity", 26)),
    )
    labels_after, n_after = ndi.label(proc, structure=structure)
    stats["num_components_after"] = int(n_after)
    stats["voxels_removed"] = max(0, vox_before - int(proc.sum()))
    return proc, stats


def postprocess_synapse_label_map(
    pred: np.ndarray,
    cfg: Mapping[str, Any],
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Per-organ CC postprocess on a full integer label map.

    Only labels listed in ``cfg['labels']`` are processed; all other voxels become 0.
    """
    pred = np.asarray(pred)
    out = np.zeros_like(pred, dtype=np.int32)
    summaries: List[Dict[str, Any]] = []
    labels_map = dict(cfg.get("labels") or {})
    for label_raw, organ_name in labels_map.items():
        label_id = int(label_raw)
        organ = str(organ_name)
        mask = pred == label_id
        params = _organ_params(cfg, organ)
        proc, st = postprocess_organ_mask(mask, params)
        out[proc] = label_id
        summaries.append(
            {
                "label": label_id,
                "organ": organ,
                "num_components_before": st["num_components_before"],
                "num_components_after": st["num_components_after"],
                "voxels_removed": st["voxels_removed"],
            }
        )
    return out, summaries


__all__ = [
    "load_postprocess_config",
    "postprocess_organ_mask",
    "postprocess_synapse_label_map",
]
