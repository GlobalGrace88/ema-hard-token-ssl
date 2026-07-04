"""Checkpoint task metadata for release-safe eval (no cross-dataset mismatch)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from dinomim_pytorch.task_registry import downstream_dataset, upstream_dataset


def _spatial_from_cfg(cfg: Mapping[str, Any]) -> list[int]:
    model = dict(cfg.get("model") or {})
    for key in ("img_size", "spatial_size"):
        sz = model.get(key)
        if isinstance(sz, (list, tuple)) and len(sz) >= 3:
            return [int(sz[0]), int(sz[1]), int(sz[2])]
    data = dict(cfg.get("data") or {})
    sz = data.get("image_size")
    if isinstance(sz, (list, tuple)) and len(sz) >= 3:
        return [int(sz[0]), int(sz[1]), int(sz[2])]
    return []


def build_metadata(
    cfg: Mapping[str, Any],
    *,
    phase: str,
    task_name: str,
    model_name: str,
    fold: Optional[int] = None,
    method: Optional[str] = None,
) -> Dict[str, Any]:
    exp = dict(cfg.get("experiment") or {})
    model = dict(cfg.get("model") or {})
    meta: Dict[str, Any] = {
        "release": "ema-hard-token-ssl",
        "phase": phase,
        "task": task_name,
        "model": model_name,
        "upstream_dataset": exp.get("upstream_dataset")
        or (upstream_dataset(cfg) if isinstance(cfg.get("upstream"), dict) else None),
        "downstream_dataset": exp.get("dataset")
        or (downstream_dataset(cfg) if isinstance(cfg.get("downstream"), dict) else None),
        "image_size": _spatial_from_cfg(cfg),
        "unetrpp_official_variant": model.get("unetrpp_official_variant"),
        "architecture": model.get("architecture") or model.get("backbone_name"),
        "method": method or exp.get("init_type") or exp.get("method"),
    }
    if fold is not None:
        meta["fold"] = int(fold)
    return meta


def sidecar_path(ckpt_path: Path) -> Path:
    return ckpt_path.with_suffix(ckpt_path.suffix + ".meta.json")


def save_sidecar(ckpt_path: Path, metadata: Mapping[str, Any]) -> None:
    path = sidecar_path(ckpt_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(metadata), f, indent=2, sort_keys=True)


def load_metadata(ckpt_path: Path) -> Dict[str, Any]:
    ckpt_path = Path(ckpt_path)
    side = sidecar_path(ckpt_path)
    if side.is_file():
        with side.open(encoding="utf-8") as f:
            return json.load(f)
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)  # type: ignore[call-arg]
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(ckpt, dict):
        extra = ckpt.get("release_metadata") or ckpt.get("extra")
        if isinstance(extra, dict) and extra.get("release") == "ema-hard-token-ssl":
            return dict(extra)
        cfg = ckpt.get("cfg") or ckpt.get("config")
        if isinstance(cfg, dict):
            exp = dict(cfg.get("experiment") or {})
            model = dict(cfg.get("model") or {})
            return {
                "release": "ema-hard-token-ssl",
                "task": exp.get("dataset") or exp.get("task"),
                "model": model.get("architecture") or model.get("backbone_name"),
                "downstream_dataset": exp.get("dataset"),
                "image_size": _spatial_from_cfg(cfg),
                "unetrpp_official_variant": model.get("unetrpp_official_variant"),
            }
    return {}


def validate_checkpoint_task(
    ckpt_path: Path,
    *,
    expected_task: str,
    expected_model: str,
    expected_downstream: str,
    expected_upstream: Optional[str] = None,
) -> None:
    meta = load_metadata(ckpt_path)
    if not meta:
        raise ValueError(
            f"Checkpoint {ckpt_path} has no release metadata. "
            "Re-train with this release or add a .pt.meta.json sidecar."
        )

    def _norm(x: Any) -> str:
        return str(x or "").strip().lower()

    mismatches = []
    if meta.get("task") and _norm(meta["task"]) not in (_norm(expected_task), _norm(expected_downstream)):
        mismatches.append(f"task: ckpt={meta.get('task')!r} expected={expected_task!r}")
    if meta.get("model") and _norm(meta["model"]) != _norm(expected_model):
        mismatches.append(f"model: ckpt={meta.get('model')!r} expected={expected_model!r}")
    if meta.get("downstream_dataset") and _norm(meta["downstream_dataset"]) != _norm(expected_downstream):
        mismatches.append(
            f"downstream_dataset: ckpt={meta.get('downstream_dataset')!r} expected={expected_downstream!r}"
        )
    if expected_upstream and meta.get("upstream_dataset"):
        if _norm(meta["upstream_dataset"]) != _norm(expected_upstream):
            mismatches.append(
                f"upstream_dataset: ckpt={meta.get('upstream_dataset')!r} expected={expected_upstream!r}"
            )
    if mismatches:
        raise ValueError(
            "Checkpoint task metadata mismatch (cross-dataset eval forbidden):\n  "
            + "\n  ".join(mismatches)
        )


def attach_to_payload(payload: Dict[str, Any], metadata: Mapping[str, Any]) -> Dict[str, Any]:
    payload = dict(payload)
    payload["release_metadata"] = dict(metadata)
    return payload


__all__ = [
    "build_metadata",
    "save_sidecar",
    "load_metadata",
    "validate_checkpoint_task",
    "attach_to_payload",
]
