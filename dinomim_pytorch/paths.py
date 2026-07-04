"""Path resolution for the ema-hard-token-ssl release (no hardcoded machine paths)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

_REPO_ROOT: Optional[Path] = None
_PATHS_CACHE: Optional[Dict[str, Any]] = None

ENV_MAP = {
    "data_root": "DINOMIM_DATA_ROOT",
    "nnformer_dir": "DINOMIM_NNFORMER_DIR",
    "ssl_ct_root": "DINOMIM_SSL_CT_ROOT",
    "unetr_pp_root": "UNETR_PP_ROOT",
    "output_root": "DINOMIM_OUTPUT_ROOT",
}


def repo_root() -> Path:
    global _REPO_ROOT
    if _REPO_ROOT is None:
        _REPO_ROOT = Path(__file__).resolve().parents[1]
    return _REPO_ROOT


def paths_yaml_candidates() -> list[Path]:
    root = repo_root()
    return [
        root / "paths.yaml",
        root / "configs" / "paths.yaml",
    ]


def load_paths_config(*, force_reload: bool = False) -> Dict[str, Any]:
    global _PATHS_CACHE
    if _PATHS_CACHE is not None and not force_reload:
        return dict(_PATHS_CACHE)

    merged: Dict[str, Any] = {}
    for p in paths_yaml_candidates():
        if not p.is_file():
            continue
        with p.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if isinstance(raw, dict):
            merged.update(raw)

    for key, env_name in ENV_MAP.items():
        val = os.environ.get(env_name, "").strip()
        if val:
            merged[key] = val

    _PATHS_CACHE = merged
    return dict(merged)


def resolve_path(key: str, default: Optional[str] = None) -> Path:
    cfg = load_paths_config()
    raw = cfg.get(key) or default or ""
    if not str(raw).strip():
        env = ENV_MAP.get(key, "")
        raise FileNotFoundError(
            f"Missing path '{key}'. Set it in paths.yaml or export {env}."
        )
    return Path(str(raw)).expanduser().resolve()


def data_root() -> Path:
    return resolve_path("data_root")


def nnformer_dir() -> Path:
    return resolve_path("nnformer_dir")


def ssl_ct_root() -> Path:
    return resolve_path("ssl_ct_root")


def unetr_pp_root() -> Path:
    return resolve_path("unetr_pp_root")


def output_root() -> Path:
    cfg = load_paths_config()
    raw = cfg.get("output_root") or os.environ.get("DINOMIM_OUTPUT_ROOT", "outputs")
    return Path(str(raw)).expanduser().resolve()


def synapse_nnformer_task_dir() -> Path:
    return nnformer_dir() / "nnFormer_preprocessed" / "Task002_Synapse"


def ssl_ct_manifest_csv() -> Path:
    return ssl_ct_root() / "processed" / "manifest_ssl_ct.csv"


def _optional_path(key: str) -> str:
    cfg = load_paths_config()
    raw = cfg.get(key) or os.environ.get(ENV_MAP.get(key, ""), "")
    if not str(raw).strip():
        return ""
    return str(Path(str(raw)).expanduser().resolve())


def substitute_placeholders(obj: Any, mapping: Optional[Mapping[str, str]] = None) -> Any:
    """Replace ``${key}`` tokens in nested dict/list/str structures."""
    if mapping is None:
        nnf = _optional_path("nnformer_dir")
        ssl = _optional_path("ssl_ct_root")
        mapping = {
            "data_root": _optional_path("data_root"),
            "nnformer_dir": nnf,
            "ssl_ct_root": ssl,
            "unetr_pp_root": _optional_path("unetr_pp_root"),
            "output_root": str(output_root()),
            "synapse_task_dir": str(Path(nnf) / "nnFormer_preprocessed" / "Task002_Synapse") if nnf else "",
            "ssl_manifest_csv": str(Path(ssl) / "processed" / "manifest_ssl_ct.csv") if ssl else "",
        }

    if isinstance(obj, dict):
        return {k: substitute_placeholders(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [substitute_placeholders(v, mapping) for v in obj]
    if isinstance(obj, str):
        out = obj
        for k, v in mapping.items():
            out = out.replace(f"${{{k}}}", v)
        return out
    return obj


__all__ = [
    "repo_root",
    "load_paths_config",
    "resolve_path",
    "data_root",
    "nnformer_dir",
    "ssl_ct_root",
    "unetr_pp_root",
    "output_root",
    "synapse_nnformer_task_dir",
    "ssl_ct_manifest_csv",
    "substitute_placeholders",
]
