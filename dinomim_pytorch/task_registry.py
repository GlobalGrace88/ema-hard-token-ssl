"""Task registry for Tier-1 Synapse reproduction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

from dinomim_pytorch.paths import repo_root, substitute_placeholders


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping in {path}")
    return raw


def load_task(name: str) -> Dict[str, Any]:
    path = repo_root() / "configs" / "tasks" / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Unknown task '{name}': missing {path}")
    task = _load_yaml(path)
    return substitute_placeholders(task)


def load_model(name: str) -> Dict[str, Any]:
    path = repo_root() / "configs" / "models" / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Unknown model '{name}': missing {path}")
    return _load_yaml(path)


def list_tasks() -> List[str]:
    d = repo_root() / "configs" / "tasks"
    return sorted(p.stem for p in d.glob("*.yaml"))


def task_folds(task: Dict[str, Any]) -> List[int]:
    folds = task.get("folds") or [0, 1, 2, 3, 4]
    return [int(f) for f in folds]


def default_model(task: Dict[str, Any]) -> str:
    return str(task.get("default_model", "unetrpp"))


def upstream_dataset(task: Dict[str, Any]) -> str:
    return str((task.get("upstream") or {}).get("dataset", ""))


def downstream_dataset(task: Dict[str, Any]) -> str:
    return str((task.get("downstream") or {}).get("dataset", ""))


__all__ = [
    "load_task",
    "load_model",
    "list_tasks",
    "task_folds",
    "default_model",
    "upstream_dataset",
    "downstream_dataset",
]
