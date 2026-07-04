"""Resolve human-readable class names for per-class metric logging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _slug(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("/", "_")


def load_class_names_from_dataset_json(
    task_dir: Path,
    n_classes: int,
) -> Optional[List[str]]:
    path = Path(task_dir) / "dataset.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as fh:
        labels = (json.load(fh) or {}).get("labels") or {}
    names = [f"class_{i}" for i in range(n_classes)]
    if n_classes > 0:
        names[0] = "background"
    for key, label in labels.items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < n_classes:
            names[idx] = str(label)
    return names


def resolve_segmentation_class_names(
    data_cfg: Dict[str, Any],
    n_classes: int,
) -> List[str]:
    """
    Names for indices ``0 .. n_classes-1``.

    Priority: ``class_names`` in data config; ``dataset.json`` under
    ``nnformer_preprocessed_dir``; generic ``class_{i}`` fallbacks.
    """
    explicit = data_cfg.get("class_names")
    if explicit:
        names = [str(x) for x in explicit]
        if len(names) < n_classes:
            names.extend(f"class_{i}" for i in range(len(names), n_classes))
        return names[:n_classes]

    for key in ("nnformer_preprocessed_dir", "preprocessed_dir"):
        raw = data_cfg.get(key)
        if raw:
            loaded = load_class_names_from_dataset_json(Path(str(raw)), n_classes)
            if loaded:
                return loaded

    names = [f"class_{i}" for i in range(n_classes)]
    if n_classes > 0:
        names[0] = "background"
    return names


def format_per_class_metric_line(
    prefix: str,
    values: Sequence[float],
    class_names: Sequence[str],
    *,
    start_class: int = 1,
) -> str:
    parts: List[str] = []
    for i, v in enumerate(values):
        c = start_class + i
        name = class_names[c] if c < len(class_names) else f"class_{c}"
        parts.append(f"c{c:02d} {name}: {v:.4f}")
    return f"{prefix} " + " | ".join(parts)


def format_per_class_metric_multiline(
    prefix: str,
    values: Sequence[float],
    class_names: Sequence[str],
    *,
    metric: str = "Dice",
    start_class: int = 1,
) -> str:
    lines = [f"{prefix} per-class {metric}:"]
    for i, v in enumerate(values):
        c = start_class + i
        name = class_names[c] if c < len(class_names) else f"class_{c}"
        lines.append(f"  c{c:02d} {name}: {v:.4f}")
    return "\n".join(lines)


__all__ = [
    "load_class_names_from_dataset_json",
    "resolve_segmentation_class_names",
    "format_per_class_metric_line",
    "format_per_class_metric_multiline",
    "_slug",
]
