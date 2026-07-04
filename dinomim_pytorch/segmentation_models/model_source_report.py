from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_LOG = logging.getLogger(__name__)

REPORT_FILE = os.path.join("outputs", "logs", "model_source_report.txt")


def _path(base: Optional[Path] = None) -> Path:
    p = (base or Path.cwd()) / REPORT_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def append_model_source_report(
    *,
    requested_architecture: str,
    preferred_source: str,
    actual_source: str,  # "monai" | "local"
    ssl_init: bool = False,
    missing_keys: Optional[List[str]] = None,
    unexpected_keys: Optional[List[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
    cwd: Optional[Path] = None,
) -> None:
    p = _path(cwd)
    lines = [
        f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---",
        f"requested_architecture: {requested_architecture}",
        f"preferred_source: {preferred_source}",
        f"actual_source: {actual_source}",
        f"ssl_init: {ssl_init}",
    ]
    if missing_keys is not None:
        lines.append(f"missing_keys ({len(missing_keys)}): {missing_keys!r}")
    if unexpected_keys is not None:
        lines.append(f"unexpected_keys ({len(unexpected_keys)}): {unexpected_keys!r}")
    if extra:
        for k, v in extra.items():
            lines.append(f"{k}: {v!r}")
    lines.append("")
    with open(p, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def log_monai_unavailable(architecture: str) -> None:
    msg = (
        f"MONAI implementation for {architecture} is unavailable. "
        "Falling back to local 3D implementation."
    )
    print(f"[segmentation] {msg}")
    _LOG.info(msg)
    p = _path()
    with open(p, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
