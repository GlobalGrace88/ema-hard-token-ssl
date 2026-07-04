#!/usr/bin/env python3
"""Smoke test: import core modules without GPU data."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    import dinomim_pytorch
    from dinomim_pytorch import paths, task_registry, checkpoint_metadata
    from dinomim_pytorch.segmentation_models import factory

    assert dinomim_pytorch is not None
    assert paths.repo_root().is_dir()
    task = task_registry.load_task("synapse")
    assert task["name"] == "synapse"
    assert task_registry.task_folds(task) == [0, 1, 2, 3, 4]
    print("smoke_pretrain: imports OK")


if __name__ == "__main__":
    main()
