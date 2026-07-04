#!/usr/bin/env python3
"""Task registry unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_synapse_task() -> None:
    from dinomim_pytorch.task_registry import (
        default_model,
        downstream_dataset,
        load_task,
        task_folds,
        upstream_dataset,
    )

    task = load_task("synapse")
    assert task["default_model"] == "unetrpp"
    assert upstream_dataset(task) == "public_ct_ssl"
    assert downstream_dataset(task) == "synapse"
    assert task_folds(task) == [0, 1, 2, 3, 4]
    assert default_model(task) == "unetrpp"


def test_model_registry() -> None:
    from dinomim_pytorch.task_registry import load_model

    unetrpp = load_model("unetrpp")
    assert unetrpp["architecture"] == "unetrpp"
    assert unetrpp["preferred_source"] == "official"

    swin = load_model("swin_unetr")
    assert swin["architecture"] == "swinunetr"
    assert swin["preferred_source"] == "monai"


def main() -> None:
    test_synapse_task()
    test_model_registry()
    print("test_task_registry: all passed")


if __name__ == "__main__":
    main()
