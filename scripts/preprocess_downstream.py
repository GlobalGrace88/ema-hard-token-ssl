#!/usr/bin/env python3
"""Downstream Synapse (nnFormer Task002) manifest preprocessing."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from dinomim_pytorch.paths import repo_root, synapse_nnformer_task_dir, substitute_placeholders
from dinomim_pytorch.task_registry import load_task, task_folds


def _write_downstream_preprocess_cfg(task_name: str, fold: int) -> Path:
    import yaml

    task = load_task(task_name)
    downstream = task["downstream"]
    cfg = substitute_placeholders(
        {
            "dataset": {
                "name": downstream["dataset"],
                "raw_root": downstream["nnformer_preprocessed_dir"],
                "nnformer_preprocessed_dir": downstream["nnformer_preprocessed_dir"],
                "preprocessed_root": f"${{output_root}}/manifests/synapse",
                "manifest_dir": f"${{output_root}}/manifests/synapse",
                "fold": fold,
                "nnformer_npz_stage": "stage1",
            }
        }
    )
    out = repo_root() / "configs" / "generated" / task_name / f"preprocess_downstream_fold{fold}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Synapse downstream manifests from nnFormer npz")
    ap.add_argument("--task", default="synapse")
    ap.add_argument("--fold", type=int, default=None, help="Fold index (default: all)")
    args = ap.parse_args()

    task = load_task(args.task)
    folds = [args.fold] if args.fold is not None else task_folds(task)
    nnf_dir = synapse_nnformer_task_dir()
    if not nnf_dir.is_dir():
        raise SystemExit(
            f"Synapse nnFormer preprocessed dir not found: {nnf_dir}\n"
            "Download Task002_Synapse and run nnFormer preprocessing, then set nnformer_dir in paths.yaml."
        )

    script = repo_root() / "scripts" / "data" / "preprocess_synapse_npz.py"
    for fold in folds:
        cfg = _write_downstream_preprocess_cfg(args.task, fold)
        cmd = [sys.executable, str(script), "--config", str(cfg)]
        print(f"[preprocess_downstream] fold={fold}", " ".join(cmd), flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            raise SystemExit(rc)


if __name__ == "__main__":
    main()
