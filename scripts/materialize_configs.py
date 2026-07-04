#!/usr/bin/env python3
"""Generate per-fold finetune and eval YAMLs from templates."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

import _bootstrap  # noqa: F401

from dinomim_pytorch.paths import output_root, repo_root, substitute_placeholders
from dinomim_pytorch.task_registry import load_model, load_task, task_folds


def _load_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _render_template(text: str, mapping: Dict[str, Any]) -> str:
    out = text
    for key, val in mapping.items():
        out = out.replace("{" + key + "}", str(val))
    return out


def _write_yaml(path: Path, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)


def materialize_pretrain(task_name: str, model_name: str) -> Path:
    task = load_task(task_name)
    model_cfg = load_model(model_name)
    tpl_path = repo_root() / str(task["pretrain"]["template"])
    cfg = yaml.safe_load(_load_template(tpl_path)) or {}
    cfg = substitute_placeholders(cfg)
    cfg["model"] = _deep_merge(cfg.get("model") or {}, model_cfg)
    cfg.setdefault("experiment", {})["model"] = model_name
    cfg["experiment"]["task_name"] = task_name
    out = repo_root() / "configs" / "generated" / task_name / f"pretrain_{model_name}.yaml"
    _write_yaml(out, cfg)
    return out


def _pretrain_ckpt(task_name: str, model_name: str) -> str:
    root = output_root()
    return str(root / "pretrain" / "synapse_stage234" / "best.pt")


def materialize_finetune(
    task_name: str,
    model_name: str,
    *,
    method: str,
    fold: int,
) -> Path:
    task = load_task(task_name)
    model_cfg = load_model(model_name)
    tpl_path = repo_root() / str(task["downstream_templates"][method if method == "ours" else "scratch"])
    text = _load_template(tpl_path)
    is_ours = method == "ours"
    mapping = {
        "fold": fold,
        "method": method,
        "pretrain_ckpt": _pretrain_ckpt(task_name, model_name) if is_ours else "null",
        "ssl_init": "true" if is_ours else "false",
        "ssl_method": "feature_recon_v4_hard_stage234_smoothl1" if is_ours else "none",
        "load_encoder_only": "true" if is_ours else "false",
    }
    rendered = substitute_placeholders(_render_template(text, mapping))
    cfg = yaml.safe_load(rendered) or {}
    cfg["model"] = _deep_merge(cfg.get("model") or {}, model_cfg)
    cfg.setdefault("experiment", {})["task_name"] = task_name
    cfg["experiment"]["model"] = model_name
    cfg["experiment"]["fold"] = fold
    if not is_ours:
        cfg["pretrained"]["checkpoint"] = None
        cfg["model"]["ssl_checkpoint"] = None
    out = repo_root() / "configs" / "generated" / task_name / "finetune" / f"{model_name}_{method}_fold{fold}.yaml"
    _write_yaml(out, cfg)
    return out


def materialize_eval(
    task_name: str,
    model_name: str,
    *,
    method: str,
    fold: int,
    overlap: float = 0.5,
) -> Path:
    task = load_task(task_name)
    model_cfg = load_model(model_name)
    finetune_cfg = yaml.safe_load(
        materialize_finetune(task_name, model_name, method=method, fold=fold).read_text(encoding="utf-8")
    )
    data = dict(finetune_cfg.get("data") or {})
    model = _deep_merge(dict(finetune_cfg.get("model") or {}), model_cfg)
    cfg: Dict[str, Any] = {
        "experiment": {
            "name": f"{model_name}_synapse_{method}_fold{fold}_official_npz_overlap{int(overlap * 100):03d}",
            "modality": "ct",
            "task": "segmentation",
            "dataset": "synapse",
            "task_name": task_name,
            "model": model_name,
            "method": method,
            "fold": fold,
        },
        "data": data,
        "model": model,
        "inference": {
            "roi_size": data.get("image_size", [64, 128, 128]),
            "sw_batch_size": 1,
            "overlap": overlap,
            "mode": "gaussian",
            "tta": False,
        },
        "eval_vis": {"enabled": False},
        "output": {
            "dir": str(
                output_root() / "downstream" / "synapse" / method / f"fold_{fold}" / f"eval_official_npz_overlap{int(overlap * 100):03d}"
            )
        },
    }
    cfg = substitute_placeholders(cfg)
    out = repo_root() / "configs" / "generated" / task_name / "eval" / f"{model_name}_{method}_fold{fold}_official.yaml"
    _write_yaml(out, cfg)
    return out


def materialize_all(
    task_name: str,
    model_name: str,
    *,
    methods: Optional[List[str]] = None,
    folds: Optional[List[int]] = None,
) -> Dict[str, List[Path]]:
    task = load_task(task_name)
    folds = folds or task_folds(task)
    methods = methods or ["scratch", "ours"]
    paths: Dict[str, List[Path]] = {"pretrain": [], "finetune": [], "eval": []}
    paths["pretrain"].append(materialize_pretrain(task_name, model_name))
    overlap = float((task.get("eval") or {}).get("official_overlap", 0.5))
    for method in methods:
        for fold in folds:
            paths["finetune"].append(materialize_finetune(task_name, model_name, method=method, fold=fold))
            paths["eval"].append(materialize_eval(task_name, model_name, method=method, fold=fold, overlap=overlap))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Materialize per-fold configs from templates")
    ap.add_argument("--task", default="synapse")
    ap.add_argument("--model", default="unetrpp")
    ap.add_argument("--phase", choices=["pretrain", "finetune", "eval", "all"], default="all")
    ap.add_argument("--method", choices=["scratch", "ours", "both"], default="both")
    ap.add_argument("--fold", type=int, default=None, help="Single fold (default: all task folds)")
    args = ap.parse_args()

    task = load_task(args.task)
    folds = [args.fold] if args.fold is not None else task_folds(task)
    methods = ["scratch", "ours"] if args.method == "both" else [args.method]

    if args.phase in ("pretrain", "all"):
        p = materialize_pretrain(args.task, args.model)
        print(p)
    if args.phase in ("finetune", "all"):
        for method in methods:
            for fold in folds:
                p = materialize_finetune(args.task, args.model, method=method, fold=fold)
                print(p)
    if args.phase in ("eval", "all"):
        overlap = float((task.get("eval") or {}).get("official_overlap", 0.5))
        for method in methods:
            for fold in folds:
                p = materialize_eval(args.task, args.model, method=method, fold=fold, overlap=overlap)
                print(p)


if __name__ == "__main__":
    main()
