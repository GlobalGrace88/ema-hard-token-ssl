#!/usr/bin/env python3
"""Unified CLI for ema-hard-token-ssl Tier-1 Synapse reproduction."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import _bootstrap  # noqa: F401

from dinomim_pytorch.checkpoint_metadata import validate_checkpoint_task
from dinomim_pytorch.paths import repo_root
from dinomim_pytorch.task_registry import (
    default_model,
    downstream_dataset,
    load_task,
    task_folds,
    upstream_dataset,
)


def _resolve_folds(task_name: str, fold_arg: str) -> List[int]:
    task = load_task(task_name)
    if fold_arg == "all":
        return task_folds(task)
    return [int(fold_arg)]


def _run_script(script: str, args: List[str]) -> int:
    cmd = [sys.executable, str(repo_root() / "scripts" / script), *args]
    print("[cli]", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def _materialize(task: str, model: str, phase: str, method: str, folds: List[int]) -> None:
    for fold in folds:
        args = ["--task", task, "--model", model, "--phase", phase]
        if phase != "pretrain":
            args.extend(["--method", method, "--fold", str(fold)])
        _run_script("materialize_configs.py", args)


def cmd_preprocess(args: argparse.Namespace) -> int:
    stages = ["upstream", "downstream"] if args.stage == "both" else [args.stage]
    rc = 0
    if "upstream" in stages:
        up_args = ["--task", args.task]
        if args.download:
            up_args.append("--download")
        rc = max(rc, _run_script("preprocess_upstream.py", up_args))
    if "downstream" in stages:
        task = load_task(args.task)
        folds = task_folds(task)
        for fold in folds:
            rc = max(rc, _run_script("preprocess_downstream.py", ["--task", args.task, "--fold", str(fold)]))
    return rc


def cmd_run_pretrain(args: argparse.Namespace) -> int:
    model = args.model or default_model(load_task(args.task))
    suffix = "_v5" if getattr(args, "recipe", "v4") == "v5" else ""
    cfg_path = repo_root() / "configs" / "generated" / args.task / f"pretrain_{model}{suffix}.yaml"
    if not cfg_path.is_file():
        _run_script(
            "materialize_configs.py",
            [
                "--task",
                args.task,
                "--model",
                model,
                "--phase",
                "pretrain",
                "--recipe",
                getattr(args, "recipe", "v4"),
            ],
        )
    extra = []
    if args.epochs is not None:
        extra.extend(["--epochs", str(args.epochs)])
    return _run_script(
        "pretrain_unetrpp_inpainting_feature_reconstruction.py",
        ["--config", str(cfg_path), *extra],
    )


def cmd_run_finetune(args: argparse.Namespace) -> int:
    model = args.model or default_model(load_task(args.task))
    folds = _resolve_folds(args.task, args.fold)
    suffix = _recipe_suffix(args.recipe)
    rc = 0
    for fold in folds:
        cfg_path = (
            repo_root()
            / "configs"
            / "generated"
            / args.task
            / "finetune"
            / f"{model}_{args.method}{suffix}_fold{fold}.yaml"
        )
        if not cfg_path.is_file():
            _run_script(
                "materialize_configs.py",
                [
                    "--task",
                    args.task,
                    "--model",
                    model,
                    "--phase",
                    "finetune",
                    "--method",
                    args.method,
                    "--fold",
                    str(fold),
                    "--recipe",
                    args.recipe,
                ],
            )
        rc = max(rc, _run_script("finetune_mri_segmentation.py", ["--config", str(cfg_path)]))
    return rc


def cmd_run_eval(args: argparse.Namespace) -> int:
    task = load_task(args.task)
    model = args.model or default_model(task)
    folds = _resolve_folds(args.task, args.fold)

    ckpt = Path(args.checkpoint).expanduser().resolve()
    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    validate_checkpoint_task(
        ckpt,
        expected_task=args.task,
        expected_model=model,
        expected_downstream=downstream_dataset(task),
        expected_upstream=upstream_dataset(task) if args.method == "ours" else None,
    )

    rc = 0
    for fold in folds:
        cfg_path = (
            repo_root()
            / "configs"
            / "generated"
            / args.task
            / "eval"
            / f"{model}_{args.method}_fold{fold}_official.yaml"
        )
        if not cfg_path.is_file():
            _run_script(
                "materialize_configs.py",
                [
                    "--task",
                    args.task,
                    "--model",
                    model,
                    "--phase",
                    "eval",
                    "--method",
                    args.method,
                    "--fold",
                    str(fold),
                ],
            )
        eval_args = ["--config", str(cfg_path), "--checkpoint", str(ckpt)]
        if args.official:
            eval_args.append("--official_npz")
        rc = max(rc, _run_script("eval.py", eval_args))
    return rc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ema-hard-token-ssl CLI")
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("preprocess", help="Prepare upstream/downstream data")
    pp.add_argument("--task", default="synapse")
    pp.add_argument("--stage", choices=["upstream", "downstream", "both"], default="both")
    pp.add_argument("--download", action="store_true", help="Attempt upstream dataset download")
    pp.set_defaults(func=cmd_preprocess)

    pr = sub.add_parser("run", help="Run pretrain / finetune / eval")
    pr_sub = pr.add_subparsers(dest="run_cmd", required=True)

    pt = pr_sub.add_parser("pretrain")
    pt.add_argument("--task", default="synapse")
    pt.add_argument("--model", default=None)
    pt.add_argument("--epochs", type=int, default=None)
    pt.add_argument("--recipe", choices=["v4", "v5"], default="v4")
    pt.set_defaults(func=cmd_run_pretrain)

    pf = pr_sub.add_parser("finetune")
    pf.add_argument("--task", default="synapse")
    pf.add_argument("--model", default=None)
    pf.add_argument("--method", choices=["scratch", "ours"], required=True)
    pf.add_argument("--fold", default="all", help="Fold index or 'all'")
    pf.add_argument("--recipe", choices=["v4", "v5"], default="v4")
    pf.set_defaults(func=cmd_run_finetune)

    pe = pr_sub.add_parser("eval")
    pe.add_argument("--task", default="synapse")
    pe.add_argument("--model", default=None)
    pe.add_argument("--method", choices=["scratch", "ours"], default="ours")
    pe.add_argument("--fold", default="all")
    pe.add_argument("--checkpoint", required=True)
    pe.add_argument("--official", action="store_true", help="Official npz sliding-window eval")
    pe.set_defaults(func=cmd_run_eval)

    mc = sub.add_parser("materialize", help="Generate configs from templates")
    mc.add_argument("--task", default="synapse")
    mc.add_argument("--model", default="unetrpp")
    mc.add_argument("--phase", choices=["pretrain", "finetune", "eval", "all"], default="all")
    mc.add_argument("--method", choices=["scratch", "ours", "both"], default="both")
    mc.add_argument("--fold", type=int, default=None)
    mc.add_argument("--recipe", choices=["v4", "v5"], default="v4")

    def _cmd_materialize(a: argparse.Namespace) -> int:
        args_list = ["--task", a.task, "--model", a.model, "--phase", a.phase, "--method", a.method, "--recipe", a.recipe]
        if a.fold is not None:
            args_list.extend(["--fold", str(a.fold)])
        return _run_script("materialize_configs.py", args_list)

    mc.set_defaults(func=_cmd_materialize)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
