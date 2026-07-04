#!/usr/bin/env python3
"""Upstream (public CT SSL) preprocessing entrypoint."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from dinomim_pytorch.paths import repo_root, ssl_ct_root, substitute_placeholders
from dinomim_pytorch.task_registry import load_task


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare upstream public CT SSL data")
    ap.add_argument("--task", default="synapse")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--preprocess", action="store_true")
    ap.add_argument("--make-manifest", action="store_true")
    args = ap.parse_args()

    task = load_task(args.task)
    upstream = task["upstream"]
    cfg_path = repo_root() / str(upstream["prepare_config"])
    root = ssl_ct_root()

    cmd = [
        sys.executable,
        str(repo_root() / "scripts" / "prepare_public_ct_ssl.py"),
        "--config",
        str(cfg_path),
        "--root",
        str(root),
    ]
    if args.download:
        cmd.append("--download")
    if args.preprocess:
        cmd.append("--preprocess")
    if args.make_manifest:
        cmd.append("--make-manifest")

    if len(cmd) <= 6:
        cmd.extend(["--preprocess", "--make-manifest"])

    # Substitute placeholders in config on the fly if needed
    raw = cfg_path.read_text(encoding="utf-8")
    if "${" in raw:
        tmp = repo_root() / "configs" / "generated" / args.task / "public_ct_ssl_resolved.yaml"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        import yaml

        resolved = substitute_placeholders(yaml.safe_load(raw))
        with tmp.open("w", encoding="utf-8") as f:
            yaml.safe_dump(resolved, f, sort_keys=False)
        cmd[cmd.index(str(cfg_path))] = str(tmp)

    print("[preprocess_upstream]", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
