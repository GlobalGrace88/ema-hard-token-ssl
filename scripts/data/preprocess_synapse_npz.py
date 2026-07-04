#!/usr/bin/env python3
"""Build Synapse train/val manifests from existing nnFormer preprocessed npz + splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from dinomim_pytorch.datasets.nnformer_npz import list_npz_files, resolve_nnformer_npz_dir
from dinomim_pytorch.datasets.nnformer_splits import load_splits_final
from scripts.data.preprocess_common import (
    ensure_dir,
    load_preprocess_config,
    repo_root,
    symlink_or_copy,
    write_csv,
    write_splits_pkl,
)


def _rows_for_ids(npz_dir: Path, case_ids: list[str], split: str, mirror_dir: Path | None) -> list[dict[str, str]]:
    id_set = set(case_ids)
    rows: list[dict[str, str]] = []
    for p in list_npz_files(npz_dir):
        cid = p.stem
        if cid not in id_set:
            continue
        pre = p
        if mirror_dir is not None:
            pre = symlink_or_copy(p, mirror_dir / p.name)
        rows.append(
            {
                "case_id": cid,
                "npz_path": str(p.resolve()),
                "preprocessed_path": str(pre.resolve()),
                "split": split,
                "image": str(pre.resolve()),
                "label": str(pre.resolve()),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=repo_root() / "configs/data_preprocessing/synapse.yaml")
    ap.add_argument("--max_cases", type=int, default=0, help="mirror max N cases per split to local preprocessed/")
    args = ap.parse_args()
    cfg = load_preprocess_config(args.config)
    ds = cfg["dataset"]
    nnf_root = Path(ds.get("nnformer_preprocessed_dir", ds["raw_root"])).expanduser()
    out_root = repo_root() / ds["preprocessed_root"]
    manifest_dir = repo_root() / ds.get("manifest_dir", "data/manifests")
    splits_pkl = nnf_root / "splits_final.pkl"
    if not splits_pkl.is_file():
        raise SystemExit(f"missing splits: {splits_pkl}")

    data_cfg = {
        "nnformer_preprocessed_dir": str(nnf_root),
        "nnformer_npz_stage": ds.get("nnformer_npz_stage", "stage1"),
        "nnformer_npz_prefer_3d": True,
    }
    npz_dir = resolve_nnformer_npz_dir(data_cfg)
    if npz_dir is None:
        raise SystemExit(f"no npz dir under {nnf_root}")

    fold = int(ds.get("fold", 0))
    train_ids, val_ids = load_splits_final(splits_pkl, fold=fold)
    if args.max_cases > 0:
        train_ids = train_ids[: args.max_cases]
        val_ids = val_ids[: max(1, args.max_cases // 2)]

    mirror = ensure_dir(out_root) if args.max_cases > 0 else None
    train_rows = _rows_for_ids(npz_dir, train_ids, "train", mirror)
    val_rows = _rows_for_ids(npz_dir, val_ids, "val", mirror)

    fields = ["case_id", "npz_path", "preprocessed_path", "split", "image", "label"]
    write_csv(manifest_dir / "synapse_train.csv", train_rows, fields)
    write_csv(manifest_dir / "synapse_val.csv", val_rows, fields)

    local_splits = out_root / "splits_final.pkl"
    if mirror is not None:
        write_splits_pkl(local_splits, [r["case_id"] for r in train_rows], [r["case_id"] for r in val_rows])

    print(f"[synapse] train={len(train_rows)} val={len(val_rows)} npz_dir={npz_dir}")
    print(f"[synapse] manifests -> {manifest_dir}/synapse_train.csv, synapse_val.csv")


if __name__ == "__main__":
    main()
