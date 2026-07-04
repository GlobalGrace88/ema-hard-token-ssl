"""Segmentation finetune/eval from nnFormer preprocessed ``*.npz`` (image + label in one file)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset

from dinomim_pytorch.datasets.medical_3d_segmentation_dataset import (
    _IdentityTransform,
    apply_label_remap,
)
from dinomim_pytorch.datasets.nnformer_npz import (
    case_id_from_npz,
    list_npz_files,
    load_nnformer_npz_image_label,
    resolve_nnformer_npz_dir,
    split_npz_paths_train_val,
)


class NnformerNpzSegDataset(Dataset):
    """
    One ``*.npz`` per case; compatible with ``_load_dict`` / ``rows`` used in eval viz.

    Config: ``data.loader: nnformer_npz`` plus ``nnformer_preprocessed_dir`` or ``nnformer_npz_dir``.
  """

    def __init__(
        self,
        data_config: Dict[str, Any],
        *,
        train: bool = True,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        npz_paths: Optional[List[Path]] = None,
    ) -> None:
        self.cfg: Dict[str, Any] = deepcopy(data_config or {})
        self.train = train
        self.image_key = str(self.cfg.get("image_key", "image"))
        self.label_key = str(self.cfg.get("label_key", "label"))
        self.num_classes = int(self.cfg.get("num_classes", 2))
        self.label_remap = self.cfg.get("label_remap")
        self.tfm = transform if transform is not None else _IdentityTransform()

        folder = resolve_nnformer_npz_dir(self.cfg)
        if folder is None:
            raise FileNotFoundError(
                "nnformer_npz loader: set data.nnformer_npz_dir or data.nnformer_preprocessed_dir"
            )
        self.folder = folder
        all_paths = list_npz_files(folder)
        if npz_paths is not None:
            self.paths = sorted(npz_paths, key=lambda p: case_id_from_npz(p))
        else:
            self.paths = split_npz_paths_train_val(all_paths, self.cfg, train=train)
        if not self.paths:
            split = "train" if train else "val"
            raise FileNotFoundError(
                f"No *.npz for {split} under {folder}. "
                "Set val_frac, index_val (case ids), or nnformer_npz_val_dir."
            )
        self.rows: List[Dict[str, str]] = [
            {
                "case_id": case_id_from_npz(p),
                "npz": str(p),
                self.image_key: str(p),
                self.label_key: str(p),
            }
            for p in self.paths
        ]

    def __len__(self) -> int:
        return len(self.paths)

    def _load_dict(self, row: Dict[str, str]) -> Dict[str, Any]:
        p = row.get("npz") or row.get(self.image_key, "")
        if not p:
            raise KeyError(f"No npz path in row {row!r}")
        img, lbl = load_nnformer_npz_image_label(p, self.cfg)
        lbl = apply_label_remap(lbl, self.num_classes, self.label_remap)
        return {
            self.image_key: img,
            self.label_key: lbl.astype(np.int64),
            "meta": dict(row),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.tfm(self._load_dict(self.rows[idx]))


__all__ = ["NnformerNpzSegDataset"]
