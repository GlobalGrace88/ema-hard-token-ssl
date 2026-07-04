"""
Generic 3D medical segmentation from a CSV index (BMCV/custom), NIfTI loading, label remapping.

Tensor convention: batch from DataLoader is ``[B, C, D, H, W]`` for ``image``;
``label`` is ``[B, D, H, W]`` or ``[B, 1, D, H, W]`` before loss (may add channel in transforms).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import nibabel as nib
except Exception:  # noqa: BLE001
    nib = None  # type: ignore[assignment]

try:
    from monai.transforms import Compose, MapTransform, ToTensord
except Exception:  # noqa: BLE001
    Compose = None  # type: ignore[assignment]

def get_spacing_from_meta(path: Union[str, Path]) -> Optional[Tuple[float, float, float]]:
    return get_spacing_from_nifti(path)


def load_nifti_tensor(path: Union[str, Path]) -> torch.Tensor:
    arr, _ = load_nifti_array(path)
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


__all__ = [
    "load_index_csv",
    "unwrap_monai_dict_batch",
    "apply_label_remap",
    "get_spacing_from_nifti",
    "get_spacing_from_meta",
    "load_nifti_array",
    "load_nifti_tensor",
    "CSVMedical3DSegDataset",
]


def load_index_csv(path: Union[str, Path]) -> List[Dict[str, str]]:
    path = Path(path)
    rows: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for line in r:
            rows.append({k: (v or "").strip() for k, v in line.items() if k is not None})
    return rows


def unwrap_monai_dict_batch(batch: Any) -> Any:
    """MONAI RandCrop / collate may yield a length-1 ``list`` of dicts instead of a ``dict``."""
    if isinstance(batch, list):
        if len(batch) != 1:
            raise RuntimeError(
                f"Unexpected batch: expected length-1 list of dicts, got len={len(batch)}"
            )
        return batch[0]
    return batch


def apply_label_remap(
    label: "np.ndarray",
    num_classes: int,
    label_remap: Optional[Dict[str, Any]] = None,
) -> "np.ndarray":
    if not label_remap or not label_remap.get("enabled", False):
        return label
    m = label_remap.get("map")
    if not m:
        return label
    out = label.copy().astype(np.int64)
    for k, v in m.items():
        a, b = int(k), int(v)
        out[label == a] = b
    return out


def get_spacing_from_nifti(path: Union[str, Path]) -> Optional[Tuple[float, float, float]]:
    if nib is None:
        return None
    p = str(path)
    try:
        img = nib.load(p)  # type: ignore[union-attr]
        zooms = img.header.get_zooms()[:3]  # type: ignore[union-attr]
        return (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    except Exception:  # noqa: BLE001
        return None


def load_nifti_array(
    path: Union[str, Path], dtype: Optional[np.dtype] = None
) -> Tuple["np.ndarray", Any]:
    if nib is None:
        raise ImportError("nibabel is required for NIfTI loading. Install: pip install nibabel")
    path = str(path)
    nii = nib.load(path)  # type: ignore[union-attr]
    data = np.asanyarray(nii.get_fdata(), dtype=dtype)
    return data, nii.affine  # type: ignore[union-attr]


class _IdentityTransform:
    def __call__(self, d: Dict[str, Any]) -> Dict[str, Any]:
        return d


class CSVMedical3DSegDataset(Dataset):
    """
    One row per case; columns include ``image`` (path) and ``label`` (path) by default, or
    custom names from ``data.image_key`` / ``data.label_key``.

    If ``data.multi_channel_images`` is a list of column names, stack along C after load.
    """

    def __init__(
        self,
        index_csv: Union[str, Path],
        data_config: Optional[Dict[str, Any]] = None,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        self.rows = load_index_csv(index_csv)
        self.cfg: Dict = dict(data_config or {})
        self.image_key = self.cfg.get("image_key", "image")
        self.label_key = self.cfg.get("label_key", "label")
        self.multi: Optional[Sequence[str]] = self.cfg.get("multi_channel_columns") or self.cfg.get(
            "multi_channel_images"
        )
        self.num_classes = int(self.cfg.get("num_classes", 2))
        self.label_remap = self.cfg.get("label_remap")
        self.tfm = transform if transform is not None else _IdentityTransform()
        if not self.rows:
            raise FileNotFoundError(f"No rows in index: {index_csv}")

    def __len__(self) -> int:
        return len(self.rows)

    def _load_dict(self, row: Dict[str, str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"meta": row.copy()}
        if self.multi:
            chans = []
            for k in self.multi:
                p = (row.get(k) or row.get(k.upper()) or row.get(k.replace("_", "")) or "")
                p = str(p).strip()
                if not p:
                    raise KeyError(f"Missing path column {k!r} in row {row!r}")
                arr, _ = load_nifti_array(p)
                if arr.ndim == 3:
                    arr = arr[None, ...]  # 1,D,H,W
                chans.append(arr)
            img = np.concatenate(chans, axis=0)  # C,D,H,W
        else:
            p = row.get(self.image_key, row.get("image", ""))
            if not p:
                raise KeyError("No image path in row")
            img, _ = load_nifti_array(p)
            if img.ndim == 3:
                img = img[None, ...]
        label_p = row.get(self.label_key, row.get("label", ""))
        if not label_p:
            raise KeyError("No label path in row")
        lbl, _ = load_nifti_array(label_p, dtype=np.int64)
        if lbl.ndim == 4 and lbl.shape[0] == 1:
            lbl = lbl[0]
        if lbl.ndim == 3:
            pass
        else:
            lbl = np.squeeze(lbl)
        lbl = apply_label_remap(lbl, self.num_classes, self.label_remap)
        out[self.image_key] = img
        out[self.label_key] = lbl.astype(np.int64)
        return out

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        d = self._load_dict(self.rows[idx])
        return self.tfm(d)
