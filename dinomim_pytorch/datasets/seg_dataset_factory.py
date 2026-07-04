"""Build 3D segmentation datasets for finetune / eval (NIfTI CSV or nnFormer npz)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from dinomim_pytorch.datasets.brats_dataset import (
    BRaTSDataset,
    adapt_brats_csv_layout,
    build_brats_row_transforms,
)
from dinomim_pytorch.datasets.btcv_dataset import BTCVDataset, build_btcv_transforms
from dinomim_pytorch.datasets.medical_3d_segmentation_dataset import CSVMedical3DSegDataset
from dinomim_pytorch.datasets.nnformer_npz import (
    filter_npz_by_case_ids,
    has_nnformer_npz_data,
    list_npz_files,
    load_case_ids_from_csv,
    resolve_nnformer_npz_dir,
)
from dinomim_pytorch.datasets.nnformer_npz_seg_dataset import NnformerNpzSegDataset
from dinomim_pytorch.datasets.nnformer_npz_patch import (
    NnformerNpzPatchSegDataset,
    patch_sampler_enabled,
)
from dinomim_pytorch.datasets.monai_transforms3d import (
    build_nnformer_npz_compose,
    build_nnformer_npz_eval_fullvolume_compose,
)


def get_seg_loader_kind(cfg: Dict[str, Any]) -> str:
    data = (cfg or {}).get("data") or {}
    kind = str(data.get("loader", "nifti_csv")).lower().replace("-", "_")
    if kind in ("nnformer", "nnformer_npz", "npz", "nnformer_preprocessed"):
        return "nnformer_npz"
    return "nifti_csv"


def has_segmentation_data(
    cfg: Dict[str, Any],
    *,
    train: bool,
    index_csv_override: Optional[str] = None,
) -> bool:
    if get_seg_loader_kind(cfg) == "nnformer_npz":
        return has_nnformer_npz_data(cfg)
    d = dict((cfg or {}).get("data") or {})
    p = index_csv_override
    if not p:
        p = d.get("index_csv") if train else d.get("index_val") or d.get("csv_val") or d.get("index_csv")
    return bool(p and Path(str(p)).is_file())


def _npz_paths_for_eval_override(
    data: Dict[str, Any], index_csv_override: Optional[str]
) -> Optional[list]:
    if not index_csv_override or not Path(str(index_csv_override)).is_file():
        return None
    folder = resolve_nnformer_npz_dir(data)
    if folder is None:
        return None
    case_ids = load_case_ids_from_csv(index_csv_override)
    if not case_ids:
        return None
    return filter_npz_by_case_ids(list_npz_files(folder), case_ids)


def build_segmentation_dataset(
    cfg: Dict[str, Any],
    *,
    train: bool,
    index_csv_override: Optional[str] = None,
) -> Optional[Any]:
    """
    Returns a Dataset or ``None`` when no data source is configured.

    ``data.loader``:
    - ``nifti_csv`` (default): ``index_csv`` / ``index_val`` NIfTI path CSV
    - ``nnformer_npz``: preprocessed ``*.npz`` folder(s)
    """
    data = dict((cfg or {}).get("data") or {})
    name = (data.get("dataset_name") or "").lower()

    if get_seg_loader_kind(cfg) == "nnformer_npz":
        if patch_sampler_enabled(data) and index_csv_override is None:
            return NnformerNpzPatchSegDataset(data, train=train)
        tfm = build_nnformer_npz_compose(data, train=train)
        npz_override = _npz_paths_for_eval_override(data, index_csv_override)
        return NnformerNpzSegDataset(
            data,
            train=train,
            transform=tfm,
            npz_paths=npz_override,
        )

    p = index_csv_override
    if not p:
        if train:
            p = data.get("index_csv")
        else:
            p = data.get("index_val") or data.get("csv_val") or data.get("index_csv")
    if not p or not Path(str(p)).is_file():
        return None

    if name == "brats":
        adapted = adapt_brats_csv_layout(str(p), data)
        tfm = build_brats_row_transforms(adapted, train=train)
        return BRaTSDataset(str(p), data, transform=tfm)
    if name in ("btcv", "bmcv", "synapse"):
        tfm = build_btcv_transforms(data, train=train)
        return BTCVDataset(str(p), data, transform=tfm)
    tfm = build_brats_row_transforms(data, train=train)
    return CSVMedical3DSegDataset(str(p), data, transform=tfm)


def build_eval_fullvolume_transform(cfg: Dict[str, Any]) -> Any:
    data = dict((cfg or {}).get("data") or {})
    name = str(data.get("dataset_name") or "").lower()
    if get_seg_loader_kind(cfg) == "nnformer_npz":
        return build_nnformer_npz_eval_fullvolume_compose(data)
    from dinomim_pytorch.datasets.monai_transforms3d import build_eval_fullvolume_compose

    return build_eval_fullvolume_compose(data, name)


__all__ = [
    "get_seg_loader_kind",
    "has_segmentation_data",
    "build_segmentation_dataset",
    "build_eval_fullvolume_transform",
]
