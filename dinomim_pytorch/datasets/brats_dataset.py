"""
BraTS-style multimodal MRI: stack T1, T1CE, T2, FLAIR as ``image`` with shape [C, D, H, W].
Index CSV: columns ``t1``, ``t1ce``, ``t2``, ``flair``, ``label`` (paths).
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dinomim_pytorch.datasets.medical_3d_segmentation_dataset import (
    CSVMedical3DSegDataset,
    load_index_csv,
)
from dinomim_pytorch.datasets.monai_transforms3d import build_brats_compose

__all__ = [
    "BRaTSDataset",
    "adapt_brats_csv_layout",
    "brats_modality_column_names",
    "build_brats_row_transforms",
    "brats_data_config_from_user_config",
]


def brats_modality_column_names(data_cfg: Dict[str, Any]) -> List[str]:
    m = (data_cfg or {}).get("modalities", ["t1", "t1ce", "t2", "flair"])
    return [str(x).lower() for x in m]


def brats_data_config_from_user_config(data_cfg: Dict[str, Any]) -> Dict[str, Any]:
    d = deepcopy(data_cfg) if data_cfg else {}
    d["image_key"] = d.get("image_key", "image")
    d["label_key"] = d.get("label_key", "label")
    d["multi_channel_columns"] = brats_modality_column_names(d)
    d["input_type"] = d.get("input_type", "volume_3d")
    d["dataset_name"] = d.get("dataset_name", "brats")
    return d


def adapt_brats_csv_layout(
    index_csv: Union[str, Path],
    data_cfg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Align BraTS ``data`` config with the CSV layout (multimodal columns vs single ``path`` + ``label_path``).
    """
    dcfg = brats_data_config_from_user_config(data_cfg or {})
    try:
        rows = load_index_csv(index_csv)
    except OSError:
        return dcfg
    if not rows:
        return dcfg
    r0 = rows[0]
    mods = brats_modality_column_names(dcfg)
    has_modal_paths = any(str(r0.get(m, "") or "").strip() for m in mods)
    path_ok = bool(str(r0.get("path") or "").strip())
    img_ok = bool(str(r0.get("image") or "").strip())
    if has_modal_paths:
        return dcfg
    if path_ok or img_ok:
        dcfg = dict(dcfg)
        dcfg["stack_single_multichannel_path"] = True
        dcfg.pop("multi_channel_columns", None)
        dcfg.pop("multi_channel_images", None)
        dcfg["image_key"] = "path" if path_ok else "image"
        if str(r0.get("label_path") or "").strip():
            dcfg["label_key"] = "label_path"
        else:
            dcfg["label_key"] = "label"
    return dcfg


def build_brats_row_transforms(
    data_cfg: Dict[str, Any], train: bool = True, keys: Optional[Dict[str, str]] = None
) -> Any:
    return build_brats_compose(data_cfg, train=train, keys=keys)


class BRaTSDataset(CSVMedical3DSegDataset):
    """BraTS index: one row with paths per modality; stacked to ``image`` in parent."""

    def __init__(
        self,
        index_csv: str,
        data_config: Optional[Dict[str, Any]] = None,
        transform: Optional[Any] = None,
    ) -> None:
        dcfg = adapt_brats_csv_layout(index_csv, data_config or {})
        super().__init__(index_csv, dcfg, transform=transform)
