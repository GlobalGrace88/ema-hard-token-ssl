"""
BTCV-style abdominal CT: single channel ``image`` [1, D, H, W], multi-class organ ``label``.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from dinomim_pytorch.datasets.medical_3d_segmentation_dataset import CSVMedical3DSegDataset
from dinomim_pytorch.datasets.monai_transforms3d import build_btcv_compose

__all__ = ["BTCVDataset", "btcv_data_config", "build_btcv_transforms"]


def btcv_data_config(data_cfg: Dict[str, Any]) -> Dict[str, Any]:
    d = deepcopy(data_cfg) if data_cfg else {}
    d["image_key"] = d.get("image_key", "image")
    d["label_key"] = d.get("label_key", "label")
    d["multi_channel_columns"] = None
    d["channels"] = int(d.get("channels", 1))
    d["input_type"] = d.get("input_type", "volume_3d")
    d["dataset_name"] = d.get("dataset_name", "btcv")
    return d


def build_btcv_transforms(
    data_cfg: Dict[str, Any], train: bool = True, keys: Optional[Dict[str, str]] = None
) -> Any:
    return build_btcv_compose(data_cfg, train=train, keys=keys)


class BTCVDataset(CSVMedical3DSegDataset):
    """Index CSV: ``image``, ``label`` columns to NIfTI paths."""

    def __init__(
        self,
        index_csv: str,
        data_config: Optional[Dict[str, Any]] = None,
        transform: Optional[Any] = None,
    ) -> None:
        dcfg = btcv_data_config(data_config or {})
        super().__init__(index_csv, dcfg, transform=transform)
