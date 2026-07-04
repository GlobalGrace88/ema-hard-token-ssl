"""
nnFormer / UNETR++ preprocessed ``*.npz`` volumes (``data`` array from ``GenericPreprocessor``).

Each file stores ``np.load(path)['data']`` with shape ``(C_img + 1, D, H, W)``; the last channel
is the segmentation label. Image channels are ``[:-1]`` and are already intensity-normalized
by the nnFormer pipeline.
"""

from __future__ import annotations

import csv
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch

from dinomim_pytorch.datasets.nnformer_splits import apply_finetune_patient_cap

_STAGE_PATTERNS = {
    "stage0": re.compile(r"nnFormerData_plans_.*_stage0$", re.I),
    "stage1": re.compile(r"nnFormerData_plans_.*_stage1$", re.I),
    "2d_stage0": re.compile(r"nnFormerData_plans_.*_2D_stage0$", re.I),
    "3d_fullres": re.compile(r"nnFormerData_plans_.*3d_fullres", re.I),
}


def load_nnformer_npz_volume(path: Union[str, Path]) -> np.ndarray:
    """Raw ``data`` array ``(C_img + 1, D, H, W)``."""
    path = Path(path)
    arr = np.load(str(path))["data"]
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D npz data at {path}, got shape {arr.shape}")
    return arr


def load_nnformer_npz_image(path: Union[str, Path]) -> torch.Tensor:
    """Image channels only, float32 ``[C, D, H, W]``."""
    arr = load_nnformer_npz_volume(path)
    img = np.ascontiguousarray(arr[:-1], dtype=np.float32)
    if img.shape[0] < 1:
        raise ValueError(f"No image channels in {path} (shape {arr.shape})")
    return torch.from_numpy(img)


def load_nnformer_npz_image_label(
    path: Union[str, Path],
    data_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Image ``[C,D,H,W]`` float32 and label ``[D,H,W]`` int64.

    Maps nnFormer ignore label (default ``-1``) to background when
    ``nnformer_map_ignore_label`` is true (default).
    """
    arr = load_nnformer_npz_volume(path)
    img = np.ascontiguousarray(arr[:-1], dtype=np.float32)
    lbl = np.ascontiguousarray(arr[-1], dtype=np.int64)
    cfg = data_cfg or {}
    if bool(cfg.get("nnformer_map_ignore_label", True)):
        ignore = int(cfg.get("nnformer_ignore_label", -1))
        lbl[lbl == ignore] = 0
    return img, lbl


def case_id_from_npz(path: Union[str, Path]) -> str:
    return Path(path).stem


def _case_id_from_csv_row(row: Dict[str, str]) -> Optional[str]:
    for key in ("case_id", "id", "case", "patient", "subject"):
        v = (row.get(key) or "").strip()
        if v:
            return v
    for key in ("path", "image", "image_path", "label_path", "label"):
        v = (row.get(key) or "").strip()
        if v:
            stem = Path(v).name
            for suf in (".nii.gz", ".nii", ".npz"):
                if stem.endswith(suf):
                    stem = stem[: -len(suf)]
                    break
            return stem
    return None


def load_case_ids_from_csv(path: Union[str, Path]) -> List[str]:
    """Case ids from a CSV (``case_id`` column or stems of path columns)."""
    path = Path(path)
    if not path.is_file():
        return []
    ids: List[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = _case_id_from_csv_row({k: (v or "").strip() for k, v in row.items() if k})
            if cid:
                ids.append(cid)
    return ids


def filter_npz_by_case_ids(
    paths: Sequence[Path], case_ids: Sequence[str]
) -> List[Path]:
    want: Set[str] = {str(c).strip() for c in case_ids if str(c).strip()}
    if not want:
        return []
    out = [p for p in paths if case_id_from_npz(p) in want]
    return sorted(out)


def _cap_train_npz_paths(paths: List[Path], data: Dict[str, Any], train: bool) -> List[Path]:
    if not train or not paths:
        return paths
    cap_ids = apply_finetune_patient_cap([case_id_from_npz(p) for p in paths], data)
    want = set(cap_ids)
    return sorted([p for p in paths if case_id_from_npz(p) in want], key=case_id_from_npz)


def split_npz_paths_train_val(
    all_paths: List[Path],
    data: Dict[str, Any],
    *,
    train: bool,
) -> List[Path]:
    """
  Pick train or val ``*.npz`` paths.

  Priority for validation cases:
  1. ``nnformer_npz_val_dir`` (all npz in that folder are val)
  2. ``nnformer_val_cases`` / ``index_val`` / ``csv_val`` case-id list
  3. ``val_frac`` + ``val_seed`` deterministic hold-out
  """
    data = dict(data or {})
    all_paths = sorted(all_paths, key=lambda p: case_id_from_npz(p))
    if not all_paths:
        return []

    val_dir = data.get("nnformer_npz_val_dir")
    if val_dir:
        val_paths = list_npz_files(val_dir)
        val_ids = {case_id_from_npz(p) for p in val_paths}
        if train:
            return _cap_train_npz_paths(
                [p for p in all_paths if case_id_from_npz(p) not in val_ids], data, train
            )
        return filter_npz_by_case_ids(all_paths, val_ids) or val_paths

    explicit_val = data.get("nnformer_val_cases")
    if explicit_val:
        val_ids = [str(x) for x in explicit_val]
    else:
        val_ids = []
        for key in ("index_val", "csv_val", "index_test", "csv_test"):
            raw = data.get(key)
            if raw and Path(str(raw)).is_file():
                val_ids = load_case_ids_from_csv(raw)
                break

    if val_ids:
        val_set = set(val_ids)
        if train:
            train_cases = data.get("nnformer_train_cases")
            if train_cases:
                want = set(str(x) for x in train_cases)
                return _cap_train_npz_paths(filter_npz_by_case_ids(all_paths, want), data, train)
            for key in ("index_csv",):
                raw = data.get(key)
                if raw and Path(str(raw)).is_file():
                    tr_ids = load_case_ids_from_csv(raw)
                    if tr_ids:
                        return _cap_train_npz_paths(
                            filter_npz_by_case_ids(all_paths, tr_ids), data, train
                        )
            return _cap_train_npz_paths(
                [p for p in all_paths if case_id_from_npz(p) not in val_set], data, train
            )
        return filter_npz_by_case_ids(all_paths, val_ids)

    val_frac = float(data.get("val_frac", 0.0) or 0.0)
    if val_frac > 0.0:
        rng = random.Random(int(data.get("val_seed", 42)))
        ids = list(all_paths)
        rng.shuffle(ids)
        n_val = max(1, int(round(len(ids) * val_frac)))
        val_paths = sorted(ids[:n_val], key=lambda p: case_id_from_npz(p))
        val_set = {case_id_from_npz(p) for p in val_paths}
        if train:
            return _cap_train_npz_paths(
                [p for p in all_paths if case_id_from_npz(p) not in val_set], data, train
            )
        return val_paths

    if train:
        for key in ("index_csv",):
            raw = data.get(key)
            if raw and Path(str(raw)).is_file():
                tr_ids = load_case_ids_from_csv(raw)
                if tr_ids:
                    return _cap_train_npz_paths(
                        filter_npz_by_case_ids(all_paths, tr_ids), data, train
                    )
        return _cap_train_npz_paths(all_paths, data, train)
    return all_paths


def list_npz_files(folder: Union[str, Path]) -> List[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.glob("*.npz") if p.is_file())


def _pick_npz_subdir(
    root: Path,
    stage: str,
    prefer_3d: bool,
) -> Optional[Path]:
    candidates = sorted(
        d for d in root.iterdir() if d.is_dir() and d.name.startswith("nnFormerData_plans")
    )
    if not candidates:
        return None

    stage_key = (stage or "auto").lower().replace("-", "_")
    if stage_key not in ("auto", ""):
        pat = _STAGE_PATTERNS.get(stage_key)
        if pat is not None:
            matched = [d for d in candidates if pat.search(d.name)]
            if matched:
                if prefer_3d:
                    non_2d = [d for d in matched if "2D" not in d.name and "2d" not in d.name]
                    if non_2d:
                        return non_2d[0]
                return matched[0]

    if prefer_3d:
        for d in candidates:
            if d.name.endswith("_stage0") and "2D" not in d.name:
                return d
        for d in candidates:
            if "2D" not in d.name and "2d" not in d.name:
                return d
    return candidates[0]


def resolve_nnformer_npz_dir(data: Dict[str, Any]) -> Optional[Path]:
    """
    Resolve folder containing ``*.npz`` from ``data`` config.

    Keys (first match wins):
    - ``nnformer_npz_dir`` / ``preprocessed_npz_dir``: explicit folder of ``*.npz``
    - ``nnformer_preprocessed_dir`` / ``preprocessed_dir``: task root; pick ``nnFormerData_plans*``
    - ``nnformer_npz_stage``: ``auto`` | ``stage0`` | ``stage1`` | ``2d_stage0`` | ``3d_fullres``
    - ``nnformer_npz_prefer_3d``: when auto-picking, prefer non-2D stage folders (default true)
    """
    if not data:
        return None

    for key in ("nnformer_npz_dir", "preprocessed_npz_dir"):
        raw = data.get(key)
        if raw:
            p = Path(str(raw)).expanduser()
            if p.is_dir() and list_npz_files(p):
                return p
            if p.is_dir():
                return p

    for key in ("nnformer_preprocessed_dir", "preprocessed_dir"):
        raw = data.get(key)
        if not raw:
            continue
        root = Path(str(raw)).expanduser()
        if not root.is_dir():
            continue
        direct = list_npz_files(root)
        if direct:
            return root
        stage = str(data.get("nnformer_npz_stage", "auto"))
        prefer_3d = bool(data.get("nnformer_npz_prefer_3d", True))
        sub = _pick_npz_subdir(root, stage, prefer_3d)
        if sub is not None:
            return sub

    return None


def has_nnformer_npz_data(cfg: Dict[str, Any]) -> bool:
    data = (cfg or {}).get("data") or {}
    folder = resolve_nnformer_npz_dir(data)
    return folder is not None and len(list_npz_files(folder)) > 0


__all__ = [
    "load_nnformer_npz_volume",
    "load_nnformer_npz_image",
    "load_nnformer_npz_image_label",
    "case_id_from_npz",
    "load_case_ids_from_csv",
    "filter_npz_by_case_ids",
    "split_npz_paths_train_val",
    "list_npz_files",
    "resolve_nnformer_npz_dir",
    "has_nnformer_npz_data",
]
