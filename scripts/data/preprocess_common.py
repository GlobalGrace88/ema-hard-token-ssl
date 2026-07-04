"""Shared helpers for dataset preprocessing scripts (no file moves/deletes on raw data)."""

from __future__ import annotations

import csv
import json
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

try:
    import nibabel as nib
except ImportError:
    nib = None  # type: ignore[assignment]

try:
    from scipy import ndimage
except ImportError:
    ndimage = None  # type: ignore[assignment]


def load_preprocess_config(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_root() -> Path:
    try:
        from dinomim_pytorch.paths import repo_root as _root

        return _root()
    except ImportError:
        return Path(__file__).resolve().parents[2]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def spacing_from_nifti(path: Path) -> str:
    if nib is None or not path.is_file():
        return ""
    try:
        img = nib.load(str(path))
        z = img.header.get_zooms()[:3]
        return ",".join(f"{float(x):.4f}" for x in z)
    except Exception:
        return ""


def load_nifti(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if nib is None:
        raise ImportError("nibabel required: pip install nibabel")
    img = nib.load(str(path))
    data = np.asanyarray(img.get_fdata(), dtype=np.float32)
    return data, np.asarray(img.affine, dtype=np.float64)


def save_nifti(path: Path, data: np.ndarray, affine: np.ndarray, dtype: str = "float32") -> None:
    if nib is None:
        raise ImportError("nibabel required")
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=dtype), affine), str(path))


def save_nnformer_npz(path: Path, image: np.ndarray, label: np.ndarray) -> None:
    """nnFormer format: ``data`` array shape ``(C_img + 1, D, H, W)``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 3:
        image = image[np.newaxis, ...]
    label = np.asarray(label, dtype=np.float32)
    if label.ndim == 4:
        label = label[0]
    stacked = np.concatenate([image.astype(np.float32), label[np.newaxis, ...]], axis=0)
    np.savez_compressed(str(path), data=stacked)


def clip_normalize_ct(vol: np.ndarray, lo: float = -175.0, hi: float = 250.0) -> np.ndarray:
    out = np.clip(vol, lo, hi)
    mean = float(out.mean())
    std = float(out.std())
    if std < 1e-6:
        return out - mean
    return (out - mean) / std


def zscore_per_channel(vol: np.ndarray) -> np.ndarray:
    if vol.ndim == 3:
        m, s = float(vol.mean()), float(vol.std())
        return (vol - m) / (s + 1e-8)
    out = vol.copy()
    for c in range(out.shape[0]):
        m, s = float(out[c].mean()), float(out[c].std())
        out[c] = (out[c] - m) / (s + 1e-8)
    return out


def resample_to_shape(
    vol: np.ndarray,
    target: Tuple[int, int, int],
    is_label: bool = False,
) -> np.ndarray:
    if vol.shape[-3:] == target:
        return vol
    if ndimage is None:
        return vol
    if vol.ndim == 3:
        src = vol.shape
        factors = tuple(t / s for t, s in zip(target, src))
        order = 0 if is_label else 1
        return ndimage.zoom(vol, factors, order=order)
    out_ch = []
    for c in range(vol.shape[0]):
        out_ch.append(resample_to_shape(vol[c], target, is_label=is_label))
    return np.stack(out_ch, axis=0)


def deterministic_split(ids: List[str], val_frac: float, seed: int) -> Tuple[List[str], List[str]]:
    ids = sorted(set(ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_frac))) if len(ids) > 1 else 0
    val = ids[:n_val]
    train = ids[n_val:] if n_val else ids
    if not train:
        train, val = ids, []
    return train, val


def write_splits_pkl(path: Path, train_ids: List[str], val_ids: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    splits = [{"train": train_ids, "val": val_ids}]
    with path.open("wb") as f:
        pickle.dump(splits, f)


def list_npz_in_dir(d: Path) -> List[Path]:
    if not d.is_dir():
        return []
    return sorted(d.glob("*.npz"))


def case_id_from_npz(p: Path) -> str:
    return p.stem


def symlink_or_copy(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        import shutil

        shutil.copy2(src, dst)
    return dst
