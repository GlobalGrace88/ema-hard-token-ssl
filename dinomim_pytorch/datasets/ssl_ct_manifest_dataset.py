"""
SSL pretraining from ``manifest_ssl_ct.csv`` produced by ``tools/datasets/prepare_public_ct_ssl.py``.

Returns multiview tensors compatible with ``ssl_volume_dataset.volume_multiview_collate_fn``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from dinomim_pytorch.datasets.medical_3d_segmentation_dataset import load_index_csv, load_nifti_array


def _parse_patch_size(data: Dict[str, Any]) -> Tuple[int, int, int]:
    roi = data.get("image_size") or data.get("roi_size") or (64, 128, 128)
    return (int(roi[0]), int(roi[1]), int(roi[2]))


def _resolve_manifest_path(data: Dict[str, Any]) -> Path:
    raw = data.get("manifest_csv") or data.get("manifest") or data.get("index_csv")
    if not raw:
        raise FileNotFoundError("Set data.manifest_csv to processed/manifest_ssl_ct.csv")
    p = Path(str(raw)).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"manifest_csv not found: {p}")
    return p


def _resolve_manifest_root(data: Dict[str, Any], manifest_path: Path) -> Path:
    """Dataset root for manifest-relative ``image_path`` entries (e.g. ``processed/images/...``)."""
    raw = data.get("manifest_root") or data.get("dataset_root") or data.get("root")
    if raw:
        return Path(str(raw)).expanduser().resolve()
    # Default: manifest lives in ``<root>/processed/manifest_ssl_ct.csv``.
    return manifest_path.parent.parent


def _resolve_image_path(data: Dict[str, Any], manifest_path: Path, raw_path: str) -> Path:
    p = Path(str(raw_path).strip())
    if p.is_file():
        return p.resolve()
    root = _resolve_manifest_root(data, manifest_path)
    candidate = (root / p).resolve()
    if candidate.is_file():
        return candidate
    alt = (manifest_path.parent / p).resolve()
    if alt.is_file():
        return alt
    return candidate


def _load_manifest_rows(data: Dict[str, Any]) -> List[Dict[str, str]]:
    p = _resolve_manifest_path(data)
    col = str(data.get("image_key", "image_path"))
    rows = load_index_csv(p)
    out: List[Dict[str, str]] = []
    for row in rows:
        raw = (row.get(col) or row.get("image_path") or row.get("path") or "").strip()
        pre = (row.get("preprocessed_path") or "").strip()
        if pre and Path(pre).is_file():
            resolved = Path(pre).resolve()
        elif not raw:
            continue
        else:
            resolved = _resolve_image_path(data, p, raw)
        if resolved.is_file():
            row = dict(row)
            row[col] = str(resolved)
            if "image_path" in row:
                row["image_path"] = str(resolved)
            if "path" in row:
                row["path"] = str(resolved)
            out.append(row)
    if not out:
        raise FileNotFoundError(f"No readable rows in manifest: {p}")
    return out


def _need_to_pad(shape: Tuple[int, int, int], patch_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return tuple(max(0, int(patch_size[i]) - int(shape[i])) for i in range(3))


def _rand_int(lb: int, ub: int) -> int:
    if ub < lb:
        return lb
    return random.randint(lb, ub)


def _random_bbox(
    shape: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    *,
    vol: Optional[np.ndarray] = None,
    fg_threshold: Optional[float] = None,
    oversample_foreground: float = 0.0,
) -> Tuple[int, int, int]:
    """Sample patch origin; allows negative starts when volume < patch_size (pad later)."""
    need = _need_to_pad(shape, patch_size)
    lb = [-(need[i] // 2) for i in range(3)]
    ub = [
        shape[i] + need[i] // 2 + need[i] % 2 - patch_size[i] for i in range(3)
    ]

    force_fg = (
        vol is not None
        and fg_threshold is not None
        and oversample_foreground > 0.0
        and random.random() < oversample_foreground
    )
    if force_fg:
        slab = vol[0] if vol.ndim == 4 else vol
        coords = np.argwhere(slab > float(fg_threshold))
        if coords.size > 0:
            z, y, x = coords[random.randrange(len(coords))]
            pd, ph, pw = patch_size
            z0 = int(np.clip(z - pd // 2, lb[0], ub[0]))
            y0 = int(np.clip(y - ph // 2, lb[1], ub[1]))
            x0 = int(np.clip(x - pw // 2, lb[2], ub[2]))
            return z0, y0, x0

    return (_rand_int(lb[0], ub[0]), _rand_int(lb[1], ub[1]), _rand_int(lb[2], ub[2]))


def _crop_pad_volume(
    vol: np.ndarray,
    z0: int,
    y0: int,
    x0: int,
    patch_size: Tuple[int, int, int],
    *,
    pad_mode: str = "edge",
) -> np.ndarray:
    """Crop a fixed-size patch, edge-padding when the source volume is smaller."""
    pd, ph, pw = patch_size
    z1, y1, x1 = z0 + pd, y0 + ph, x0 + pw
    if vol.ndim == 3:
        d, h, w = vol.shape
        sl = vol[max(0, z0) : min(d, z1), max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)]
        pad_w = (
            (max(0, -z0), max(0, z1 - d)),
            (max(0, -y0), max(0, y1 - h)),
            (max(0, -x0), max(0, x1 - w)),
        )
        if any(p[0] or p[1] for p in pad_w):
            sl = np.pad(sl, pad_w, mode=pad_mode)
        if sl.shape != (pd, ph, pw):
            raise ValueError(f"patch shape {sl.shape} != expected {(pd, ph, pw)}")
        return sl[None, ...]
    d, h, w = vol.shape[-3:]
    sl = vol[:, max(0, z0) : min(d, z1), max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)]
    pad_w = (
        (0, 0),
        (max(0, -z0), max(0, z1 - d)),
        (max(0, -y0), max(0, y1 - h)),
        (max(0, -x0), max(0, x1 - w)),
    )
    if any(p[0] or p[1] for p in pad_w):
        sl = np.pad(sl, pad_w, mode=pad_mode)
    if sl.shape[-3:] != (pd, ph, pw):
        raise ValueError(f"patch shape {sl.shape[-3:]} != expected {(pd, ph, pw)}")
    return sl


class SslCtManifestPatchDataset(Dataset):
    """Random patches from manifest volumes (inpainting / DINO SSL)."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config
        data = (config or {}).get("data") or {}
        self.rows = _load_manifest_rows(data)
        self.patch_size = _parse_patch_size(data)
        self.samples_per_epoch = int(
            data.get("samples_per_epoch", data.get("ssl_samples_per_epoch", 500))
        )
        self.oversample = float(data.get("oversample_foreground", 0.33))
        self.fg_threshold = data.get("foreground_intensity_threshold", 0.0)
        if self.fg_threshold is None:
            self.fg_threshold = 0.0
        else:
            self.fg_threshold = float(self.fg_threshold)
        self.image_key = str(data.get("image_key", "image_path"))
        self.normalize = str(data.get("volume_normalize", "none"))
        self.seed = int(data.get("seed", 42))
        self._rng = random.Random(self.seed)
        mcfg = (config or {}).get("model") or {}
        self.want_channels = int(data.get("channels", mcfg.get("in_channels", 1)))
        raw_cols = data.get("modality_columns") or data.get("modality_paths")
        if raw_cols:
            self.modality_columns = [str(c).strip() for c in raw_cols if str(c).strip()]
        else:
            self.modality_columns = []

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _load_volume_from_row(self, row: Dict[str, str]) -> np.ndarray:
        """Load single- or multi-channel patch volume from manifest row."""
        pre = (row.get("preprocessed_path") or "").strip()
        if pre and Path(pre).is_file() and self.want_channels != 4:
            arr, _ = load_nifti_array(pre)
            if arr.ndim == 4:
                return arr
            return arr[None, ...] if arr.ndim == 3 else arr
        if self.modality_columns and self.want_channels > 1:
            chans: List[np.ndarray] = []
            for col in self.modality_columns:
                p = (row.get(col) or "").strip()
                if not p:
                    continue
                arr, _ = load_nifti_array(p)
                if arr.ndim == 4:
                    arr = arr[0]
                chans.append(arr)
            if not chans:
                raise FileNotFoundError(f"No modality paths in row: {row}")
            if len(chans) != self.want_channels:
                print(
                    f"[ssl-manifest] warning: loaded {len(chans)} modalities "
                    f"but want_channels={self.want_channels} case={row.get('case_id')}",
                    flush=True,
                )
            patch = np.stack(chans[: self.want_channels], axis=0)
            return patch
        path = row.get(self.image_key) or row.get("image_path") or row.get("path")
        arr, _ = load_nifti_array(path)
        if arr.ndim == 4:
            arr = arr[0]
        return arr[None, ...] if arr.ndim == 3 else arr

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        from dinomim_pytorch.ssl_volume_dataset import _make_views_3d, _normalize_channels

        row = self.rows[idx % len(self.rows) if self.rows else 0]
        if len(self.rows) > 1:
            row = self.rows[self._rng.randrange(len(self.rows))]
        arr = self._load_volume_from_row(row)
        shape = arr.shape[-3:]
        z0, y0, x0 = _random_bbox(
            shape,
            self.patch_size,
            vol=arr[0] if arr.ndim == 4 else arr,
            fg_threshold=self.fg_threshold,
            oversample_foreground=self.oversample,
        )
        patch = _crop_pad_volume(arr, z0, y0, x0, self.patch_size)
        vol = torch.from_numpy(np.ascontiguousarray(patch, dtype=np.float32))
        want_c = self.want_channels
        if vol.shape[0] != want_c:
            if want_c == 1 and vol.shape[0] > 1:
                vol = vol[:1]
            else:
                raise ValueError(f"channels={vol.shape[0]} vs want_channels={want_c}")
        vol = _normalize_channels(vol, self.normalize)
        tg, sg, sl, meta = _make_views_3d(vol, self.cfg)
        meta["patch_sampler"] = True
        meta["manifest_path"] = str(row.get(self.image_key) or row.get("case_id"))
        meta["modalities"] = self.modality_columns or ["image_path"]
        return {
            "teacher_glob": tg,
            "student_glob": sg,
            "student_loc": sl,
            "volume": vol,
            "path": str(row.get("case_id", "")),
            "mask_meta": meta,
        }


class SslCtManifestVolumeDataset(Dataset):
    """One manifest volume per index (full-volume SSL)."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config
        data = (config or {}).get("data") or {}
        self.rows = _load_manifest_rows(data)
        self.image_key = str(data.get("image_key", "image_path"))
        self.normalize = str(data.get("volume_normalize", "none"))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        from dinomim_pytorch.ssl_volume_dataset import _make_views_3d, _normalize_channels

        row = self.rows[idx % len(self.rows)]
        path = row.get(self.image_key) or row.get("image_path") or row.get("path")
        arr, _ = load_nifti_array(path)
        if arr.ndim == 3:
            vol = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).unsqueeze(0)
        elif arr.ndim == 4:
            vol = torch.from_numpy(np.ascontiguousarray(arr[0:1], dtype=np.float32))
        else:
            raise ValueError(f"Expected 3D/4D volume, got {arr.shape}")
        want_c = int((self.cfg.get("model") or {}).get("in_channels", vol.shape[0]))
        if vol.shape[0] != want_c:
            vol = vol[:want_c]
        vol = _normalize_channels(vol, self.normalize)
        tg, sg, sl, meta = _make_views_3d(vol, self.cfg)
        return {
            "teacher_glob": tg,
            "student_glob": sg,
            "student_loc": sl,
            "volume": vol,
            "path": str(path),
            "mask_meta": meta,
        }


def has_ssl_ct_manifest_data(cfg: Dict[str, Any]) -> bool:
    data = (cfg or {}).get("data") or {}
    try:
        _load_manifest_rows(data)
        return True
    except FileNotFoundError:
        return False


__all__ = [
    "SslCtManifestPatchDataset",
    "SslCtManifestVolumeDataset",
    "has_ssl_ct_manifest_data",
]
