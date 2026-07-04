"""
Random-patch nnFormer / UNETR++ ``.npz`` loader (MAE_BYOL ``unetrpp_npz`` parity).

Uses ``class_locations`` in per-case ``.pkl`` for foreground-biased crops during training.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from dinomim_pytorch.datasets.nnformer_npz import resolve_nnformer_npz_dir
from dinomim_pytorch.datasets.nnformer_splits import resolve_npz_case_ids_for_split


def list_npz_case_ids(folder: Path) -> List[str]:
    ids = sorted(
        p.stem for p in folder.glob("*.npz") if p.is_file() and "segFromPrevStage" not in p.name
    )
    if not ids:
        raise FileNotFoundError(f"No case .npz files in {folder}")
    return ids


def _load_case_array(npz_path: Path) -> np.ndarray:
    npy_path = npz_path.with_suffix(".npy")
    if npy_path.is_file():
        return np.load(str(npy_path))
    with np.load(str(npz_path)) as z:
        return np.asarray(z["data"])


def _load_properties(pkl_path: Path) -> dict:
    if not pkl_path.is_file():
        return {}
    import pickle

    with open(pkl_path, "rb") as fh:
        return pickle.load(fh)


def _adjust_need_to_pad(
    shape: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    need_to_pad: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    out = list(need_to_pad)
    for d in range(3):
        if out[d] + shape[d] < patch_size[d]:
            out[d] = patch_size[d] - shape[d]
    return tuple(out)


def _rand_lb(lb: int, ub: int) -> int:
    if ub < lb:
        return lb
    return random.randint(lb, ub)


def _random_bbox(
    shape: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    need_to_pad: Tuple[int, int, int],
    force_fg: bool,
    properties: dict,
) -> Tuple[int, int, int]:
    pd, ph, pw = patch_size
    need_to_pad = _adjust_need_to_pad(shape, patch_size, need_to_pad)
    lb = [-(need_to_pad[i] // 2) for i in range(3)]
    ub = [
        shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - patch_size[i] for i in range(3)
    ]
    if not force_fg:
        return (_rand_lb(lb[0], ub[0]), _rand_lb(lb[1], ub[1]), _rand_lb(lb[2], ub[2]))
    locs = properties.get("class_locations", {})
    fg_classes = [int(k) for k in locs.keys() if int(k) > 0 and len(locs[k]) > 0]
    if not fg_classes:
        return (_rand_lb(lb[0], ub[0]), _rand_lb(lb[1], ub[1]), _rand_lb(lb[2], ub[2]))
    cls = random.choice(fg_classes)
    vox = locs[cls][random.randrange(len(locs[cls]))]
    z0 = max(lb[0], int(vox[0]) - pd // 2)
    y0 = max(lb[1], int(vox[1]) - ph // 2)
    x0 = max(lb[2], int(vox[2]) - pw // 2)
    return (min(z0, ub[0]), min(y0, ub[1]), min(x0, ub[2]))


def _crop_pad_volume(
    case_all_data: np.ndarray,
    bbox_lb: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    pad_mode: str = "edge",
) -> Tuple[np.ndarray, np.ndarray]:
    pd, ph, pw = patch_size
    z0, y0, x0 = bbox_lb
    z1, y1, x1 = z0 + pd, y0 + ph, x0 + pw
    shape = case_all_data.shape[1:]
    valid = (
        max(0, z0),
        min(shape[0], z1),
        max(0, y0),
        min(shape[1], y1),
        max(0, x0),
        min(shape[2], x1),
    )
    sl = case_all_data[:, valid[0] : valid[1], valid[2] : valid[3], valid[4] : valid[5]]
    pad_w = [
        (0, 0),
        (-min(0, z0), max(z1 - shape[0], 0)),
        (-min(0, y0), max(y1 - shape[1], 0)),
        (-min(0, x0), max(x1 - shape[2], 0)),
    ]
    if pad_mode == "edge":
        data = np.pad(sl[:-1], pad_w, mode="edge")
        seg = np.pad(sl[-1:], pad_w, mode="constant", constant_values=-1)[0]
    else:
        data = np.pad(sl[:-1], pad_w, mode="constant", constant_values=0)
        seg = np.pad(sl[-1:], pad_w, mode="constant", constant_values=-1)[0]
    return data.astype(np.float32, copy=False), seg.astype(np.int64, copy=False)


class _NpzPatchSamplerDataset(Dataset):
    """Inner sampler: ``(C,D,H,W)`` image + ``(D,H,W)`` label tensors."""

    def __init__(
        self,
        npz_dir: Path,
        case_ids: Sequence[str],
        patch_size: Tuple[int, int, int],
        num_classes: int,
        *,
        oversample_foreground: float,
        samples_per_epoch: int,
        seed: int,
    ) -> None:
        self.npz_dir = Path(npz_dir)
        self.case_ids = list(case_ids)
        self.patch_size = tuple(int(x) for x in patch_size)
        self.num_classes = int(num_classes)
        self.oversample_foreground = float(oversample_foreground)
        self.samples_per_epoch = int(samples_per_epoch)
        self._rng = random.Random(int(seed))
        self.need_to_pad = (0, 0, 0)
        missing = [c for c in self.case_ids if not (self.npz_dir / f"{c}.npz").is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} npz under {self.npz_dir} (e.g. {missing[0]}.npz)"
            )

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        del idx
        force_fg = self._rng.random() < self.oversample_foreground
        case_id = self._rng.choice(self.case_ids)
        props = _load_properties(self.npz_dir / f"{case_id}.pkl")
        case_all = _load_case_array(self.npz_dir / f"{case_id}.npz")
        shape = tuple(case_all.shape[1:])
        bbox = _random_bbox(shape, self.patch_size, self.need_to_pad, force_fg, props)
        data, seg = _crop_pad_volume(case_all, bbox, self.patch_size)
        seg = np.clip(seg, 0, self.num_classes - 1)
        seg[seg < 0] = 0
        return torch.from_numpy(data), torch.from_numpy(seg)


class NnformerNpzPatchSegDataset(Dataset):
    """
    Dict batches ``{image_key, label_key}`` for ``finetune_mri_segmentation.py``.

    Enable with ``data.patch_sampler: true`` (uses ``splits_final.pkl`` when set).
    """

    def __init__(self, data_config: Dict[str, Any], *, train: bool = True) -> None:
        from copy import deepcopy

        self.cfg: Dict[str, Any] = deepcopy(data_config or {})
        self.train = train
        self.image_key = str(self.cfg.get("image_key", "image"))
        self.label_key = str(self.cfg.get("label_key", "label"))
        self.num_classes = int(self.cfg.get("num_classes", 2))

        self.npz_dir = resolve_nnformer_npz_dir(self.cfg)
        if self.npz_dir is None:
            raise FileNotFoundError(
                "patch_sampler: set nnformer_npz_dir or nnformer_preprocessed_dir"
            )
        self._inner = build_npz_patch_sampler(self.cfg, train=train)
        self.n_cases = len(self._inner.case_ids)

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img, lbl = self._inner[idx]
        return {self.image_key: img, self.label_key: lbl}


def patch_sampler_enabled(data_cfg: Dict[str, Any]) -> bool:
    return bool(data_cfg.get("patch_sampler") or data_cfg.get("unetrpp_patch_sampler"))


def build_npz_patch_sampler(
    data_cfg: Dict[str, Any],
    *,
    train: bool,
) -> _NpzPatchSamplerDataset:
    """Shared random-patch sampler for seg finetune and DINO SSL pretrain."""
    folder = resolve_nnformer_npz_dir(data_cfg)
    if folder is None:
        raise FileNotFoundError(
            "patch_sampler: set nnformer_npz_dir or nnformer_preprocessed_dir"
        )
    roi = data_cfg.get("image_size") or data_cfg.get("roi_size") or (64, 128, 128)
    patch_size = (int(roi[0]), int(roi[1]), int(roi[2]))
    num_classes = int(data_cfg.get("num_classes", 14))
    case_ids = resolve_npz_case_ids_for_split(data_cfg, train=train, npz_dir=folder)
    seed = int(data_cfg.get("seed", data_cfg.get("val_seed", 42)))
    if train:
        spe = int(data_cfg.get("samples_per_epoch", data_cfg.get("ssl_samples_per_epoch", 500)))
        oversample = float(data_cfg.get("oversample_foreground", 0.33))
        ds_seed = seed
    else:
        spe = int(data_cfg.get("val_samples_per_epoch", max(50, 500 // 4)))
        oversample = 0.0
        ds_seed = seed + 1
    return _NpzPatchSamplerDataset(
        folder,
        case_ids,
        patch_size,
        num_classes,
        oversample_foreground=oversample,
        samples_per_epoch=spe,
        seed=ds_seed,
    )


__all__ = [
    "NnformerNpzPatchSegDataset",
    "patch_sampler_enabled",
    "build_npz_patch_sampler",
    "list_npz_case_ids",
]
