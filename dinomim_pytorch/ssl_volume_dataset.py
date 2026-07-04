"""
3D NIfTI volume loading and multi-view generation for DINO SSL (no labels).

Per-branch ``spatial_size: [D,H,W]`` (or legacy ``output_size`` cube) sets the voxel grid after
crop+resize; training interpolates to ``model.spatial_size`` for the backbone. Optional
``student.local.nested_in_student_global`` constrains local crops inside the first student-global box.
Optional ``student.global_local_3d`` uses fixed native global/local window sizes (nested by default),
then resizes to each branch grid (BYOL-style 3D global–local).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from dinomim_pytorch.datasets.medical_3d_segmentation_dataset import load_nifti_array
from dinomim_pytorch.datasets.nnformer_npz import (
    has_nnformer_npz_data,
    list_npz_files,
    load_nnformer_npz_image,
    resolve_nnformer_npz_dir,
)
from dinomim_pytorch.datasets.nnformer_npz_patch import (
    build_npz_patch_sampler,
    patch_sampler_enabled,
)
from dinomim_pytorch.augmentations_3d import (
    build_volume_view_augmentor,
    sample_anisotropic_crop_resize,
)
from dinomim_pytorch.masking import apply_view_masking, mark_masked_view_indices
from dinomim_pytorch.volume_global_local import resize_volume_to_spatial, sample_global_local_crops_3d


def get_volume_loader_kind(cfg: Dict[str, Any]) -> str:
    """``nifti_csv`` (default), ``nnformer_npz``, or ``ssl_ct_manifest``."""
    data = (cfg or {}).get("data") or {}
    kind = str(data.get("loader", "nifti_csv")).lower().replace("-", "_")
    if kind in ("ssl_ct_manifest", "manifest_csv", "public_ct_ssl", "ssl_manifest"):
        return "ssl_ct_manifest"
    if kind in ("nnformer", "nnformer_npz", "npz", "nnformer_preprocessed"):
        return "nnformer_npz"
    return "nifti_csv"


def has_ssl_volume_data(cfg: Dict[str, Any]) -> bool:
    """True when config points to a usable 3D SSL volume source."""
    if get_volume_loader_kind(cfg) == "ssl_ct_manifest":
        from dinomim_pytorch.datasets.ssl_ct_manifest_dataset import has_ssl_ct_manifest_data

        return has_ssl_ct_manifest_data(cfg)
    if get_volume_loader_kind(cfg) == "nnformer_npz":
        return has_nnformer_npz_data(cfg)
    return csv_has_ssl_volume_paths(cfg)


def build_ssl_volume_dataset(cfg: Dict[str, Any]) -> Dataset:
    data = (cfg or {}).get("data") or {}
    if get_volume_loader_kind(cfg) == "ssl_ct_manifest":
        from dinomim_pytorch.datasets.ssl_ct_manifest_dataset import (
            SslCtManifestPatchDataset,
            SslCtManifestVolumeDataset,
        )

        if patch_sampler_enabled(data):
            return SslCtManifestPatchDataset(cfg)
        return SslCtManifestVolumeDataset(cfg)
    if get_volume_loader_kind(cfg) == "nnformer_npz":
        if patch_sampler_enabled(data):
            return NnformerNpzPatchVolumeDataset(cfg)
        return NnformerNpzVolumeDataset(cfg)
    return VolumeMaskedDINODataset(cfg)


def csv_has_ssl_volume_paths(cfg: Dict[str, Any]) -> bool:
    data = (cfg or {}).get("data") or {}
    idx = data.get("index_csv")
    if not idx or not Path(str(idx)).is_file():
        return False
    return len(_load_paths(data)) > 0


def _load_paths(data: Dict[str, Any]) -> List[str]:
    import csv

    idx = data.get("index_csv")
    col = str(data.get("image_col", "path"))
    if not idx:
        return []
    rows: List[str] = []
    with open(idx, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for line in r:
            if col in line and line[col]:
                rows.append(str(line[col]).strip())
    return rows


def _normalize_channels(t: torch.Tensor, mode: str) -> torch.Tensor:
    if t.dim() == 3:
        t = t.unsqueeze(0)
    m = (mode or "zscore").lower()
    if m in ("none", "off", "identity"):
        return t
    if m == "zscore":
        out = t.clone()
        for c in range(out.shape[0]):
            ch = out[c]
            mu = ch.mean()
            sd = ch.std().clamp_min(1e-6)
            out[c] = (ch - mu) / sd
        return out
    raise ValueError(f"Unknown data.volume_normalize={mode!r}")


def model_spatial_tuple_from_cfg(config: Dict[str, Any]) -> Tuple[int, int, int]:
    """Canonical 3D grid from ``model.spatial_size`` (matches UNETR ``img_size``)."""
    m = (config or {}).get("model") or {}
    sp = m.get("spatial_size")
    if isinstance(sp, (list, tuple)) and len(sp) == 3:
        return (int(sp[0]), int(sp[1]), int(sp[2]))
    if isinstance(sp, (int, float)):
        s = int(sp)
        return (s, s, s)
    return (96, 96, 96)


def resolve_branch_spatial(
    section: Optional[Dict[str, Any]],
    model_spatial: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """
    Per-view output voxel grid ``(D,H,W)``.

    Priority: ``spatial_size`` (3 ints) > legacy ``output_size`` (cube) > ``model.spatial_size``.
    """
    if not isinstance(section, dict):
        return model_spatial
    ss = section.get("spatial_size")
    if isinstance(ss, (list, tuple)) and len(ss) == 3:
        return (int(ss[0]), int(ss[1]), int(ss[2]))
    oz = section.get("output_size")
    if oz is not None:
        s = int(oz)
        return (s, s, s)
    return model_spatial


def _sample_cube_resize(
    x: torch.Tensor,
    target_dhw: Tuple[int, int, int],
    scale: Tuple[float, float],
    noise_std: float,
    *,
    max_scale_when_fits: Optional[float] = 0.9,
) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """
    Legacy alias: anisotropic crop (replaces isotropic cube on [C,D,H,W]).
    Returns patch and ``(z0, y0, x0, side)`` with ``side=max(sd,sh,sw)`` for nesting.
    """
    patch, box = sample_anisotropic_crop_resize(
        x,
        target_dhw,
        scale,
        max_scale_when_fits=max_scale_when_fits,
    )
    z0, y0, x0, sd, sh, sw = box
    if noise_std > 0:
        patch = (patch + noise_std * torch.randn_like(patch)).clamp(-6.0, 6.0)
    side = max(sd, sh, sw)
    return patch, (z0, y0, x0, side)


def _nested_anisotropic_resize(
    x: torch.Tensor,
    parent_box: Tuple[int, int, int, int, int, int],
    target_dhw: Tuple[int, int, int],
    scale: Tuple[float, float],
    noise_std: float,
) -> torch.Tensor:
    """
    Random axis-aligned crop **inside** parent ``(z0,y0,x0,sd,sh,sw)``, resize to ``target_dhw``.
    """
    pz0, py0, px0, pd, ph, pw = parent_box
    pd = max(1, int(pd))
    ph = max(1, int(ph))
    pw = max(1, int(pw))
    lo, hi = float(scale[0]), float(scale[1])
    sd = max(1, min(pd, int(random.uniform(lo, hi) * pd)))
    sh = max(1, min(ph, int(random.uniform(lo, hi) * ph)))
    sw = max(1, min(pw, int(random.uniform(lo, hi) * pw)))
    zz = random.randint(pz0, max(pz0, pz0 + pd - sd))
    yy = random.randint(py0, max(py0, py0 + ph - sh))
    xx = random.randint(px0, max(px0, px0 + pw - sw))
    patch = x[:, zz : zz + sd, yy : yy + sh, xx : xx + sw]
    if noise_std > 0:
        patch = (patch + noise_std * torch.randn_like(patch)).clamp(-6.0, 6.0)
    td, th, tw = int(target_dhw[0]), int(target_dhw[1]), int(target_dhw[2])
    if patch.shape[-3:] != (td, th, tw):
        if min(patch.shape[-3:]) < 1:
            raise ValueError(
                f"Nested local crop empty: parent_box={parent_box} patch_shape={tuple(patch.shape)}"
            )
        patch = F.interpolate(
            patch.unsqueeze(0),
            size=(td, th, tw),
            mode="trilinear",
            align_corners=False,
        )[0]
    return patch


def _random_cubic_crop_resize(
    x: torch.Tensor,
    out_size: int,
    scale: Tuple[float, float],
    noise_std: float,
) -> torch.Tensor:
    """Legacy: cubic ``out_size``³ output (used when callers pass scalar only)."""
    t = (int(out_size), int(out_size), int(out_size))
    p, _ = _sample_cube_resize(x, t, scale, noise_std)
    return p


def _make_views_3d_global_local(
    x: torch.Tensor,
    config: Dict[str, Any],
    gl: Dict[str, Any],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], Dict[str, Any]]:
    """
    Fixed-size global + local native voxel crops (BYOL-style), then resize to each branch
    ``spatial_size``. Teacher uses the **global** crop only; student global / local use global /
    local crops respectively. Each view index re-draws an independent (global, local) pair.
    """
    teacher = (config or {}).get("teacher") or {}
    student = (config or {}).get("student") or {}
    model_sp = model_spatial_tuple_from_cfg(config)

    gc = gl.get("global_crop_size")
    lc = gl.get("local_crop_size")
    if not (isinstance(gc, (list, tuple)) and len(gc) == 3):
        raise ValueError("student.global_local_3d.global_crop_size must be [D,H,W]")
    if not (isinstance(lc, (list, tuple)) and len(lc) == 3):
        raise ValueError("student.global_local_3d.local_crop_size must be [D,H,W]")
    g_sz = (int(gc[0]), int(gc[1]), int(gc[2]))
    l_sz = (int(lc[0]), int(lc[1]), int(lc[2]))
    nested = bool(gl.get("nested_local_in_global", True))

    tg = teacher.get("global") or {}
    sg = student.get("global") or {}
    sl = student.get("local") or {}
    n_t = int(tg.get("num_views", 2))
    n_sg = int(sg.get("num_views", 2))
    n_sl = int(sl.get("num_views", 0))

    t_sp = resolve_branch_spatial(tg, model_sp)
    sg_sp = resolve_branch_spatial(sg, model_sp)
    sl_sp = resolve_branch_spatial(sl, model_sp)

    t_mask = (tg.get("masking") or {})
    t_aug = build_volume_view_augmentor(tg, t_sp, default_strength="weak")
    s_aug = build_volume_view_augmentor(sg, sg_sp, default_strength="strong")
    sm = (sg.get("masking") or {})
    n_mg = int(sm.get("num_masked_views", 0) or 0) if sm.get("enabled") else 0
    masked_g_idx = set(mark_masked_view_indices(n_sg, n_mg, "first")) if n_mg else set()
    lm = (sl.get("masking") or {})
    n_ml = int(lm.get("num_masked_views", 0) or 0) if lm.get("enabled") else 0
    masked_l_idx = set(mark_masked_view_indices(n_sl, n_ml, "first")) if n_ml else set()
    sl_aug = build_volume_view_augmentor(sl, sl_sp, default_strength="strong")

    def _branch(x5: torch.Tensor, sp: Tuple[int, int, int]) -> torch.Tensor:
        return resize_volume_to_spatial(x5, sp)[0]

    x5 = x.unsqueeze(0)
    t_views: List[torch.Tensor] = []
    for _ in range(n_t):
        xg, _xl = sample_global_local_crops_3d(
            x5, g_sz, l_sz, nested_local_in_global=nested
        )
        v = t_aug(_branch(xg, t_sp))
        t_views.append(v)
    if t_mask.get("enabled") is True:
        for i in mark_masked_view_indices(
            n_t, int(t_mask.get("num_masked_views", 0)), "first"
        ):
            t_views[i] = apply_view_masking(t_views[i], t_mask)

    s_glob: List[torch.Tensor] = []
    for i in range(n_sg):
        xg, _xl = sample_global_local_crops_3d(
            x5, g_sz, l_sz, nested_local_in_global=nested
        )
        v = s_aug(_branch(xg, sg_sp))
        if i in masked_g_idx:
            v = apply_view_masking(v, sm)
        s_glob.append(v)

    s_loc: List[torch.Tensor] = []
    for j in range(n_sl):
        _xg, xl = sample_global_local_crops_3d(
            x5, g_sz, l_sz, nested_local_in_global=nested
        )
        v = sl_aug(_branch(xl, sl_sp))
        if j in masked_l_idx:
            v = apply_view_masking(v, lm)
        s_loc.append(v)

    meta = {
        "masked_global_idx": list(masked_g_idx),
        "masked_local_idx": list(masked_l_idx),
        "teacher_spatial": list(t_sp),
        "student_global_spatial": list(sg_sp),
        "student_local_spatial": list(sl_sp),
        "nested_local": nested,
        "global_local_3d": True,
        "global_crop_size": list(g_sz),
        "local_crop_size": list(l_sz),
    }
    return t_views, s_glob, s_loc, meta


def _make_views_3d(
    x: torch.Tensor, config: Dict[str, Any]
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], Dict[str, Any]]:
    """
    Build teacher / student global / student local views ``[C,D,H,W]``.

    Each branch can use a distinct ``spatial_size: [D,H,W]`` (true global vs local voxel grids
    before the backbone). Training then interpolates to ``model.spatial_size`` for UNETR.

    Optional ``student.local.nested_in_student_global``: local crops are sampled **inside** the
    first student-global crop box (voxel-aligned), then resized to the local ``spatial_size``.

    Optional ``student.global_local_3d.enabled``: fixed ``global_crop_size`` / ``local_crop_size``
    native crops (nested local in global by default), then resize to each branch grid — aligned
    with BYOL global–local 3D pretrain.
    """
    teacher = (config or {}).get("teacher") or {}
    student = (config or {}).get("student") or {}
    gl3 = student.get("global_local_3d") or {}
    if gl3.get("enabled") is True:
        return _make_views_3d_global_local(x, config, gl3)

    model_sp = model_spatial_tuple_from_cfg(config)

    tg = teacher.get("global") or {}
    n_t = int(tg.get("num_views", 2))
    t_sp = resolve_branch_spatial(tg, model_sp)
    t_mask = (tg.get("masking") or {})
    t_aug = build_volume_view_augmentor(tg, t_sp, default_strength="weak")
    t_views: List[torch.Tensor] = []
    for _ in range(n_t):
        t_views.append(t_aug(x.clone()))
    if t_mask.get("enabled") is True:
        for i in mark_masked_view_indices(
            n_t, int(t_mask.get("num_masked_views", 0)), "first"
        ):
            t_views[i] = apply_view_masking(t_views[i], t_mask)

    sg = student.get("global") or {}
    n_sg = int(sg.get("num_views", 2))
    sg_sp = resolve_branch_spatial(sg, model_sp)
    s_aug = build_volume_view_augmentor(sg, sg_sp, default_strength="strong")
    sm = (sg.get("masking") or {})
    n_mg = int(sm.get("num_masked_views", 0) or 0) if sm.get("enabled") else 0
    masked_g_idx = set(mark_masked_view_indices(n_sg, n_mg, "first")) if n_mg else set()
    s_glob: List[torch.Tensor] = []
    sg_boxes: List[Tuple[int, int, int, int]] = []
    for i in range(n_sg):
        v = s_aug(x.clone())
        if i in masked_g_idx:
            v = apply_view_masking(v, sm)
        s_glob.append(v)

    sl = student.get("local") or {}
    n_sl = int(sl.get("num_views", 0))
    sl_sp = resolve_branch_spatial(sl, model_sp)
    sl_aug = build_volume_view_augmentor(sl, sl_sp, default_strength="strong")
    nested = bool(sl.get("nested_in_student_global", False))
    lm = (sl.get("masking") or {})
    n_ml = int(lm.get("num_masked_views", 0) or 0) if lm.get("enabled") else 0
    masked_l_idx = set(mark_masked_view_indices(n_sl, n_ml, "first")) if n_ml else set()
    s_loc: List[torch.Tensor] = []
    ref_box6: Optional[Tuple[int, int, int, int, int, int]] = None
    if nested and s_glob:
        _, ref_box6 = sample_anisotropic_crop_resize(
            x,
            sg_sp,
            tuple(sg.get("crop_scale", (0.35, 0.75))),
            max_scale_when_fits=float(sg.get("max_scale_when_fits", 0.8)),
        )
    for j in range(n_sl):
        if nested and ref_box6 is not None:
            v = _nested_anisotropic_resize(
                x.clone(),
                ref_box6,
                sl_sp,
                tuple(sl.get("crop_scale", (0.35, 0.65))),
                float(sl.get("noise_std", 0.0)),
            )
        else:
            v = sl_aug(x.clone())
        if j in masked_l_idx:
            v = apply_view_masking(v, lm)
        s_loc.append(v)

    meta = {
        "masked_global_idx": list(masked_g_idx),
        "masked_local_idx": list(masked_l_idx),
        "teacher_spatial": list(t_sp),
        "student_global_spatial": list(sg_sp),
        "student_local_spatial": list(sl_sp),
        "nested_local": nested,
    }
    return t_views, s_glob, s_loc, meta


class NnformerNpzPatchVolumeDataset(Dataset):
    """
    DINO SSL on random nnFormer patches (MAE_BYOL ``unetrpp_npz`` parity).

    Uses ``splits_final.pkl`` train cases only, ``samples_per_epoch`` random crops,
    and foreground oversampling via per-case ``.pkl`` ``class_locations``.
    Multi-view DINO crops are drawn **inside each patch** (not from full volumes).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = config
        data = (config or {}).get("data") or {}
        self.normalize = str(data.get("volume_normalize", "none"))
        self._sampler = build_npz_patch_sampler(data, train=True)
        self.n_cases = len(self._sampler.case_ids)
        self._debug = (config or {}).get("logging", {}).get("save_debug_views", False)

    def __len__(self) -> int:
        return len(self._sampler)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img, _ = self._sampler[idx]
        want_c = int((self.cfg.get("model") or {}).get("in_channels", img.shape[0]))
        if img.shape[0] != want_c:
            raise ValueError(
                f"npz channels={img.shape[0]} vs model.in_channels={want_c}"
            )
        vol = _normalize_channels(img, self.normalize)
        tg, sg, sl, meta = _make_views_3d(vol, self.cfg)
        meta["patch_sampler"] = True
        meta["n_train_cases"] = self.n_cases
        return {
            "teacher_glob": tg,
            "student_glob": sg,
            "student_loc": sl,
            "volume": vol,
            "path": "patch",
            "mask_meta": meta,
        }


class NnformerNpzVolumeDataset(Dataset):
    """
    SSL from nnFormer preprocessed ``*.npz`` (same tensors as UNETR++ / nnFormer training).

    Set ``data.loader: nnformer_npz`` and either ``data.nnformer_npz_dir`` (folder of npz) or
    ``data.nnformer_preprocessed_dir`` (task root; auto-picks ``nnFormerData_plans*_stage0``).
    Use ``data.volume_normalize: none`` — intensities are already normalized in preprocessing.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        npz_paths: Optional[List[Path]] = None,
    ) -> None:
        super().__init__()
        self.cfg = config
        data = (config or {}).get("data") or {}
        self.normalize = str(data.get("volume_normalize", "none"))
        folder = resolve_nnformer_npz_dir(data)
        if folder is None:
            raise FileNotFoundError(
                "nnformer_npz loader: set data.nnformer_npz_dir or "
                "data.nnformer_preprocessed_dir to a folder with *.npz"
            )
        self.folder = folder
        self.paths: List[Path] = npz_paths or list_npz_files(folder)
        if not self.paths:
            raise FileNotFoundError(f"No *.npz files under {folder}")
        self._debug = (config or {}).get("logging", {}).get("save_debug_views", False)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        p = self.paths[idx % len(self.paths)]
        vol = load_nnformer_npz_image(p)
        want_c = int((self.cfg.get("model") or {}).get("in_channels", vol.shape[0]))
        if vol.shape[0] != want_c:
            raise ValueError(
                f"npz channels={vol.shape[0]} vs model.in_channels={want_c} ({p})"
            )
        vol = _normalize_channels(vol, self.normalize)
        tg, sg, sl, meta = _make_views_3d(vol, self.cfg)
        return {
            "teacher_glob": tg,
            "student_glob": sg,
            "student_loc": sl,
            "volume": vol,
            "path": str(p),
            "mask_meta": meta,
        }


class VolumeMaskedDINODataset(Dataset):
    """
    SSL-only: rows from ``data.index_csv`` with ``data.image_col`` (default ``path``).
    Loads NIfTI volumes [C,D,H,W], optional ``data.volume_normalize``.

    Default when ``data.loader`` is omitted or ``nifti_csv``.
    """

    def __init__(self, config: Dict[str, Any], paths: Optional[List[str]] = None) -> None:
        super().__init__()
        self.cfg = config
        data = (config or {}).get("data") or {}
        self.paths: List[str] = paths or _load_paths(data)
        self.normalize = str(data.get("volume_normalize", "zscore"))
        self._debug = (config or {}).get("logging", {}).get("save_debug_views", False)

    def __len__(self) -> int:
        n = len(self.paths)
        return n if n > 0 else 1

    def _dummy_volume(self, config: Dict[str, Any]) -> torch.Tensor:
        mcfg = (config or {}).get("model") or {}
        sp = mcfg.get("spatial_size") or [96, 96, 96]
        if isinstance(sp, (list, tuple)) and len(sp) == 3:
            d, h, w = int(sp[0]), int(sp[1]), int(sp[2])
        else:
            d = h = w = 96
        c = int(mcfg.get("in_channels", 1))
        return torch.randn(c, d, h, w)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not self.paths:
            vol = self._dummy_volume(self.cfg)
        else:
            p = self.paths[idx % len(self.paths)]
            arr, _ = load_nifti_array(p)
            if arr.ndim == 3:
                vol = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).unsqueeze(0)
            elif arr.ndim == 4:
                vol = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
            else:
                raise ValueError(f"Expected 3D or 4D NIfTI data, got shape {arr.shape}")
            want_c = int((self.cfg.get("model") or {}).get("in_channels", vol.shape[0]))
            if vol.shape[0] != want_c:
                raise ValueError(
                    f"Volume channels={vol.shape[0]} vs model.in_channels={want_c} ({p})"
                )
            vol = _normalize_channels(vol, self.normalize)

        tg, sg, sl, meta = _make_views_3d(vol, self.cfg)
        return {
            "teacher_glob": tg,
            "student_glob": sg,
            "student_loc": sl,
            "volume": vol,
            "path": self.paths[idx % len(self.paths)] if self.paths else "",
            "mask_meta": meta,
        }


def volume_multiview_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack batch along B for each view (5D tensors)."""

    def _stack(views: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(views, dim=0)

    t_lists = [b["teacher_glob"] for b in batch]
    s_glob_lists = [b["student_glob"] for b in batch]
    s_loc_lists = [b["student_loc"] for b in batch]
    t_dim = max(len(t) for t in t_lists)
    sg_dim = max(len(t) for t in s_glob_lists)
    sl_dim = max(len(t) for t in s_loc_lists)
    teacher = [_stack([b["teacher_glob"][i] for b in batch]) for i in range(t_dim)]
    s_glob = [_stack([b["student_glob"][i] for b in batch]) for i in range(sg_dim)]
    s_loc = [_stack([b["student_loc"][i] for b in batch]) for i in range(sl_dim)]
    out: Dict[str, Any] = {
        "teacher_glob": teacher,
        "student_glob": s_glob,
        "student_loc": s_loc,
        "path": [b.get("path", "") for b in batch],
        "mask_meta": [b.get("mask_meta", {}) for b in batch],
    }
    if batch and "volume" in batch[0]:
        out["volume"] = _stack([b["volume"] for b in batch])
    return out


make_views_3d = _make_views_3d

__all__ = [
    "VolumeMaskedDINODataset",
    "NnformerNpzVolumeDataset",
    "volume_multiview_collate_fn",
    "make_views_3d",
    "_make_views_3d",
    "build_ssl_volume_dataset",
    "get_volume_loader_kind",
    "has_ssl_volume_data",
    "csv_has_ssl_volume_paths",
    "model_spatial_tuple_from_cfg",
    "resolve_branch_spatial",
]
