"""
Multi-view dataset: teacher global, student global, student local crops from 2D images (or 3D volumes — optional slice).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from dinomim_pytorch.augmentations import MedicalStrongAug, MedicalWeakGlobalAug
from dinomim_pytorch.masking import apply_view_masking, mark_masked_view_indices


def _load_image_2d(path: str, channels: int) -> torch.Tensor:
    from PIL import Image

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    img = Image.open(p)
    arr = np.array(img.convert("L" if channels == 1 else "RGB"))
    if arr.ndim == 2:
        arr = arr.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        t = torch.from_numpy(arr).unsqueeze(0)
        if channels > 1:
            t = t.repeat(channels, 1, 1)
        return t.clamp(0, 1)
    if arr.ndim == 3:
        t = torch.from_numpy(arr.astype(np.float32))
        if t.dim() == 3 and t.shape[-1] <= 4:
            t = t.permute(2, 0, 1)
        if t.shape[0] > channels:
            t = t[:channels]
        if t.max() > 1.5:
            t = t / 255.0
        if t.shape[0] < channels:
            t = t[0:1].repeat(channels, 1, 1)
        return t.clamp(0, 1)
    raise ValueError(f"Unsupported array shape {arr.shape}")


class MultiviewMaskedDINODataset(Dataset):
    """
    Returns dict:
      teacher_glob: List[Tensor C,H,W]
      student_glob: List[Tensor]
      student_loc: List[Tensor]
      meta: debug info
    """

    def __init__(self, config: Dict[str, Any], paths: Optional[List[str]] = None):
        super().__init__()
        self.cfg = config
        data = (config or {}).get("data") or {}
        self.paths: List[str] = paths or self._load_paths(data)
        self.channels = int(data.get("channels", 1))
        self.teacher = (config or {}).get("teacher") or {}
        self.student = (config or {}).get("student") or {}
        self._debug_mask = (config or {}).get("logging", {}).get("save_debug_views", False)

    @staticmethod
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

    def __len__(self) -> int:
        n = len(self.paths)
        return n if n > 0 else 1

    def _single_dummy(self) -> torch.Tensor:
        s = int((self.cfg.get("data") or {}).get("image_size", 224))
        c = int(self.channels)
        return torch.rand(c, s, s)

    def _make_views_2d(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], Dict[str, Any]]:
        tg = self.teacher.get("global") or {}
        n_t = int(tg.get("num_views", 2))
        t_size = int(tg.get("output_size", 224))
        t_scale = tuple(tg.get("crop_scale", (0.6, 1.0)))
        weak = MedicalWeakGlobalAug(t_size, t_scale)
        t_mask = (tg.get("masking") or {})
        t_views = [weak(x.clone()) for _ in range(n_t)]
        if t_mask.get("enabled") is True:
            for i in mark_masked_view_indices(n_t, int(t_mask.get("num_masked_views", 0)), "first"):
                t_views[i] = apply_view_masking(t_views[i], t_mask)

        sg = (self.student.get("global") or {})
        n_sg = int(sg.get("num_views", 2))
        sg_size = int(sg.get("output_size", 224))
        sg_scale = tuple(sg.get("crop_scale", (0.6, 1.0)))
        strong = MedicalStrongAug(sg_size, sg_scale)
        sm = (sg.get("masking") or {})
        n_mg = int(sm.get("num_masked_views", 0) or 0) if sm.get("enabled") else 0
        masked_g_idx = set(mark_masked_view_indices(n_sg, n_mg, "first")) if n_mg else set()
        s_glob = []
        for i in range(n_sg):
            v = strong(x.clone())
            if i in masked_g_idx:
                v = apply_view_masking(v, sm)
            s_glob.append(v)

        sl = (self.student.get("local") or {})
        n_sl = int(sl.get("num_views", 4))
        sl_size = int(sl.get("output_size", 96))
        sl_scale = tuple(sl.get("crop_scale", (0.2, 0.5)))
        loc = MedicalStrongAug(sl_size, sl_scale)
        lm = (sl.get("masking") or {})
        n_ml = int(lm.get("num_masked_views", 0) or 0) if lm.get("enabled") else 0
        masked_l_idx = set(mark_masked_view_indices(n_sl, n_ml, "first")) if n_ml else set()
        s_loc = []
        for j in range(n_sl):
            v = loc(x.clone())
            if j in masked_l_idx:
                v = apply_view_masking(v, lm)
            s_loc.append(v)

        return t_views, s_glob, s_loc, {
            "masked_global_idx": list(masked_g_idx),
            "masked_local_idx": list(masked_l_idx),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not self.paths:
            x0 = self._single_dummy()
        else:
            x0 = _load_image_2d(self.paths[idx % len(self.paths)], self.channels)
        tg, sg, sl, m = self._make_views_2d(x0)
        return {
            "teacher_glob": tg,
            "student_glob": sg,
            "student_loc": sl,
            "path": self.paths[idx % len(self.paths)] if self.paths else "",
            "mask_meta": m,
        }


def multiview_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
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
    return {
        "teacher_glob": teacher,
        "student_glob": s_glob,
        "student_loc": s_loc,
        "path": [b.get("path", "") for b in batch],
        "mask_meta": [b.get("mask_meta", {}) for b in batch],
    }
