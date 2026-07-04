from __future__ import annotations

import copy
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dinomim_pytorch.datasets.medical_3d_segmentation_dataset import load_nifti_array


def _normalize_volume_channels(x: torch.Tensor, mode: str) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0)
    mode_l = str(mode or "zscore").lower()
    if mode_l in ("none", "off", "identity"):
        return x
    if mode_l != "zscore":
        raise ValueError(f"Unsupported normalization mode: {mode!r}")
    out = x.clone()
    for c in range(out.shape[0]):
        mu = out[c].mean()
        sd = out[c].std().clamp_min(1e-6)
        out[c] = (out[c] - mu) / sd
    return out


class UnlabeledVolumeDataset(Dataset):
    """Reads unlabeled NIfTI volumes from data.index_csv/data.image_col."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        data = (cfg or {}).get("data") or {}
        self.index_csv = str(data.get("index_csv", "") or "")
        self.image_col = str(data.get("image_col", "path") or "path")
        self.normalize_mode = str(data.get("volume_normalize", "zscore"))
        self.want_c = int(((cfg or {}).get("model") or {}).get("in_channels", 1))
        self.is_3d = _is_3d_model_cfg((cfg or {}).get("model") or {}, data)
        model_cfg = (cfg or {}).get("model") or {}
        self.spatial = _spatial_tuple(
            model_cfg.get("spatial_size", model_cfg.get("img_size", data.get("image_size"))),
            is_3d=self.is_3d,
        )
        self.paths = self._read_paths()

    def _read_paths(self) -> List[str]:
        if not self.index_csv or not Path(self.index_csv).is_file():
            return []
        rows: List[str] = []
        with open(self.index_csv, "r", encoding="utf-8", newline="") as fh:
            r = csv.DictReader(fh)
            for line in r:
                p = (line.get(self.image_col) or "").strip()
                if p:
                    rows.append(p)
        return rows

    def __len__(self) -> int:
        return len(self.paths) if self.paths else 1

    def _dummy(self) -> torch.Tensor:
        if self.is_3d:
            d, h, w = self.spatial
            return torch.randn(self.want_c, d, h, w)
        h, w = self.spatial
        return torch.randn(self.want_c, h, w)

    def _load_2d_image(self, p: str) -> torch.Tensor:
        with Image.open(p) as img:
            if self.want_c == 1:
                img = img.convert("L")
                arr = np.asarray(img, dtype=np.float32)[None, ...]
            else:
                img = img.convert("RGB")
                arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)
        x = torch.from_numpy(np.ascontiguousarray(arr))
        if x.max() > 1.5:
            x = x / 255.0
        return x

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not self.paths:
            x = self._dummy()
            p = ""
        else:
            p = self.paths[idx % len(self.paths)]
            if self.is_3d:
                arr, _ = load_nifti_array(p)
                if arr.ndim == 3:
                    x = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).unsqueeze(0)
                elif arr.ndim == 4:
                    x = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
                else:
                    raise ValueError(f"Expected 3D/4D NIfTI volume, got {arr.shape} ({p})")
            else:
                x = self._load_2d_image(p)
            if x.shape[0] != self.want_c:
                raise ValueError(f"in_channels mismatch for {p}: got {x.shape[0]}, want {self.want_c}")
            x = _normalize_volume_channels(x, self.normalize_mode)
        x = _resize_spatial(x, self.spatial)
        return {"volume": x, "path": p}


def _collate_unlabeled(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    x = torch.stack([b["volume"] for b in batch], dim=0)
    return {"volume": x, "path": [b.get("path", "") for b in batch]}


def build_unlabeled_loader(cfg: Dict[str, Any]) -> DataLoader:
    train = (cfg or {}).get("training") or {}
    ds = UnlabeledVolumeDataset(cfg)
    return DataLoader(
        ds,
        batch_size=int(train.get("batch_size", 2)),
        shuffle=True,
        num_workers=int(train.get("num_workers", 0)),
        drop_last=True,
        collate_fn=_collate_unlabeled,
    )


def _is_3d_model_cfg(model_cfg: Dict[str, Any], data_cfg: Dict[str, Any]) -> bool:
    if "is_3d" in model_cfg:
        return bool(model_cfg.get("is_3d"))
    if int(model_cfg.get("spatial_dims", 3)) == 2:
        return False
    return str(data_cfg.get("input_type", "")).lower() not in {"image_2d", "slice_2d"}


def _spatial_tuple(sp: Any, *, is_3d: bool) -> Tuple[int, ...]:
    if isinstance(sp, (list, tuple)):
        if is_3d and len(sp) == 3:
            return (int(sp[0]), int(sp[1]), int(sp[2]))
        if (not is_3d) and len(sp) >= 2:
            return (int(sp[0]), int(sp[1]))
    if isinstance(sp, (int, float)):
        s = int(sp)
        return (s, s, s) if is_3d else (s, s)
    return (96, 96, 96) if is_3d else (224, 224)


def _interp_mode_for(x: torch.Tensor, *, spatial_ndim: int | None = None) -> str:
    """Pick interpolate mode: 3D spatial needs trilinear, 2D needs bilinear.

    Batched volumes are 5D (N,C,D,H,W); batched images 4D (N,C,H,W). Unbatched
    volumes are 4D (C,D,H,W) — indistinguishable from batched 2D by dim alone,
    so callers with unbatched tensors must pass spatial_ndim=len(size).
    """
    if spatial_ndim is not None:
        return "trilinear" if spatial_ndim == 3 else "bilinear"
    return "trilinear" if x.dim() == 5 else "bilinear"


def _resize_spatial(x: torch.Tensor, size: Tuple[int, ...]) -> torch.Tensor:
    if tuple(x.shape[-len(size) :]) == tuple(size):
        return x
    mode = _interp_mode_for(x, spatial_ndim=len(size))
    return F.interpolate(x.unsqueeze(0), size=size, mode=mode, align_corners=False)[0]


def apply_pretrain_scope(model: nn.Module, scope: str) -> Dict[str, int]:
    decoder_keys = (
        "decoder",
        "dec",
        "up",
        "head",
        "out",
        "seg",
        "cls",
        "final",
    )
    n_train, n_frozen = 0, 0
    for name, p in model.named_parameters():
        if scope == "encoder_decoder":
            p.requires_grad = True
        else:
            l = name.lower()
            is_decoderish = any(k in l for k in decoder_keys)
            p.requires_grad = not is_decoderish
        if p.requires_grad:
            n_train += p.numel()
        else:
            n_frozen += p.numel()
    return {"trainable": n_train, "frozen": n_frozen}


def build_optimizer_for_scope(model: nn.Module, cfg: Dict[str, Any], scope: str) -> torch.optim.Optimizer:
    train = (cfg or {}).get("training") or {}
    if scope == "encoder_decoder":
        lr_enc = float(train.get("lr_encoder", train.get("lr", 1e-4)))
        lr_dec = float(train.get("lr_decoder", train.get("lr", 1e-4)))
        wd = float(train.get("weight_decay", 0.0))
        enc, dec = [], []
        decoder_keys = ("decoder", "dec", "up", "head", "out", "seg", "cls", "final")
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in n.lower() for k in decoder_keys):
                dec.append(p)
            else:
                enc.append(p)
        groups = []
        if enc:
            groups.append({"params": enc, "lr": lr_enc})
        if dec:
            groups.append({"params": dec, "lr": lr_dec})
        return torch.optim.AdamW(groups, lr=lr_enc, weight_decay=wd)
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        params,
        lr=float(train.get("lr", 1e-4)),
        weight_decay=float(train.get("weight_decay", 0.0)),
    )


def random_block_mask_nd(x: torch.Tensor, mask_ratio: float, patch_size: int, mask_value: float = 0.0) -> torch.Tensor:
    y = x.clone()
    b = y.shape[0]
    spatial = tuple(y.shape[2:])
    dims = len(spatial)
    if dims not in (2, 3):
        raise ValueError(f"random_block_mask_nd expects 2D/3D tensors, got shape {tuple(y.shape)}")
    ps = max(1, int(patch_size))
    n_cells = [max(1, s // ps) for s in spatial]
    n_total = int(np.prod(n_cells))
    n_mask = max(1, int(mask_ratio * n_total))
    for bi in range(b):
        picks = random.sample(range(n_total), k=min(n_mask, n_total))
        for idx in picks:
            if dims == 3:
                n_d, n_h, n_w = n_cells
                zi = idx // (n_h * n_w)
                yi = (idx % (n_h * n_w)) // n_w
                xi = idx % n_w
                z0, y0, x0 = zi * ps, yi * ps, xi * ps
                y[bi, :, z0 : z0 + ps, y0 : y0 + ps, x0 : x0 + ps] = mask_value
            else:
                n_h, n_w = n_cells
                yi = idx // n_w
                xi = idx % n_w
                y0, x0 = yi * ps, xi * ps
                y[bi, :, y0 : y0 + ps, x0 : x0 + ps] = mask_value
    return y


def weak_strong_augment_pair(x: torch.Tensor, noise_std: float, cutout_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
    weak = x + 0.25 * noise_std * torch.randn_like(x)
    strong = x + noise_std * torch.randn_like(x)
    if cutout_ratio > 0:
        strong = random_block_mask_nd(strong, cutout_ratio, patch_size=8, mask_value=0.0)
    if all(s >= 8 for s in strong.shape[2:]):
        if strong.dim() == 5:
            strong = F.avg_pool3d(strong, kernel_size=3, stride=1, padding=1)
        elif strong.dim() == 4:
            strong = F.avg_pool2d(strong, kernel_size=3, stride=1, padding=1)
    return weak, strong


def random_crop_pair(x: torch.Tensor, size: Tuple[int, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
    spatial = tuple(x.shape[2:])
    if len(size) != len(spatial):
        size = tuple(spatial)
    if any(t > s for t, s in zip(size, spatial)):
        rz = F.interpolate(x, size=size, mode=_interp_mode_for(x), align_corners=False)
        return rz, rz.clone()

    def _crop(inp: torch.Tensor) -> torch.Tensor:
        starts = [random.randint(0, s - t) for s, t in zip(spatial, size)]
        if len(spatial) == 3:
            z0, y0, x0 = starts
            cd, ch, cw = size
            return inp[:, :, z0 : z0 + cd, y0 : y0 + ch, x0 : x0 + cw]
        y0, x0 = starts
        ch, cw = size
        return inp[:, :, y0 : y0 + ch, x0 : x0 + cw]

    return _crop(x), _crop(x)


class ReconAdapter(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, spatial_dims: int = 3) -> None:
        super().__init__()
        if int(spatial_dims) == 2:
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        else:
            self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def copy_teacher(model: nn.Module) -> nn.Module:
    t = copy.deepcopy(model)
    for p in t.parameters():
        p.requires_grad = False
    t.eval()
    return t


@torch.no_grad()
def update_teacher_ema(student: nn.Module, teacher: nn.Module, m: float) -> None:
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)


@dataclass
class ObjectiveResult:
    loss: torch.Tensor
    metrics: Dict[str, float]


def _ce_from_pseudo(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    pseudo = torch.argmax(teacher_logits.detach(), dim=1)
    return F.cross_entropy(student_logits, pseudo)


def compute_objective_losses(
    objectives_cfg: Dict[str, Any],
    model: nn.Module,
    teacher_model: Optional[nn.Module],
    x: torch.Tensor,
    recon_adapter: Optional[nn.Module],
    num_classes_for_pseudo: int,
) -> ObjectiveResult:
    losses: List[torch.Tensor] = []
    metrics: Dict[str, float] = {}

    def _enabled(name: str) -> Tuple[bool, float]:
        sec = objectives_cfg.get(name) or {}
        on = bool(sec.get("enabled", False))
        wt = float(sec.get("weight", 1.0))
        return on, wt

    pred_base = model(x)
    recon_base = recon_adapter(pred_base) if recon_adapter is not None else pred_base

    on, w = _enabled("mae_recon")
    if on:
        sec = objectives_cfg.get("mae_recon") or {}
        mask_ratio = float(sec.get("mask_ratio", 0.4))
        patch_size = int(sec.get("patch_size", 8))
        only_masked = bool(sec.get("reconstruct_only_masked", True))
        x_masked = random_block_mask_nd(x, mask_ratio=mask_ratio, patch_size=patch_size, mask_value=0.0)
        p = model(x_masked)
        pr = recon_adapter(p) if recon_adapter is not None else p
        if only_masked:
            m = (x_masked == 0).float()
            l = (torch.abs(pr - x) * m).sum() / m.sum().clamp_min(1.0)
        else:
            l = F.l1_loss(pr, x)
        losses.append(w * l)
        metrics["mae_recon"] = float(l.detach().cpu())

    on, w = _enabled("denoise_inpaint")
    if on:
        sec = objectives_cfg.get("denoise_inpaint") or {}
        noise_std = float(sec.get("noise_std", 0.1))
        cutout = float(sec.get("cutout_ratio", 0.2))
        _weak, strong = weak_strong_augment_pair(x, noise_std=noise_std, cutout_ratio=cutout)
        p = model(strong)
        pr = recon_adapter(p) if recon_adapter is not None else p
        l = F.l1_loss(pr, x)
        losses.append(w * l)
        metrics["denoise_inpaint"] = float(l.detach().cpu())

    on, w = _enabled("cross_view_recon")
    if on:
        sec = objectives_cfg.get("cross_view_recon") or {}
        crop_size = _spatial_tuple(sec.get("crop_size", [64, 64, 64]), is_3d=(x.dim() == 5))
        va, vb = random_crop_pair(x, crop_size)
        # Fixed-grid encoders (UNETR ViT, etc.) need inputs at training spatial_size for patch/pos embeddings.
        full_sz = tuple(int(s) for s in x.shape[2:])
        spatial_nd = len(full_sz)
        mode_resize = _interp_mode_for(va, spatial_ndim=spatial_nd)
        va_in = F.interpolate(va, size=full_sz, mode=mode_resize, align_corners=False)
        vb_tgt = F.interpolate(vb, size=full_sz, mode=mode_resize, align_corners=False)
        pa = model(va_in)
        pra = recon_adapter(pa) if recon_adapter is not None else pa
        if pra.shape[2:] != vb_tgt.shape[2:]:
            pra = F.interpolate(
                pra,
                size=vb_tgt.shape[2:],
                mode=_interp_mode_for(pra),
                align_corners=False,
            )
        l = F.l1_loss(pra, vb_tgt)
        losses.append(w * l)
        metrics["cross_view_recon"] = float(l.detach().cpu())

    on, w = _enabled("pseudo_label_dense")
    if on and teacher_model is not None:
        weak, strong = weak_strong_augment_pair(x, noise_std=0.05, cutout_ratio=0.0)
        with torch.no_grad():
            tlogits = teacher_model(weak)
            if tlogits.shape[1] > num_classes_for_pseudo > 1:
                tlogits = tlogits[:, :num_classes_for_pseudo]
        slogits = model(strong)
        if slogits.shape[1] > num_classes_for_pseudo > 1:
            slogits = slogits[:, :num_classes_for_pseudo]
        l = _ce_from_pseudo(slogits, tlogits)
        losses.append(w * l)
        metrics["pseudo_label_dense"] = float(l.detach().cpu())

    on, w = _enabled("consistency_dense")
    if on and teacher_model is not None:
        weak, strong = weak_strong_augment_pair(x, noise_std=0.08, cutout_ratio=0.1)
        with torch.no_grad():
            t = teacher_model(weak)
        s = model(strong)
        if t.shape != s.shape:
            c = min(t.shape[1], s.shape[1])
            t = t[:, :c]
            s = s[:, :c]
        l = F.mse_loss(torch.softmax(s, dim=1), torch.softmax(t.detach(), dim=1))
        losses.append(w * l)
        metrics["consistency_dense"] = float(l.detach().cpu())

    if not losses:
        dummy = recon_base.mean() * 0.0
        return ObjectiveResult(loss=dummy, metrics=metrics)
    total = torch.stack([v for v in losses]).sum()
    metrics["total_ssl_loss"] = float(total.detach().cpu())
    return ObjectiveResult(loss=total, metrics=metrics)
