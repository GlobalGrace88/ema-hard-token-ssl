#!/usr/bin/env python3
"""
Unified evaluation for DINO_MIM downstream tasks (MRI / CT 3D segmentation, CXR 2D classification).

Mirrors medical_mim layout: ``--checkpoint`` for a single run, or ``--ckpt_scratch`` + ``--ckpt_ssl``
for side-by-side metrics (segmentation + classification).

YAML hints:
  - Segmentation (mri/ct): ``data.index_val`` / CSV override, or ``data.loader: nnformer_npz`` for preprocessed npz.
  - ``mean_hd95`` uses MONAI 95% Hausdorff on foreground classes; set ``data.spacing`` (mm) for distances in mm, else voxel units.
  - Visualization (segmentation): default on. PNGs under ``viz_patch/`` (patch crops), ``viz_full_volume/`` (legacy name),
    and ``full_volume/cases_full_volume/`` (MAE_BYOL-style per-case three-plane panels).
  - Paper-style full-volume eval: ``--official_npz`` (all fold-0 val cases, sliding-window, 8-organ Synapse Dice+HD95).

YAML ``eval_vis`` (segmentation, optional):

  enabled: true
  num_patch_batches: 4
  num_full_volume_cases: 1
  sw_overlap: 0.5
  sw_batch_size: 2
  postprocess:                # CC filtering for viz/full-volume only (not patch metrics unless data.val_postprocess: true)
    enabled: true
    mode: auto
    keep_largest_cc: true
    keep_top_k: 1
    min_component_voxels: 50
    connectivity: 26
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import _bootstrap  # noqa: F401


def _write_compare_csv(metrics_a: Dict[str, Any], metrics_b: Dict[str, Any], out_path: Path) -> None:
    names = sorted(
        k
        for k in metrics_a
        if isinstance(metrics_a[k], (int, float, np.floating))
        and k in metrics_b
        and isinstance(metrics_b[k], (int, float, np.floating))
    )
    rows = [{"metric": m, "scratch": float(metrics_a[m]), "ssl": float(metrics_b[m]), "delta": float(metrics_b[m]) - float(metrics_a[m])} for m in names]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "scratch", "ssl", "delta"])
        w.writeheader()
        w.writerows(rows)


def _load_seg_state_dict(model: Any, ckpt_path: str) -> None:
    import torch

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # type: ignore[call-arg]
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        model.load_state_dict(ckpt, strict=False)  # type: ignore[arg-type]
        return
    if any(
        k in ckpt
        for k in ("student_backbone", "teacher_backbone", "student_head", "online_encoder")
    ):
        raise ValueError(
            f"{ckpt_path} looks like an SSL pretrain checkpoint, not a segmentation finetune "
            "checkpoint (e.g. last_model.pt). Use a downstream training export."
        )
    sd = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    if isinstance(sd, dict) and any(
        isinstance(k, str)
        and (
            k.startswith("student_backbone")
            or k.startswith("teacher_backbone")
            or k.startswith("online_encoder")
        )
        for k in sd
    ):
        raise ValueError(
            f"{ckpt_path} contains SSL encoder keys in the weight dict; expected downstream ``model`` tensors."
        )
    if isinstance(sd, dict) and any(isinstance(k, str) and k.startswith("net.") for k in sd):
        sd = {k.removeprefix("net."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)  # type: ignore[arg-type]


def _make_seg_loader(cfg: dict, *, train: bool, index_csv: str | None) -> Any:
    from torch.utils.data import DataLoader

    ds = _build_seg_dataset(cfg, train=train, index_csv=index_csv)
    if ds is None:
        return None
    return DataLoader(
        ds,
        batch_size=int((cfg or {}).get("training", {}).get("batch_size", 1)),
        shuffle=train,
        num_workers=int((cfg or {}).get("training", {}).get("num_workers", 0)),
    )


def _data_with_crop_from_model(cfg: dict) -> dict:
    """MONAI BraTS compose defaults ``image_size`` to 96³; UNETR needs Rand/center crop == ``model.img_size``."""
    d = dict((cfg or {}).get("data") or {})
    if d.get("image_size"):
        return d
    m = (cfg or {}).get("model") or {}
    sz = m.get("img_size") or m.get("spatial_size")
    if isinstance(sz, (list, tuple)) and len(sz) >= 3:
        d["image_size"] = [int(sz[0]), int(sz[1]), int(sz[2])]
    elif isinstance(sz, (int, float)) and int(sz) > 0:
        v = int(sz)
        d["image_size"] = [v, v, v]
    return d


def _build_seg_dataset(cfg: dict, *, train: bool, index_csv: str | None) -> Any:
    from dinomim_pytorch.datasets.seg_dataset_factory import (
        build_segmentation_dataset,
        has_segmentation_data,
    )

    if not has_segmentation_data(cfg, train=train, index_csv_override=index_csv):
        return None
    return build_segmentation_dataset(cfg, train=train, index_csv_override=index_csv)


def _voxel_spacing_mm_for_hd(cfg: dict) -> Optional[Tuple[float, float, float]]:
    """``data.spacing`` (e.g. NIfTI zooms in mm) for HD95; ``None`` uses unit voxel spacing."""
    sp = (cfg.get("data") or {}).get("spacing")
    if isinstance(sp, (list, tuple)) and len(sp) >= 3:
        return (float(sp[0]), float(sp[1]), float(sp[2]))
    return None


def _val_metric_accumulated(cfg: dict) -> bool:
    data = (cfg or {}).get("data") or {}
    mode = str(data.get("val_metric_mode", "auto")).strip().lower()
    if mode in ("accumulated", "tpfpfn", "global", "mae"):
        return True
    if mode in ("per_batch", "legacy", "batch"):
        return False
    from dinomim_pytorch.datasets.nnformer_npz_patch import patch_sampler_enabled

    return patch_sampler_enabled(data)


def _metrics_use_postprocess(cfg: dict) -> bool:
    """Scalar metrics: honor ``data.val_postprocess`` (finetune parity). Viz may still use ``eval_vis.postprocess``."""
    data = (cfg or {}).get("data") or {}
    if "val_postprocess" in data:
        return bool(data["val_postprocess"])
    return False


def _evaluate_segmentation(
    model: Any,
    loader: Any,
    device: Any,
    n_classes: int,
    *,
    use_amp: bool,
    image_key: str,
    label_key: str,
    spacing_mm: Optional[Tuple[float, float, float]] = None,
    cfg: Optional[dict] = None,
) -> Dict[str, float]:
    import torch
    import torch.nn.functional as F
    from torch import amp as torch_amp

    from dinomim_pytorch.datasets import unwrap_monai_dict_batch
    from dinomim_pytorch.eval import logits_to_label_map, resolve_postprocess_cfg
    from dinomim_pytorch.segmentation_models.class_labels import resolve_segmentation_class_names
    from dinomim_pytorch.segmentation_models.losses import (
        align_logits_to_labels,
        primary_segmentation_logits,
    )
    from dinomim_pytorch.segmentation_models.metrics import dice_per_class, iou_per_class, mean_hd95
    from dinomim_pytorch.segmentation_models.val_metrics_accum import (
        val_confusion_update,
        val_dice_from_accumulated,
    )

    pp_cfg = resolve_postprocess_cfg(cfg or {})
    use_pp = _metrics_use_postprocess(cfg or {})
    use_accum = _val_metric_accumulated(cfg or {})

    model.eval()
    dice_accum: List[float] = []
    iou_accum: List[float] = []
    hd_accum: List[float] = []
    tp = fp = fn = None
    if use_accum:
        tp = torch.zeros(n_classes, dtype=torch.float64, device=device)
        fp = torch.zeros(n_classes, dtype=torch.float64, device=device)
        fn = torch.zeros(n_classes, dtype=torch.float64, device=device)
    use_cuda_amp = use_amp and device.type == "cuda"
    with torch.no_grad():
        for batch in loader:
            batch = unwrap_monai_dict_batch(batch)
            if not isinstance(batch, dict):
                raise TypeError(f"Expected dict batch after unwrap, got {type(batch)}")
            x = batch.get(image_key, batch.get("image", batch.get("path")))
            y = batch.get(label_key, batch.get("label", batch.get("label_path")))
            if x is None or y is None:
                raise KeyError(
                    f"Batch missing tensors (image_key={image_key!r}, label_key={label_key!r}); "
                    f"keys={list(batch.keys())!r}"
                )
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x, dtype=torch.float32)
            if not isinstance(y, torch.Tensor):
                y = torch.as_tensor(y, dtype=torch.long)
            x = x.to(device)
            if y.dim() == 5 and y.size(1) == 1:
                y = y[:, 0]
            y = y.long().to(device)
            with torch_amp.autocast("cuda" if use_cuda_amp else "cpu", enabled=use_cuda_amp):
                logits_f = primary_segmentation_logits(model(x)).float()
            logits_f = align_logits_to_labels(logits_f, y.unsqueeze(1) if y.dim() == 4 else y)
            if use_pp:
                pred = logits_to_label_map(logits_f, n_classes, pp_cfg)
            else:
                pred = logits_f.argmax(dim=1)
            if use_accum and tp is not None and fp is not None and fn is not None:
                val_confusion_update(pred, y, n_classes, tp, fp, fn)
            else:
                dice_accum.append(
                    float(dice_per_class(pred, y, n_classes, softmax=False).detach().cpu())
                )
                iou_accum.append(
                    float(iou_per_class(pred, y, n_classes, softmax=False).detach().cpu())
                )
            hd_accum.append(
                float(
                    mean_hd95(
                        pred,
                        y,
                        n_classes,
                        softmax=False,
                        spacing=spacing_mm,
                    ).detach().cpu()
                )
            )
    out: Dict[str, float] = {"num_batches": float(len(loader))}
    if use_accum and tp is not None and fp is not None and fn is not None:
        class_names = resolve_segmentation_class_names((cfg or {}).get("data") or {}, n_classes)
        mean_dice, per_class, note = val_dice_from_accumulated(
            tp, fp, fn, n_classes, class_names=class_names
        )
        out["mean_dice"] = mean_dice
        for k, v in per_class.items():
            out[k.replace("val_dice_", "dice_")] = v
        print(f"[eval] metric=accumulated TP/FP/FN{note}", flush=True)
    else:
        n = max(len(dice_accum), 1)
        out["mean_dice"] = sum(dice_accum) / n
        out["mean_iou"] = sum(iou_accum) / n
        print("[eval] metric=per-batch macro Dice/IoU", flush=True)
    hd_vals = np.asarray(hd_accum, dtype=np.float64)
    finite_hd = hd_vals[np.isfinite(hd_vals)]
    out["mean_hd95"] = float(finite_hd.mean()) if finite_hd.size else float("nan")
    return out


def resolve_modality(cfg: dict, modality_arg: str) -> str:
    m = (modality_arg or "auto").strip().lower()
    exp = cfg.get("experiment") or {}
    task = str(exp.get("task", "") or "").lower()
    if m == "auto":
        if task == "classification":
            return "cxr"
        return str(exp.get("modality", "mri") or "mri").lower()
    return m


def _build_cxr_dataset(cfg: dict, csv_override: str | None) -> Any:
    from torch.utils.data import Dataset

    d = cfg.get("data") or {}
    raw = csv_override or d.get("index_csv") or d.get("csv_test") or d.get("csv_val")
    if not raw:
        raise ValueError("CXR eval needs data.index_csv, data.csv_test, or data.csv_val (or --csv).")
    csv_path = Path(str(raw)).expanduser().resolve()
    if not csv_path.is_file():
        raise ValueError(f"CXR CSV not found: {csv_path}")

    img_size = int(d.get("img_size", d.get("image_size", 224)))
    ic = int((cfg.get("model") or {}).get("in_channels", 1))
    icol = str(d.get("image_col", "path"))
    lcol = str(d.get("label_col", "label"))

    class CXRDatasetTorch(Dataset):
        def __init__(self) -> None:
            super().__init__()
            self._rows: List[Dict[str, str]] = []
            with open(csv_path, newline="", encoding="utf-8") as fh:
                for line in csv.DictReader(fh):
                    self._rows.append({k: (v or "").strip() for k, v in line.items() if k is not None})
            self.image_col = icol
            self.label_col = lcol
            self.img_size = img_size
            self.in_channels = ic

        def __len__(self) -> int:
            return len(self._rows)

        def __getitem__(self, idx: int) -> Tuple[Any, Any]:
            import torch
            from torchvision.transforms.functional import resize as tv_resize

            from dinomim_pytorch.multiview_dataset import _load_image_2d

            row = self._rows[idx]
            path = row.get(self.image_col, "")
            if not path or not Path(path).is_file():
                raise FileNotFoundError(f"Missing CXR image: {path!r}")
            x = _load_image_2d(path, self.in_channels)
            x = tv_resize(x, [self.img_size, self.img_size], antialias=True)
            lab = row.get(self.label_col, "")
            y = int(float(lab)) if str(lab).strip() != "" else 0
            return x, torch.tensor(y, dtype=torch.long)

    return CXRDatasetTorch()


def _evaluate_classification(
    model: Any,
    loader: Any,
    device: Any,
    num_classes: int,
    *,
    use_amp: bool,
) -> Dict[str, Any]:
    import torch
    from torch import amp as torch_amp

    from dinomim_pytorch.cxr_metrics import (
        accuracy as cxr_accuracy,
        auc_binary,
        confusion_matrix,
        precision_recall_f1,
        sensitivity_specificity,
    )

    model.eval()
    all_logits: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    use_cuda_amp = use_amp and device.type == "cuda"
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device)
            with torch_amp.autocast("cuda" if use_cuda_amp else "cpu", enabled=use_cuda_amp):
                logits = model(imgs)
            all_logits.append(logits.float().detach().cpu().numpy())
            all_y.append(labels.detach().cpu().numpy())
    y_true = np.concatenate(all_y, axis=0).reshape(-1)
    logits_np = np.concatenate(all_logits, axis=0)

    preds = logits_np.argmax(axis=1)
    if num_classes <= 2 and logits_np.shape[1] >= 2:
        probs = softmax_np(logits_np)[:, 1]
    elif num_classes <= 2:
        probs = 1.0 / (1.0 + np.exp(-logits_np.reshape(-1)))
    else:
        probs = softmax_np(logits_np)

    out: Dict[str, Any] = {
        "accuracy": cxr_accuracy(preds, y_true),
        "num_classes": int(num_classes),
        "num_samples": int(len(y_true)),
    }

    try:
        from sklearn.metrics import f1_score, roc_auc_score

        if num_classes <= 2:
            out["auroc"] = auc_binary(np.asarray(probs).reshape(-1), y_true)
            prec, rec, f1 = precision_recall_f1(preds, y_true)
            se, sp = sensitivity_specificity(preds, y_true)
            out.update({"precision": prec, "recall": rec, "f1": f1, "sensitivity": se, "specificity": sp})
            cm = confusion_matrix(y_true, preds, num_classes=2)
        else:
            out["macro_f1"] = float(f1_score(y_true, preds, average="macro", zero_division=0))
            out["weighted_f1"] = float(f1_score(y_true, preds, average="weighted", zero_division=0))
            try:
                out["macro_auroc_ovr"] = float(
                    roc_auc_score(
                        np.eye(num_classes)[y_true.astype(int)],
                        softmax_np(logits_np),
                        average="macro",
                        multi_class="ovr",
                    )
                )
            except Exception:  # noqa: BLE001
                out["macro_auroc_ovr"] = 0.0
            cm = confusion_matrix(y_true, preds, num_classes=num_classes)
        out["confusion_matrix"] = cm.tolist()
    except Exception:  # noqa: BLE001
        out["macro_f1"] = 0.0

    return out


def softmax_np(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=1, keepdims=True)


def _save_metrics(metrics: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {}
    for k, v in metrics.items():
        if isinstance(v, (np.ndarray, np.generic)):
            payload[k] = v.tolist() if isinstance(v, np.ndarray) else float(v)
        elif isinstance(v, (float, int, str)):
            payload[k] = v
        elif isinstance(v, list):
            payload[k] = v
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    scalar_row = {
        kk: vv
        for kk, vv in payload.items()
        if isinstance(vv, (int, float)) and kk != "confusion_matrix"
    }
    if scalar_row:
        with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(scalar_row.keys()))
            w.writeheader()
            w.writerow(scalar_row)


def _eval_vis_merged(cfg: dict, args: argparse.Namespace) -> Dict[str, Any]:
    sec = dict((cfg.get("eval_vis") or {}) or {})
    if getattr(args, "no_vis", False):
        sec["enabled"] = False
    elif getattr(args, "vis", False):
        sec["enabled"] = True
    elif sec.get("enabled") is None:
        sec["enabled"] = True
    sec.setdefault("enabled", True)
    npb = getattr(args, "num_patch_vis", None)
    if npb is not None:
        sec["num_patch_batches"] = max(1, int(npb))
    else:
        sec.setdefault(
            "num_patch_batches",
            max(1, int(sec.get("num_patch_batches", sec.get("num_patch_cases", 4)))),
        )
    nfv = getattr(args, "num_full_volume_vis", None)
    if nfv is not None:
        sec["num_full_volume_cases"] = max(0, int(nfv))
    else:
        sec.setdefault("num_full_volume_cases", max(0, int(sec.get("num_full_volume_cases", 1))))
    sec.setdefault("sw_overlap", float(sec.get("sw_overlap", 0.5)))
    sec.setdefault("sw_batch_size", int(sec.get("sw_batch_size", 2)))
    sec.setdefault("num_cxr_images", int(sec.get("num_cxr_images", 12)))
    if getattr(args, "cxr_vis", False):
        sec["cxr_visualize"] = True
    sec.setdefault("cxr_visualize", bool(sec.get("cxr_visualize", False)))
    return sec


def segmentation_vis_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_cxr_pred_grid(
    model: Any,
    loader: Any,
    device: Any,
    out_path: Path,
    *,
    max_images: int,
    num_classes: int,
    use_amp: bool,
) -> None:
    import torch
    from torch import amp as torch_amp

    plt_mod = segmentation_vis_plt()
    plt = plt_mod
    imgs_l: List[Any] = []
    pred_l: List[int] = []
    true_l: List[int] = []
    use_cuda_amp = use_amp and device.type == "cuda"
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            with torch_amp.autocast("cuda" if use_cuda_amp else "cpu", enabled=use_cuda_amp):
                logits = model(xb)
            pr = logits.argmax(dim=1).cpu().numpy().ravel().tolist()
            yt = yb.detach().cpu().numpy().ravel().tolist()
            for i in range(xb.shape[0]):
                imgs_l.append(xb[i].detach().cpu())
                pred_l.append(int(pr[i]))
                true_l.append(int(yt[i]))
                if len(imgs_l) >= max_images:
                    break
            if len(imgs_l) >= max_images:
                break
    n = len(imgs_l)
    if n == 0:
        return
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows))
    if rows * cols == 1:
        axes = np.array([axes])
    axes_flat = np.atleast_1d(axes).ravel()
    for k in range(rows * cols):
        ax = axes_flat[k]
        if k < n:
            t = imgs_l[k]
            if t.shape[0] == 1:
                g = t[0].numpy()
            else:
                g = t.numpy().mean(axis=0)
            lo, hi = float(g.min()), float(g.max())
            if hi - lo > 1e-6:
                g = (g - lo) / (hi - lo + 1e-8)
            ax.imshow(np.stack([g, g, g], axis=-1), vmin=0, vmax=1)
            ok = "OK" if pred_l[k] == true_l[k] else "X"
            ax.set_title(f"{ok} pred={pred_l[k]} gt={true_l[k]} (nc={num_classes})")
        ax.axis("off")
    fig.suptitle("CXR classification (first eval batch samples)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


def _roi3(x: Any) -> Tuple[int, int, int]:
    if x is None:
        return (96, 96, 96)
    if isinstance(x, (list, tuple)) and len(x) >= 3:
        return (int(x[0]), int(x[1]), int(x[2]))
    if isinstance(x, (int, float)):
        v = int(x)
        return (v, v, v)
    return (96, 96, 96)

def _segmentation_visualize(
    *,
    cfg: dict,
    model: Any,
    device: Any,
    loader: Any,
    dataset: Any,
    merged_model: Dict[str, Any],
    n_classes: int,
    use_amp: bool,
    out_root: Path,
    viz_cfg: Dict[str, Any],
) -> None:
    import torch
    import torch.nn.functional as F
    from torch import amp as torch_amp

    from dinomim_pytorch.datasets.seg_dataset_factory import build_eval_fullvolume_transform
    from dinomim_pytorch.eval import logits_to_label_map, postprocess_enabled, resolve_postprocess_cfg
    from dinomim_pytorch.segmentation_inference import sliding_window_predict, sliding_window_inference
    from dinomim_pytorch.segmentation_vis import patch_axial_middle_slice, save_gray_pred_gt_panel, save_three_plane_panel

    pp_cfg = resolve_postprocess_cfg(cfg)
    use_pp = postprocess_enabled(cfg)

    if not viz_cfg.get("enabled", False):
        return

    ik = getattr(dataset, "image_key", "image")
    lk = getattr(dataset, "label_key", "label")
    patches_dir = out_root / "viz_patch"
    full_dir = out_root / "viz_full_volume"
    cases_full_dir = out_root / "full_volume" / "cases_full_volume"
    patches_dir.mkdir(parents=True, exist_ok=True)

    num_pb = max(1, int(viz_cfg.get("num_patch_batches", 4)))
    use_cuda_amp = use_amp and device.type == "cuda"
    pb_count = 0
    flat_patch_idx = 0

    with torch.no_grad():
        model.eval()
        for b_idx, batch in enumerate(loader):
            if pb_count >= num_pb:
                break
            x = batch[ik]
            y = batch[lk]
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x, dtype=torch.float32)
            if not isinstance(y, torch.Tensor):
                y = torch.as_tensor(y, dtype=torch.long)
            x = x.to(device)
            if y.dim() == 5 and y.size(1) == 1:
                y = y[:, 0]
            y = y.long().to(device)
            with torch_amp.autocast("cuda" if use_cuda_amp else "cpu", enabled=use_cuda_amp):
                from dinomim_pytorch.segmentation_models.losses import primary_segmentation_logits

                logits = primary_segmentation_logits(model(x))
            if logits.shape[-3:] != y.shape[-3:]:
                logits = F.interpolate(logits, size=y.shape[-3:], mode="trilinear", align_corners=False)

            pred_labels = logits_to_label_map(logits, n_classes, pp_cfg) if use_pp else None
            bs = int(x.shape[0])
            for j in range(bs):
                ga, pra, gta = patch_axial_middle_slice(
                    x.cpu(),
                    y.cpu(),
                    logits.cpu(),
                    j,
                    n_classes,
                    pred_labels_bdhw=pred_labels.cpu() if pred_labels is not None else None,
                )
                subtitle = f"patch batch{b_idx} item{j}"
                meta = batch.get("meta") if isinstance(batch, dict) else None
                if isinstance(meta, list) and j < len(meta) and isinstance(meta[j], dict):
                    path_hint = meta[j].get("path") or meta[j].get(ik) or ""
                    if path_hint:
                        subtitle = Path(str(path_hint)).name
                save_gray_pred_gt_panel(
                    patches_dir / f"patch_{flat_patch_idx:03d}_axial.png",
                    ga,
                    pra,
                    gta,
                    n_classes,
                    subtitle=subtitle,
                )
                flat_patch_idx += 1
            pb_count += 1

    n_full = int(viz_cfg.get("num_full_volume_cases", 0))
    if n_full <= 0:
        print(f"[eval] segmentation viz -> {out_root} (patch only)", flush=True)
        return
    if sliding_window_inference is None:
        print(
            "[eval] full-volume viz skipped (MONAI sliding_window_inference not available).",
            file=sys.stderr,
        )
        print(f"[eval] segmentation viz -> {out_root}", flush=True)
        return

    dcfg = dict((cfg.get("data") or {}))
    fv_tf = build_eval_fullvolume_transform(cfg)
    roi_tup = _roi3(merged_model.get("img_size") or dcfg.get("image_size"))
    overlap = float(viz_cfg.get("sw_overlap", 0.5))
    sw_bs = max(1, int(viz_cfg.get("sw_batch_size", 2)))
    full_dir.mkdir(parents=True, exist_ok=True)
    cases_full_dir.mkdir(parents=True, exist_ok=True)

    n_rows = len(getattr(dataset, "rows", ()))
    for vi in range(min(n_full, n_rows)):
        try:
            d0 = dataset._load_dict(dataset.rows[vi])  # type: ignore[arg-type]
            d1 = fv_tf(d0)
            xv = d1[ik].to(device).unsqueeze(0).float()
            yv_full = d1[lk].long()

            xv_in = xv
            yv_aligned = yv_full
            max_d = tuple(int(s) for s in xv.shape[2:])
            pad_need = tuple(max(0, roi_tup[i] - max_d[i]) for i in range(3))
            if any(p > 0 for p in pad_need):
                xv_in = F.pad(
                    xv,
                    (0, pad_need[2], 0, pad_need[1], 0, pad_need[0]),
                    mode="replicate",
                )
                yz = yv_full if isinstance(yv_full, torch.Tensor) else torch.as_tensor(yv_full, dtype=torch.long)
                yz = yz.unsqueeze(0).unsqueeze(0).float()
                yz_p = F.pad(
                    yz,
                    (0, pad_need[2], 0, pad_need[1], 0, pad_need[0]),
                    mode="constant",
                    value=0.0,
                )
                yv_aligned = yz_p.long().squeeze(0).squeeze(0)

            logits_f = sliding_window_predict(
                model, xv_in, roi_size=roi_tup, sw_batch_size=sw_bs, overlap=overlap, device=device
            )
            logits_f = logits_f.float()
            if logits_f.shape[-3:] != yv_aligned.shape[-3:]:
                logits_f = F.interpolate(logits_f, size=yv_aligned.shape[-3:], mode="trilinear", align_corners=False)

            xv_cpu = xv_in[0].detach().cpu()
            gt_cpu = yv_aligned if isinstance(yv_aligned, torch.Tensor) else torch.as_tensor(yv_aligned)
            gt_cpu = gt_cpu.long().detach().cpu() if isinstance(gt_cpu, torch.Tensor) else torch.as_tensor(gt_cpu).long()

            lf_cpu = logits_f[0].detach().cpu()
            pred_viz = (
                logits_to_label_map(logits_f, n_classes, pp_cfg)[0].detach().cpu()
                if use_pp
                else lf_cpu
            )
            row = dataset.rows[vi]
            path_cell = str(row.get(ik, "") or row.get("image", "") or "")
            stem = Path(path_cell).stem if path_cell else ""
            fname = "".join(ch for ch in stem if ch.isalnum() or ch in ("_", "-"))
            fname = fname[:80] if fname else f"idx{vi}"
            save_three_plane_panel(
                full_dir / f"fullvol_case{vi:02d}_{fname}.png",
                xv_cpu,
                pred_viz,
                gt_cpu,
                n_classes,
                title_prefix=f"case{vi} ",
            )
            save_three_plane_panel(
                cases_full_dir / f"case_{vi:02d}_{fname}.png",
                xv_cpu,
                pred_viz,
                gt_cpu,
                n_classes,
                title_prefix=f"case{vi} ",
            )
        except Exception as ex:  # noqa: BLE001
            print(f"[eval] full-volume viz case {vi} failed: {ex}", file=sys.stderr)

    print(
        f"[eval] segmentation viz -> {out_root} ({patches_dir.name}/ + {full_dir.name}/ + "
        f"{cases_full_dir.relative_to(out_root)}/)",
        flush=True,
    )


def run_seg_eval(args: argparse.Namespace, cfg: dict, *, compare: bool) -> None:
    import copy

    import torch

    from dinomim_pytorch.eval import postprocess_enabled, resolve_postprocess_cfg
    from dinomim_pytorch.segmentation_models import build_segmentation_model, get_merged_model_config

    if compare:
        if not args.ckpt_scratch or not args.ckpt_ssl:
            raise SystemExit("--ckpt_scratch and --ckpt_ssl are required for segmentation compare.")
    elif not args.checkpoint and not getattr(args, "checkpoint_list", None) and not getattr(args, "predictions_dir", None):
        raise SystemExit("Provide --checkpoint, --checkpoint_list, --predictions_dir, or both --ckpt_scratch and --ckpt_ssl.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    merged = get_merged_model_config(cfg)
    n_classes = int((cfg.get("data") or {}).get("num_classes") or merged.get("out_channels", 2))
    # Segmentation metrics/HD95 are evaluated in full precision.  CUDA fp16
    # autocast can collapse UNETR/ViT logits during eval even when training is
    # stable, producing all-background predictions and nan HD95.
    use_amp = False

    out_dir = Path(args.output_dir)
    cfg_build = copy.deepcopy(cfg)
    m_build = cfg_build.setdefault("model", {})
    if m_build.get("ssl_init") or m_build.get("ssl_checkpoint"):
        print(
            "[eval] skipping model.ssl_init / ssl_checkpoint for model build "
            "(weights loaded from --checkpoint).",
            flush=True,
        )
    m_build["ssl_init"] = False
    m_build.pop("ssl_checkpoint", None)

    if getattr(args, "official_npz", False):
        from dinomim_pytorch.eval.inference_tta import parse_tta_axes
        from dinomim_pytorch.eval.official_npz import evaluate_official_npz

        use_tta = bool(getattr(args, "tta", False) or getattr(args, "tta_mirror", False))
        tta_axes = parse_tta_axes(str(getattr(args, "tta_axes", "0,1,2")))
        tta_mode = str(getattr(args, "tta_mode", "mirror"))

        data_cfg = dict((cfg or {}).get("data") or {})
        loader = str(data_cfg.get("loader", "")).strip().lower()
        dataset = str(data_cfg.get("dataset_name", data_cfg.get("task", ""))).strip().lower()
        is_synapse_nnformer_npz = loader == "nnformer_npz" and dataset in (
            "synapse",
            "task002",
            "task002_synapse",
            "btcv",
        )
        if use_tta and is_synapse_nnformer_npz:
            print(
                "[warning] Mirror TTA was empirically harmful for this UNETR++ Synapse nnFormer-npz "
                "model. Use only for diagnostics.",
                flush=True,
            )

        def _official_kwargs(sub: Path) -> dict:
            return dict(
                cfg=cfg,
                device=device,
                out_dir=sub,
                tta_mirror=use_tta,
                tta_axes=tta_axes,
                tta_mode=tta_mode,
                max_cases=getattr(args, "max_cases", None),
                eval_split=str(getattr(args, "eval_split", "val")),
                overlap=getattr(args, "overlap", None),
                sw_batch_size=getattr(args, "sw_batch_size", None),
                save_predictions=not bool(getattr(args, "no_save_predictions", False)),
                save_probabilities=bool(getattr(args, "save_probabilities", False)),
                predictions_dir=Path(args.predictions_dir) if getattr(args, "predictions_dir", None) else None,
                run_sanity_checks=bool(getattr(args, "sanity_checks", True)),
            )

        if getattr(args, "predictions_dir", None):
            payload = evaluate_official_npz(None, **_official_kwargs(out_dir))
            agg = payload["aggregate"]
            class_names = [c["name"] for c in payload["classes"]]
            dice_bits = " ".join(f"{n}={agg[f'Dice_{n}']:.4f}" for n in class_names)
            print(
                f"[official_npz] scored predictions_dir={args.predictions_dir} "
                f"n={int(agg['n_cases'])} | {dice_bits} | "
                f"Dice_mean={agg['Dice_mean']:.4f} HD95_mean={agg['HD95_mean']:.3f} mm",
                flush=True,
            )
            return

        def _build_ensemble_models(ckpt_paths: List[str]) -> List[Any]:
            models = []
            for ck in ckpt_paths:
                m = build_segmentation_model(cfg_build).to(device)
                _load_seg_state_dict(m, ck)
                models.append(m)
            return models

        def _eval_official_one(ckpt_paths: List[str], subdir: Optional[Path]) -> dict:
            sub = (out_dir / subdir) if subdir is not None else out_dir
            models = _build_ensemble_models(ckpt_paths)
            model_arg: Any = models[0] if len(models) == 1 else models
            return evaluate_official_npz(model_arg, **_official_kwargs(sub))

        if compare:
            p_scratch = _eval_official_one([args.ckpt_scratch], Path("scratch"))
            p_ssl = _eval_official_one([args.ckpt_ssl], Path("ssl"))
            agg_a = p_scratch["aggregate"]
            agg_b = p_ssl["aggregate"]
            cmp_a = {k: float(v) for k, v in agg_a.items() if isinstance(v, (int, float, np.floating))}
            cmp_b = {k: float(v) for k, v in agg_b.items() if isinstance(v, (int, float, np.floating))}
            _write_compare_csv(cmp_a, cmp_b, out_dir / "metrics_official_compare.csv")
            summary = {
                "scratch_Dice_mean": agg_a["Dice_mean"],
                "ssl_Dice_mean": agg_b["Dice_mean"],
                "delta_Dice_mean": agg_b["Dice_mean"] - agg_a["Dice_mean"],
                "scratch_HD95_mean": agg_a["HD95_mean"],
                "ssl_HD95_mean": agg_b["HD95_mean"],
                "delta_HD95_mean": agg_b["HD95_mean"] - agg_a["HD95_mean"],
            }
            _save_metrics(summary | {f"scratch_{k}": v for k, v in cmp_a.items()} | {f"ssl_{k}": v for k, v in cmp_b.items()}, out_dir)
            class_names = [c["name"] for c in p_scratch["classes"]]
            dice_bits = " ".join(
                f"{n}={agg_b[f'Dice_{n}']:.4f}({agg_b[f'Dice_{n}']-agg_a[f'Dice_{n}']:+.4f})"
                for n in class_names
            )
            print(
                f"[official_npz compare] scratch Dice_mean={agg_a['Dice_mean']:.4f} "
                f"ssl Dice_mean={agg_b['Dice_mean']:.4f} delta={agg_b['Dice_mean']-agg_a['Dice_mean']:+.4f} | "
                f"{dice_bits}",
                flush=True,
            )
            print("Wrote", out_dir / "metrics_official_compare.csv", out_dir / "metrics.json", flush=True)
            return

        ckpt_paths = list(getattr(args, "checkpoint_list", None) or [])
        if not ckpt_paths:
            if not args.checkpoint:
                raise SystemExit("--checkpoint or --checkpoint_list required for official_npz inference")
            ckpt_paths = [args.checkpoint]
        payload = _eval_official_one(ckpt_paths, None)
        agg = payload["aggregate"]
        class_names = [c["name"] for c in payload["classes"]]
        dice_bits = " ".join(f"{n}={agg[f'Dice_{n}']:.4f}" for n in class_names)
        print(
            f"[official_npz] split={payload['eval_split']} task={payload['task']} "
            f"n={int(agg['n_cases'])} fold={payload['fold']} tta={payload['tta_mirror']} | "
            f"{dice_bits} | Dice_mean={agg['Dice_mean']:.4f} HD95_mean={agg['HD95_mean']:.3f} mm",
            flush=True,
        )
        return

    loader = _make_seg_loader(cfg, train=False, index_csv=args.index_csv)
    if loader is None or len(loader) == 0:
        print(
            "[eval] ERROR: no segmentation val loader. "
            "Set data.index_val, data.loader=nnformer_npz (+ nnformer_preprocessed_dir), or --index_csv.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # ``--checkpoint`` is a finetuned ``last_model.pt`` / ``best_model.pt`` with full weights.
    # Do not re-run ``model.ssl_checkpoint`` init (path is often missing on other machines / stale).
    viz_cfg = _eval_vis_merged(cfg, args)
    dataset_obj = loader.dataset
    ds_cfg = getattr(dataset_obj, "cfg", {}) or {}
    ik = str(
        getattr(dataset_obj, "image_key", None)
        or ds_cfg.get("image_key")
        or (cfg.get("data") or {}).get("image_key")
        or "image"
    )
    lk = str(
        getattr(dataset_obj, "label_key", None)
        or ds_cfg.get("label_key")
        or (cfg.get("data") or {}).get("label_key")
        or "label"
    )
    spacing_mm = _voxel_spacing_mm_for_hd(cfg)

    metrics_pp = _metrics_use_postprocess(cfg)
    viz_pp = postprocess_enabled(cfg)
    print(
        "[eval] segmentation metrics/visualization run in full precision"
        + f"; metrics_postprocess={'on' if metrics_pp else 'off'}"
        + (
            f"; viz_postprocess={resolve_postprocess_cfg(cfg)}"
            if viz_pp
            else "; viz_postprocess=off"
        ),
        flush=True,
    )

    def _eval_one_ckpt(state_path: str, viz_rel: Optional[Path] = None) -> Dict[str, float]:
        m = build_segmentation_model(cfg_build).to(device)
        _load_seg_state_dict(m, state_path)
        metrics = _evaluate_segmentation(
            m,
            loader,
            device,
            n_classes,
            use_amp=use_amp,
            image_key=ik,
            label_key=lk,
            spacing_mm=spacing_mm,
            cfg=cfg,
        )
        out_viz = (out_dir / viz_rel) if viz_rel is not None else out_dir
        _segmentation_visualize(
            cfg=cfg,
            model=m,
            device=device,
            loader=loader,
            dataset=dataset_obj,
            merged_model=dict(merged),
            n_classes=n_classes,
            use_amp=use_amp,
            out_root=out_viz,
            viz_cfg=viz_cfg,
        )
        return metrics

    if compare:
        m_scratch = _eval_one_ckpt(args.ckpt_scratch, Path("scratch"))
        m_ssl = _eval_one_ckpt(args.ckpt_ssl, Path("ssl"))
        print("scratch metrics:", m_scratch)
        print("ssl metrics:", m_ssl)
        _write_compare_csv(m_scratch, m_ssl, out_dir / "metrics_compare.csv")
        _save_metrics({"scratch_" + k: v for k, v in m_scratch.items()} | {"ssl_" + k: v for k, v in m_ssl.items()}, out_dir)
        print("Wrote", out_dir / "metrics_compare.csv", out_dir / "metrics.json")
        return

    metrics = _eval_one_ckpt(args.checkpoint, None)
    print("metrics:", metrics)
    _save_metrics(metrics, out_dir)
    print("Wrote", out_dir / "metrics.json")


def run_cxr_eval(args: argparse.Namespace, cfg: dict, *, compare: bool) -> None:
    import torch
    from torch.utils.data import DataLoader

    from dinomim_pytorch.downstream_classification import build_classification_model

    if compare:
        if not args.ckpt_scratch or not args.ckpt_ssl:
            raise SystemExit("CXR compare needs --ckpt_scratch and --ckpt_ssl.")
    elif not args.checkpoint:
        raise SystemExit("Provide --checkpoint or both scratch/ssl checkpoints.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nc = int((cfg.get("model") or {}).get("num_classes", 2))
    bs = int((cfg.get("training") or {}).get("batch_size", 16))
    nw = int((cfg.get("training") or {}).get("num_workers", 0))
    use_amp = bool((cfg.get("training") or {}).get("mixed_precision", False)) and device.type == "cuda"

    try:
        ds = _build_cxr_dataset(cfg, args.csv or None)
    except ValueError as e:
        print(f"[eval] ERROR: {e}", file=sys.stderr)
        raise SystemExit(1) from e
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw)

    out_dir = Path(args.output_dir)
    viz_cfg = _eval_vis_merged(cfg, args)

    def _run_ckpt(path: str, cxr_vis_root: Optional[Path] = None) -> Dict[str, Any]:
        m = build_classification_model(cfg).to(device)
        _load_seg_state_dict(m, path)
        met = _evaluate_classification(m, loader, device, nc, use_amp=use_amp)
        root = cxr_vis_root if cxr_vis_root is not None else out_dir
        if viz_cfg.get("cxr_visualize"):
            _save_cxr_pred_grid(
                m,
                loader,
                device,
                root / "viz_cxr" / "pred_grid.png",
                max_images=max(1, int(viz_cfg.get("num_cxr_images", 12))),
                num_classes=nc,
                use_amp=use_amp,
            )
            print(f"[eval] CXR viz -> {root / 'viz_cxr' / 'pred_grid.png'}", flush=True)
        return met

    if compare:
        a = _run_ckpt(args.ckpt_scratch, out_dir / "scratch")
        b = _run_ckpt(args.ckpt_ssl, out_dir / "ssl")
        print("scratch metrics:", {k: v for k, v in a.items() if isinstance(v, (int, float))})
        print("ssl metrics:", {k: v for k, v in b.items() if isinstance(v, (int, float))})
        cmp_a = {k: float(v) for k, v in a.items() if isinstance(v, (int, float, np.floating))}
        cmp_b = {k: float(v) for k, v in b.items() if isinstance(v, (int, float, np.floating))}
        _write_compare_csv(cmp_a, cmp_b, out_dir / "metrics_compare.csv")
        _save_metrics({f"scratch_{k}": v for k, v in a.items() if isinstance(v, (int, float, list))} |
                      {f"ssl_{k}": v for k, v in b.items()}, out_dir)
        print("Wrote", out_dir / "metrics_compare.csv")
        return

    metrics = _run_ckpt(args.checkpoint, None)
    flat = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    print("metrics:", flat)
    _save_metrics(metrics, out_dir)
    print("Wrote", out_dir / "metrics.json")


def main() -> None:
    from dinomim_pytorch.config_utils import load_yaml

    p = argparse.ArgumentParser(description="DINO_MIM unified eval: MRI/CT segmentation, CXR classification.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--modality", type=str, default="auto", help="auto | mri | ct | cxr")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--ckpt_scratch", type=str, default=None)
    p.add_argument("--ckpt_ssl", type=str, default=None)
    p.add_argument("--index_csv", type=str, default=None, help="Segmentation val/test CSV override.")
    p.add_argument("--csv", type=str, default=None, help="CXR CSV override (defaults to YAML data paths).")
    p.add_argument("--output_dir", type=str, default="outputs/eval")
    p.add_argument(
        "--no_vis",
        action="store_true",
        help="Segmentation: disable patch + full-volume PNG exports.",
    )
    p.add_argument(
        "--official_npz",
        action="store_true",
        help="Paper-style full-volume sliding-window eval on all fold val cases "
        "(Synapse: 8-organ Dice+HD95; ACDC: RV/MYO/LV). Uses splits_final.pkl + nnformer npz.",
    )
    p.add_argument(
        "--eval_split",
        type=str,
        default="val",
        choices=["val"],
        help="(with --official_npz) Eval split; currently only 'val' is supported.",
    )
    p.add_argument(
        "--tta_mirror",
        action="store_true",
        help="(with --official_npz) Mirror TTA diagnostic only (alias: --tta).",
    )
    p.add_argument(
        "--tta",
        action="store_true",
        help="(with --official_npz) Mirror TTA diagnostic only; harmful on Synapse nnFormer-npz UNETR++.",
    )
    p.add_argument(
        "--tta_axes",
        type=str,
        default="0,1,2",
        help="(with --official_npz) Spatial axes for mirror TTA: 0=D,1=H,2=W.",
    )
    p.add_argument(
        "--tta_mode",
        type=str,
        default="mirror",
        choices=["mirror"],
        help="(with --official_npz) TTA mode.",
    )
    p.add_argument(
        "--overlap",
        type=float,
        default=None,
        help="(with --official_npz) Sliding-window overlap fraction (default from config or 0.5).",
    )
    p.add_argument(
        "--sw_batch_size",
        type=int,
        default=None,
        help="(with --official_npz) Sliding-window batch size override.",
    )
    p.add_argument(
        "--checkpoint_list",
        nargs="+",
        default=None,
        help="(with --official_npz) Ensemble multiple checkpoints (probability average).",
    )
    p.add_argument(
        "--predictions_dir",
        type=str,
        default=None,
        help="(with --official_npz) Score existing saved predictions instead of running inference.",
    )
    p.add_argument(
        "--no_save_predictions",
        action="store_true",
        help="(with --official_npz) Do not write predictions/ under output_dir.",
    )
    p.add_argument(
        "--save_probabilities",
        action="store_true",
        help="(with --official_npz) Save per-case softmax probability volumes in predictions/*.npz.",
    )
    p.add_argument(
        "--sanity_checks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="(with --official_npz) Run first-case sanity checks.",
    )
    p.add_argument(
        "--max_cases",
        type=int,
        default=None,
        help="(with --official_npz) Limit val cases (debug). Default: all fold val cases.",
    )
    p.add_argument(
        "--vis",
        action="store_true",
        help="Force eval_vis.enabled true (segmentation PNGs).",
    )
    p.add_argument("--num_patch_vis", type=int, default=None, help="Max val loader batches for patch PNGs.")
    p.add_argument(
        "--num_full_volume_vis",
        type=int,
        default=None,
        help="Number of CSV rows for full-volume sliding-window slice montages (0=patch only).",
    )
    p.add_argument(
        "--cxr_vis",
        action="store_true",
        help="CXR: write viz_cxr/pred_grid.png (first N images; also set eval_vis.cxr_visualize in YAML).",
    )
    args = p.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.is_file():
        print(f"[eval] config not found: {cfg_path}", file=sys.stderr)
        raise SystemExit(1)

    cfg = load_yaml(str(cfg_path)) or {}
    out_root = Path(args.output_dir).expanduser().resolve()
    args.output_dir = str(out_root)
    print(f"[eval] output_dir={out_root}", flush=True)
    modality = resolve_modality(cfg, args.modality)
    compare = bool(args.ckpt_scratch and args.ckpt_ssl)

    if modality == "cxr":
        run_cxr_eval(args, cfg, compare=compare)
    elif modality in ("mri", "ct"):
        run_seg_eval(args, cfg, compare=compare)
    else:
        print(f"[eval] unknown modality {modality!r}; use --modality mri|ct|cxr", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
