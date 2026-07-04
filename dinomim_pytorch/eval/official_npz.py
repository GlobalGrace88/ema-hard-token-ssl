"""
UNETR++ ``inference_synapse.py`` / ``inference_acdc.py``-style full-volume eval.

Per-case sliding-window inference on nnFormer ``.npz`` volumes; per-organ Dice + HD95
on fold val (or challenge test). Matches MAE_BYOL ``official_npz_eval.py`` conventions:

- Dice = 1 when both pred and GT empty for a class.
- HD95 = 0 mm when either pred or GT empty.
- Synapse paper table uses **8 organs** (not all 13 foreground classes).
"""
from __future__ import annotations

import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dinomim_pytorch.datasets.nnformer_npz import resolve_nnformer_npz_dir
from dinomim_pytorch.datasets.nnformer_splits import load_splits_final, resolve_splits_pkl
from dinomim_pytorch.eval.inference_tta import mirror_tta_flip_dims, parse_tta_axes
from dinomim_pytorch.segmentation_models.losses import primary_segmentation_logits

SYNAPSE_8_ORGANS: List[Tuple[int, str]] = [
    (1, "spleen"),
    (2, "right_kidney"),
    (3, "left_kidney"),
    (4, "gallbladder"),
    (6, "liver"),
    (7, "stomach"),
    (8, "aorta"),
    (11, "pancreas"),
]

ACDC_3_CLASSES: List[Tuple[int, str]] = [
    (1, "RV"),
    (2, "MYO"),
    (3, "LV"),
]

# BraTS / Task003 tumor: region Dice (not single-label equality).
BRAATS_3_REGIONS: List[Tuple[int, str]] = [
    (1, "WT"),
    (2, "TC"),
    (3, "ET"),
]

TASK_CLASS_SPECS: Dict[str, List[Tuple[int, str]]] = {
    "synapse": SYNAPSE_8_ORGANS,
    "task002": SYNAPSE_8_ORGANS,
    "task002_synapse": SYNAPSE_8_ORGANS,
    "btcv": SYNAPSE_8_ORGANS,
    "acdc": ACDC_3_CLASSES,
    "task001": ACDC_3_CLASSES,
    "task001_acdc": ACDC_3_CLASSES,
    "brats": BRAATS_3_REGIONS,
    "tumor": BRAATS_3_REGIONS,
    "task003": BRAATS_3_REGIONS,
    "task003_tumor": BRAATS_3_REGIONS,
}


@dataclass
class OfficialInferenceConfig:
    roi_size: Tuple[int, int, int]
    overlap: float = 0.5
    sw_batch_size: int = 1
    mode: str = "gaussian"
    tta: bool = False
    tta_axes: Tuple[int, ...] = (0, 1, 2)
    tta_mode: str = "mirror"
    save_predictions: bool = True
    save_probabilities: bool = False
    checkpoint_paths: Tuple[str, ...] = tuple()


def _normalize_task(task: str) -> str:
    t = str(task).strip().lower()
    if t in ("task002", "task002_synapse", "synapse", "btcv"):
        return "synapse"
    if t in ("task001", "task001_acdc", "acdc"):
        return "acdc"
    if t in ("task003", "task003_tumor", "tumor", "brats", "msd_brain"):
        return "brats"
    return t


def class_spec_for_task(task: str) -> List[Tuple[int, str]]:
    key = _normalize_task(task)
    if key not in TASK_CLASS_SPECS:
        raise ValueError(f"Unsupported task {task!r} for official_npz (use synapse, acdc, or tumor/brats)")
    return TASK_CLASS_SPECS[key]


def resolve_official_inference_config(
    cfg: Mapping[str, Any],
    *,
    overlap: Optional[float] = None,
    sw_batch_size: Optional[int] = None,
    roi_size: Optional[Sequence[int]] = None,
    tta: Optional[bool] = None,
    tta_axes: Optional[Sequence[int]] = None,
    tta_mode: Optional[str] = None,
    checkpoint_paths: Optional[Sequence[str]] = None,
) -> OfficialInferenceConfig:
    data = dict(cfg.get("data") or {})
    merged_model = dict(cfg.get("model") or {})
    infer = dict(cfg.get("inference") or {})
    eval_vis = dict(cfg.get("eval_vis") or {})

    roi = roi_size or infer.get("roi_size") or data.get("image_size") or merged_model.get("img_size") or (64, 128, 128)
    roi_t = (int(roi[0]), int(roi[1]), int(roi[2]))
    ov = float(overlap if overlap is not None else infer.get("overlap", eval_vis.get("full_volume_overlap", eval_vis.get("sw_overlap", 0.5))))
    sb = int(sw_batch_size if sw_batch_size is not None else infer.get("sw_batch_size", eval_vis.get("sw_batch_size", 1)))
    mode = str(infer.get("mode", "gaussian"))
    use_tta = bool(tta if tta is not None else infer.get("tta", False))
    axes_raw = infer.get("tta_axes", "0,1,2")
    if tta_axes is not None:
        axes = tuple(int(a) for a in tta_axes)
    elif isinstance(axes_raw, str):
        axes = parse_tta_axes(axes_raw)
    else:
        axes = tuple(int(a) for a in axes_raw)
    tmode = str(tta_mode if tta_mode is not None else infer.get("tta_mode", "mirror"))
    ckpts = tuple(str(p) for p in (checkpoint_paths or infer.get("checkpoint_list") or []))
    return OfficialInferenceConfig(
        roi_size=roi_t,
        overlap=ov,
        sw_batch_size=sb,
        mode=mode,
        tta=use_tta,
        tta_axes=axes,
        tta_mode=tmode,
        checkpoint_paths=ckpts,
    )


def _window_starts(dim: int, win: int, stride: int) -> List[int]:
    if dim <= win:
        return [0]
    starts = list(range(0, dim - win + 1, stride))
    last = dim - win
    if not starts:
        return [0]
    if starts[-1] != last:
        starts.append(last)
    return starts


class _PrimaryLogitsWrapper(nn.Module):
    """Wrap UNETR++ deep-supervision models for sliding-window inference."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return primary_segmentation_logits(self.model(x))


class _EnsemblePrimaryLogits(nn.Module):
    """Average primary logits from multiple checkpoints for MONAI sliding-window."""

    def __init__(self, models: Sequence[nn.Module]) -> None:
        super().__init__()
        self.nets = nn.ModuleList([_PrimaryLogitsWrapper(m) for m in models])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.nets[0](x)
        for net in self.nets[1:]:
            out = out + net(x)
        return out / float(len(self.nets))


def _tta_flips(infer_cfg: OfficialInferenceConfig) -> List[Tuple[int, ...]]:
    if not infer_cfg.tta:
        return [tuple()]
    return mirror_tta_flip_dims(tensor_ndim=5, spatial_axes=infer_cfg.tta_axes, tta_mode=infer_cfg.tta_mode)


@torch.no_grad()
def _sliding_window_logits_monai(
    model: nn.Module,
    x_bcdhw: torch.Tensor,
    roi_size: Sequence[int],
    *,
    overlap: float = 0.5,
    sw_batch_size: int = 1,
    mode: str = "gaussian",
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    MONAI sliding-window on logits with replicate pad when volume < roi (matches eval.py viz).
    Returns ``(logits [1,K,D,H,W], original_spatial (D,H,W))``.
    """
    from dinomim_pytorch.segmentation_inference import sliding_window_predict

    device = device or next(model.parameters()).device
    vol = x_bcdhw.to(device, dtype=torch.float32)
    _, _, d0, h0, w0 = vol.shape
    rd, rh, rw = (int(roi_size[0]), int(roi_size[1]), int(roi_size[2]))
    pad_d = max(0, rd - d0)
    pad_h = max(0, rh - h0)
    pad_w = max(0, rw - w0)
    if pad_d or pad_h or pad_w:
        vol = F.pad(vol, (0, pad_w, 0, pad_h, 0, pad_d), mode="replicate")

    logits = sliding_window_predict(
        model,
        vol,
        roi_size=(rd, rh, rw),
        sw_batch_size=int(sw_batch_size),
        overlap=float(overlap),
        mode=str(mode),
        device=device,
    )
    logits = logits[:, :, :d0, :h0, :w0]
    return logits, (d0, h0, w0)


def _flip_logits_bcdhw(logits: torch.Tensor, flip_dims: Tuple[int, ...]) -> torch.Tensor:
    if not flip_dims:
        return logits
    return torch.flip(logits, dims=tuple(int(d) for d in flip_dims))


def _permute_mirror_logits_channels(
    logits_kdhw: torch.Tensor,
    flip_dims: Tuple[int, ...],
    *,
    task: str,
) -> torch.Tensor:
    """Swap left/right kidney logits after W-axis mirror (Synapse only)."""
    if not flip_dims or 4 not in flip_dims:
        return logits_kdhw
    task_key = str(task).strip().lower()
    if task_key not in ("synapse", "task002", "task002_synapse", "btcv"):
        return logits_kdhw
    out = logits_kdhw.clone()
    out[2], out[3] = logits_kdhw[3], logits_kdhw[2]
    return out


@torch.no_grad()
def sliding_window_predict_official(
    model: Union[nn.Module, Sequence[nn.Module]],
    x_bcdhw: torch.Tensor,
    roi_size: Sequence[int],
    *,
    overlap: float = 0.5,
    sw_batch_size: int = 1,
    mode: str = "gaussian",
    device: Optional[torch.device] = None,
    tta_mirror: bool = False,
    tta_axes: Tuple[int, ...] = (0, 1, 2),
    tta_mode: str = "mirror",
    tta_task: str = "synapse",
    return_probabilities: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Dense sliding-window inference; returns ``(D,H,W)`` int32 label map."""
    models = [model] if isinstance(model, nn.Module) else list(model)
    device = device or next(models[0].parameters()).device
    ensemble = _EnsemblePrimaryLogits(models).to(device).eval()

    infer_cfg = OfficialInferenceConfig(
        roi_size=(int(roi_size[0]), int(roi_size[1]), int(roi_size[2])),
        overlap=float(overlap),
        sw_batch_size=int(sw_batch_size),
        mode=str(mode),
        tta=bool(tta_mirror),
        tta_axes=tta_axes,
        tta_mode=tta_mode,
    )
    flips = _tta_flips(infer_cfg)

    vol = x_bcdhw.to(device, dtype=torch.float32)
    logits_terms: List[torch.Tensor] = []
    orig_shape: Optional[Tuple[int, int, int]] = None
    for flip_dims in flips:
        x_aug = torch.flip(vol, flip_dims) if flip_dims else vol
        logits_aug, orig_shape = _sliding_window_logits_monai(
            ensemble,
            x_aug,
            infer_cfg.roi_size,
            overlap=infer_cfg.overlap,
            sw_batch_size=infer_cfg.sw_batch_size,
            mode=infer_cfg.mode,
            device=device,
        )
        logits_aug = _flip_logits_bcdhw(logits_aug, flip_dims)
        logits_kdhw = _permute_mirror_logits_channels(
            logits_aug[0], flip_dims, task=tta_task
        )
        logits_terms.append(logits_kdhw)

    logits_final = logits_terms[0] if len(logits_terms) == 1 else torch.stack(logits_terms, dim=0).mean(dim=0)
    assert orig_shape is not None
    d0, h0, w0 = orig_shape
    if tuple(logits_final.shape[-3:]) != (d0, h0, w0):
        logits_final = logits_final[:, :d0, :h0, :w0]

    prob_final = F.softmax(logits_final, dim=0).cpu().numpy()
    pred = torch.argmax(logits_final, dim=0).cpu().numpy().astype(np.int32)
    if return_probabilities:
        return pred, prob_final.astype(np.float32, copy=False)
    return pred


def load_npz_case(
    npz_dir: Path,
    case_id: str,
    data_cfg: Optional[Mapping[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """Load preprocessed case: ``(image[C,D,H,W], seg[D,H,W], spacing_mm)``."""
    npz_path = npz_dir / f"{case_id}.npz"
    pkl_path = npz_dir / f"{case_id}.pkl"
    npy_path = npz_dir / f"{case_id}.npy"
    if npy_path.is_file():
        data = np.load(str(npy_path))
    else:
        with np.load(str(npz_path)) as z:
            data = np.asarray(z["data"])
    image = data[:-1].astype(np.float32, copy=False)
    seg = data[-1].astype(np.int64, copy=False)
    cfg = dict(data_cfg or {})
    if bool(cfg.get("nnformer_map_ignore_label", True)):
        ignore = int(cfg.get("nnformer_ignore_label", -1))
        seg[seg == ignore] = 0
    else:
        seg[seg < 0] = 0
    with open(pkl_path, "rb") as fh:
        props = pickle.load(fh)
    sp = props.get("spacing_after_resampling", props.get("original_spacing", (1.0, 1.0, 1.0)))
    spacing = (float(sp[0]), float(sp[1]), float(sp[2]))
    return image, seg, spacing


def save_case_prediction(
    out_path: Path,
    *,
    case_id: str,
    prediction: np.ndarray,
    spacing_mm: Sequence[float],
    probabilities: Optional[np.ndarray] = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_id": str(case_id),
        "prediction": np.asarray(prediction, dtype=np.int32),
        "spacing_mm": np.asarray(spacing_mm, dtype=np.float32),
    }
    if probabilities is not None:
        payload["probabilities"] = np.asarray(probabilities, dtype=np.float32)
    if out_path.suffix == ".npy":
        np.save(out_path, payload["prediction"])
        meta_path = out_path.with_suffix(".meta.npz")
        np.savez_compressed(meta_path, spacing_mm=payload["spacing_mm"], case_id=str(case_id))
        return
    np.savez_compressed(out_path, **payload)


def load_case_prediction(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float], str]:
    path = Path(path)
    if path.suffix == ".npy":
        pred = np.load(str(path)).astype(np.int32)
        meta = path.with_suffix(".meta.npz")
        spacing = (1.0, 1.0, 1.0)
        case_id = path.stem
        if meta.is_file():
            with np.load(str(meta)) as z:
                if "spacing_mm" in z:
                    sp = z["spacing_mm"]
                    spacing = (float(sp[0]), float(sp[1]), float(sp[2]))
                if "case_id" in z:
                    case_id = str(z["case_id"])
        return pred, spacing, case_id
    if path.suffix == ".gz" and path.name.endswith(".nii.gz"):
        import nibabel as nib  # type: ignore

        img = nib.load(str(path))
        pred = np.asarray(img.dataobj).astype(np.int32)
        zooms = img.header.get_zooms()[:3]
        spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
        return pred, spacing, path.name.replace(".nii.gz", "")
    with np.load(str(path)) as z:
        pred = np.asarray(z["prediction"] if "prediction" in z else z["seg"]).astype(np.int32)
        sp = z.get("spacing_mm", np.array([1.0, 1.0, 1.0], dtype=np.float32))
        spacing = (float(sp[0]), float(sp[1]), float(sp[2]))
        case_id = str(z.get("case_id", path.stem))
    return pred, spacing, case_id


def _dice_official(pred: np.ndarray, gt: np.ndarray) -> float:
    p_sum = float(pred.sum())
    g_sum = float(gt.sum())
    if p_sum + g_sum == 0.0:
        return 1.0
    inter = float(np.logical_and(pred, gt).sum())
    return 2.0 * inter / (p_sum + g_sum)


def _hd95_official(pred: np.ndarray, gt: np.ndarray, spacing_mm: Sequence[float]) -> float:
    from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

    if not pred.any() or not gt.any():
        return 0.0

    def _surface(mask: np.ndarray) -> np.ndarray:
        struct = generate_binary_structure(mask.ndim, 1)
        eroded = binary_erosion(mask, structure=struct, border_value=0)
        surf = np.logical_xor(mask, eroded)
        return mask if not surf.any() else surf

    sp = np.asarray(spacing_mm, dtype=np.float64)
    ps, gs = _surface(pred), _surface(gt)
    dt_p = distance_transform_edt(~ps, sampling=sp)
    dt_g = distance_transform_edt(~gs, sampling=sp)
    dists = np.concatenate([dt_p[gs], dt_g[ps]], axis=0)
    return float(np.percentile(dists, 95))


def _binary_masks(label_map: np.ndarray, class_spec: Sequence[Tuple[int, str]]) -> Dict[str, np.ndarray]:
    lab = np.asarray(label_map)
    return {name: (lab == idx) for idx, name in class_spec}


def _brats_region_masks(label_map: np.ndarray) -> Dict[str, np.ndarray]:
    """BraTS labels {0,1,2,3}: WT=any tumor, TC={1,3}, ET={3} (matches training metrics)."""
    lab = np.asarray(label_map)
    return {
        "WT": lab > 0,
        "TC": (lab == 1) | (lab == 3),
        "ET": lab == 3,
    }


def _region_masks(label_map: np.ndarray, norm_task: str, class_spec: Sequence[Tuple[int, str]]) -> Dict[str, np.ndarray]:
    if norm_task == "brats":
        return _brats_region_masks(label_map)
    return _binary_masks(label_map, class_spec)


def _val_case_ids(cfg: Mapping[str, Any], npz_dir: Path) -> List[str]:
    data = dict(cfg.get("data") or {})
    pkl = resolve_splits_pkl(data, str(data.get("dataset_name", "")))
    if pkl is None:
        raise FileNotFoundError("official_npz requires data.splits_pkl (or nnformer_preprocessed_dir with splits_final.pkl)")
    fold = int(data.get("fold", data.get("nnformer_fold", 0)))
    _, val_ids = load_splits_final(pkl, fold=fold)
    present: List[str] = []
    missing: List[str] = []
    for cid in val_ids:
        if (npz_dir / f"{cid}.npz").is_file() or (npz_dir / f"{cid}.npy").is_file():
            present.append(cid)
        else:
            missing.append(cid)
    if missing:
        print(
            f"[official_npz] WARNING: {len(missing)}/{len(val_ids)} val cases missing from {npz_dir} "
            f"(first: {missing[0]})",
            flush=True,
        )
    if not present:
        raise FileNotFoundError(f"No val cases found under {npz_dir} for fold {fold}")
    return present


def _sanity_check_prediction(
    pred: np.ndarray,
    image_shape: Tuple[int, ...],
    prob: Optional[np.ndarray],
    *,
    max_label: int,
    case_id: str,
) -> None:
    if tuple(pred.shape) != tuple(image_shape):
        raise ValueError(f"[sanity] {case_id}: pred shape {pred.shape} != image {image_shape}")
    if not np.isfinite(pred).all():
        raise ValueError(f"[sanity] {case_id}: non-finite values in prediction")
    uniq = np.unique(pred)
    if int(uniq.max(initial=0)) > int(max_label):
        print(f"[sanity] WARNING {case_id}: labels up to {int(uniq.max())} (expected <= {max_label})", flush=True)
    if prob is not None:
        if not np.isfinite(prob).all():
            raise ValueError(f"[sanity] {case_id}: NaN/Inf in probability map")
        sums = prob.sum(axis=0)
        if not np.allclose(sums, 1.0, atol=1e-2, rtol=0.0):
            bad = float(np.max(np.abs(sums - 1.0)))
            print(f"[sanity] WARNING {case_id}: softmax sum deviates by up to {bad:.4f}", flush=True)


def compute_case_official_metrics(
    pred_lab: np.ndarray,
    seg_gt: np.ndarray,
    spacing: Sequence[float],
    class_spec: Sequence[Tuple[int, str]],
    *,
    case_id: str,
    norm_task: str,
) -> Dict[str, Any]:
    pred_masks = _region_masks(pred_lab, norm_task, class_spec)
    gt_masks = _region_masks(seg_gt, norm_task, class_spec)
    case_metrics: Dict[str, Any] = {"case_id": case_id, "spacing_mm": list(spacing)}
    dice_vals: List[float] = []
    hd_vals: List[float] = []
    for _, name in class_spec:
        d = _dice_official(pred_masks[name], gt_masks[name])
        h = _hd95_official(pred_masks[name], gt_masks[name], spacing)
        case_metrics[f"Dice_{name}"] = d
        case_metrics[f"HD95_{name}"] = h
        dice_vals.append(d)
        hd_vals.append(h)
    case_metrics["Dice_mean"] = float(np.mean(dice_vals))
    case_metrics["HD95_mean"] = float(np.mean(hd_vals))
    return case_metrics


def aggregate_official_metrics(per_case: List[Dict[str, Any]], class_names: List[str]) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    for name in class_names:
        agg[f"Dice_{name}"] = float(np.mean([c[f"Dice_{name}"] for c in per_case]))
        agg[f"HD95_{name}"] = float(np.mean([c[f"HD95_{name}"] for c in per_case]))
    agg["Dice_mean"] = float(np.mean([agg[f"Dice_{n}"] for n in class_names]))
    agg["HD95_mean"] = float(np.mean([agg[f"HD95_{n}"] for n in class_names]))
    agg["n_cases"] = float(len(per_case))
    return agg


def write_official_metrics_bundle(
    out_dir: Path,
    *,
    norm_task: str,
    payload: Dict[str, Any],
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"metrics_official_{norm_task}"
    agg = payload["aggregate"]
    class_names = [c["name"] for c in payload["classes"]]

    with open(out_dir / f"{stem}.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    with open(out_dir / f"{stem}.txt", "w", encoding="utf-8") as fh:
        fh.write(
            f"eval_split={payload['eval_split']} n_cases={payload['n_cases']} fold={payload['fold']} task={norm_task} "
            f"roi={tuple(payload['roi_size'])} overlap={payload['overlap']:.2f} sw_batch_size={payload['sw_batch_size']} "
            f"tta={payload.get('tta', payload.get('tta_mirror', False))} mode={payload.get('mode', 'gaussian')}\n"
        )
        for name in class_names:
            fh.write(f"Dice_{name} = {agg[f'Dice_{name}']:.4f}\n")
        for name in class_names:
            fh.write(f"HD95_{name} = {agg[f'HD95_{name}']:.4f} mm\n")
        fh.write(f"Dice_mean = {agg['Dice_mean']:.4f}\n")
        fh.write(f"HD95_mean = {agg['HD95_mean']:.4f} mm\n")

    per_organ = {
        "organs": [
            {
                "name": n,
                "label": next(c["label"] for c in payload["classes"] if c["name"] == n),
                "Dice_mean": agg[f"Dice_{n}"],
                "HD95_mean": agg[f"HD95_{n}"],
            }
            for n in class_names
        ],
        "Dice_mean": agg["Dice_mean"],
        "HD95_mean": agg["HD95_mean"],
    }
    with open(out_dir / "per_organ_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(per_organ, fh, indent=2)
    with open(out_dir / "per_case_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(payload["per_case"], fh, indent=2)


def score_predictions_directory(
    predictions_dir: Path,
    cfg: Mapping[str, Any],
    *,
    out_dir: Optional[Path] = None,
    max_cases: Optional[int] = None,
) -> dict:
    """Score saved predictions (raw or postprocessed) with official metrics."""
    predictions_dir = Path(predictions_dir)
    data = dict(cfg.get("data") or {})
    task = str(data.get("dataset_name") or data.get("task") or "synapse")
    class_spec = class_spec_for_task(task)
    norm_task = _normalize_task(task)
    class_names = [name for _, name in class_spec]
    npz_dir = Path(resolve_nnformer_npz_dir(data)).resolve()
    fold = int(data.get("fold", data.get("nnformer_fold", 0)))

    pred_files = sorted(predictions_dir.glob("*.npz"))
    pred_files += sorted(predictions_dir.glob("*.npy"))
    pred_files += sorted(predictions_dir.glob("*.nii.gz"))
    if not pred_files:
        raise FileNotFoundError(f"No prediction files under {predictions_dir}")

    case_ids_cfg = _val_case_ids(cfg, npz_dir)
    per_case: List[Dict[str, Any]] = []
    for pf in pred_files:
        pred, spacing, case_id = load_case_prediction(pf)
        if case_id not in case_ids_cfg:
            continue
        if max_cases is not None and len(per_case) >= int(max_cases):
            break
        _, seg_gt, gt_spacing = load_npz_case(npz_dir, case_id, data)
        spacing_use = spacing if any(s > 0 for s in spacing) else gt_spacing
        per_case.append(
            compute_case_official_metrics(
                pred, seg_gt, spacing_use, class_spec, case_id=case_id, norm_task=norm_task
            )
        )

    if not per_case:
        raise FileNotFoundError(f"No scored cases from {predictions_dir}")

    agg = aggregate_official_metrics(per_case, class_names)
    infer_cfg = resolve_official_inference_config(cfg)
    payload = {
        "protocol": f"score_saved_predictions_{norm_task}",
        "eval_split": "val",
        "task": norm_task,
        "classes": [{"label": idx, "name": name} for idx, name in class_spec],
        "fold": fold,
        "n_cases": len(per_case),
        "case_ids": [c["case_id"] for c in per_case],
        "predictions_dir": str(predictions_dir),
        "roi_size": list(infer_cfg.roi_size),
        "overlap": float(infer_cfg.overlap),
        "sw_batch_size": int(infer_cfg.sw_batch_size),
        "mode": infer_cfg.mode,
        "tta": False,
        "tta_mirror": False,
        "aggregate": agg,
        "per_case": per_case,
    }
    if out_dir is not None:
        write_official_metrics_bundle(out_dir, norm_task=norm_task, payload=payload)
    return payload


def evaluate_official_npz(
    model: Union[nn.Module, Sequence[nn.Module], None],
    cfg: Mapping[str, Any],
    device: torch.device,
    *,
    out_dir: Optional[Path] = None,
    tta_mirror: bool = False,
    tta_axes: Tuple[int, ...] = (0, 1, 2),
    tta_mode: str = "mirror",
    max_cases: Optional[int] = None,
    eval_split: str = "val",
    overlap: Optional[float] = None,
    sw_batch_size: Optional[int] = None,
    roi_size: Optional[Sequence[int]] = None,
    save_predictions: bool = True,
    save_probabilities: bool = False,
    predictions_dir: Optional[Path] = None,
    run_sanity_checks: bool = True,
) -> dict:
    """Full-volume sliding-window eval on all val cases (or score existing predictions)."""
    if predictions_dir is not None:
        return score_predictions_directory(predictions_dir, cfg, out_dir=out_dir, max_cases=max_cases)

    if model is None:
        raise ValueError("model is required unless predictions_dir is set")
    if str(eval_split).strip().lower() != "val":
        raise NotImplementedError("official_npz currently supports eval_split='val' only")

    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    data = dict(cfg.get("data") or {})
    task = str(data.get("dataset_name") or data.get("task") or "synapse")
    class_spec = class_spec_for_task(task)
    norm_task = _normalize_task(task)
    class_names = [name for _, name in class_spec]

    npz_dir = resolve_nnformer_npz_dir(data)
    if npz_dir is None:
        raise FileNotFoundError("Set data.nnformer_preprocessed_dir or data.nnformer_npz_dir")
    npz_dir = Path(npz_dir).resolve()

    infer_cfg = resolve_official_inference_config(
        cfg,
        overlap=overlap,
        sw_batch_size=sw_batch_size,
        roi_size=roi_size,
        tta=tta_mirror,
        tta_axes=tta_axes,
        tta_mode=tta_mode,
    )
    fold = int(data.get("fold", data.get("nnformer_fold", 0)))

    case_ids = _val_case_ids(cfg, npz_dir)
    if max_cases is not None:
        case_ids = case_ids[: int(max_cases)]

    models = [model] if isinstance(model, nn.Module) else list(model)
    for m in models:
        m.to(device).eval()

    pred_dir = None
    if out_dir is not None and save_predictions:
        pred_dir = Path(out_dir) / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)

    per_case: List[Dict[str, Any]] = []
    max_label = int((cfg.get("data") or {}).get("num_classes", 14)) - 1

    pbar = tqdm(case_ids, desc=f"official-{norm_task} val n={len(case_ids)}")
    for ci, cid in enumerate(pbar):
        image, seg_gt, spacing = load_npz_case(npz_dir, cid, data)
        x = torch.from_numpy(image).unsqueeze(0)
        if save_probabilities:
            pred_lab, prob = sliding_window_predict_official(
                models,
                x,
                roi_size=infer_cfg.roi_size,
                overlap=infer_cfg.overlap,
                sw_batch_size=infer_cfg.sw_batch_size,
                mode=infer_cfg.mode,
                device=device,
                tta_mirror=infer_cfg.tta,
                tta_axes=infer_cfg.tta_axes,
                tta_mode=infer_cfg.tta_mode,
                tta_task=norm_task,
                return_probabilities=True,
            )
        else:
            pred_lab = sliding_window_predict_official(
                models,
                x,
                roi_size=infer_cfg.roi_size,
                overlap=infer_cfg.overlap,
                sw_batch_size=infer_cfg.sw_batch_size,
                mode=infer_cfg.mode,
                device=device,
                tta_mirror=infer_cfg.tta,
                tta_axes=infer_cfg.tta_axes,
                tta_mode=infer_cfg.tta_mode,
                tta_task=norm_task,
                return_probabilities=False,
            )
            prob = None

        if run_sanity_checks and ci == 0:
            _sanity_check_prediction(
                pred_lab,
                tuple(image.shape[-3:]),
                prob,
                max_label=max_label,
                case_id=cid,
            )
            print("[official_npz][sanity] first case checks passed", flush=True)

        if pred_dir is not None:
            save_case_prediction(
                pred_dir / f"{cid}.npz",
                case_id=cid,
                prediction=pred_lab,
                spacing_mm=spacing,
                probabilities=prob,
            )

        case_metrics = compute_case_official_metrics(
            pred_lab, seg_gt, spacing, class_spec, case_id=cid, norm_task=norm_task
        )
        per_case.append(case_metrics)
        pbar.set_postfix(mean=f"{case_metrics['Dice_mean']:.3f}")

    agg = aggregate_official_metrics(per_case, class_names)
    payload = {
        "protocol": f"unetrpp_inference_{norm_task}_style",
        "eval_split": eval_split,
        "task": norm_task,
        "classes": [{"label": idx, "name": name} for idx, name in class_spec],
        "fold": fold,
        "n_cases": len(per_case),
        "case_ids": case_ids,
        "roi_size": list(infer_cfg.roi_size),
        "overlap": infer_cfg.overlap,
        "sw_batch_size": infer_cfg.sw_batch_size,
        "mode": infer_cfg.mode,
        "tta": infer_cfg.tta,
        "tta_mirror": infer_cfg.tta,
        "tta_axes": list(infer_cfg.tta_axes),
        "tta_mode": infer_cfg.tta_mode,
        "n_checkpoints": len(models),
        "aggregate": agg,
        "per_case": per_case,
    }

    if out_dir is not None:
        write_official_metrics_bundle(Path(out_dir), norm_task=norm_task, payload=payload)
        print(f"[official_npz] wrote metrics under {out_dir}", flush=True)
    return payload


def write_synapse_mode_comparison_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "dice_mean",
        "hd95_mean",
        "spleen",
        "rkidney",
        "lkidney",
        "gallbladder",
        "liver",
        "stomach",
        "aorta",
        "pancreas",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


__all__ = [
    "SYNAPSE_8_ORGANS",
    "ACDC_3_CLASSES",
    "OfficialInferenceConfig",
    "aggregate_official_metrics",
    "class_spec_for_task",
    "compute_case_official_metrics",
    "evaluate_official_npz",
    "load_case_prediction",
    "load_npz_case",
    "resolve_official_inference_config",
    "save_case_prediction",
    "score_predictions_directory",
    "sliding_window_predict_official",
    "write_official_metrics_bundle",
    "write_synapse_mode_comparison_csv",
]
