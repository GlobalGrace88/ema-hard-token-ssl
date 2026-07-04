"""
Upstream Synapse eval for MONAI / non-UNETR++ models.

Runs full-volume sliding-window inference on nnFormer stage1 ``.npz`` cases,
exports predictions as ``.nii.gz`` in upstream layout, and scores with the
MAE_BYOL ``score_upstream_paper_metrics.py`` protocol (gt_segmentations).
"""
from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from dinomim_pytorch.datasets.nnformer_npz import resolve_nnformer_npz_dir
from dinomim_pytorch.eval.official_npz import (
    _val_case_ids,
    load_npz_case,
    resolve_official_inference_config,
    sliding_window_predict_official,
)


def _export_upstream_nifti(pred: np.ndarray, props: Mapping[str, Any], out_path: Path) -> None:
    """Place resampled prediction back into upstream gt nifti geometry."""
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError("SimpleITK required for upstream export") from exc

    pred = np.asarray(pred, dtype=np.int32)
    shape_after_crop = tuple(int(x) for x in props["size_after_cropping"])
    shape_before_crop = tuple(int(x) for x in props["original_size_of_raw_data"])
    bbox = props.get("crop_bbox")

    if pred.shape != shape_after_crop:
        from scipy.ndimage import zoom

        factors = [shape_after_crop[i] / max(pred.shape[i], 1) for i in range(3)]
        pred = zoom(pred, factors, order=0).astype(np.int32, copy=False)

    if bbox is not None:
        full = np.zeros(shape_before_crop, dtype=np.int32)
        z0, z1 = int(bbox[0][0]), int(min(bbox[0][0] + pred.shape[0], shape_before_crop[0]))
        y0, y1 = int(bbox[1][0]), int(min(bbox[1][0] + pred.shape[1], shape_before_crop[1]))
        x0, x1 = int(bbox[2][0]), int(min(bbox[2][0] + pred.shape[2], shape_before_crop[2]))
        full[z0:z1, y0:y1, x0:x1] = pred[: z1 - z0, : y1 - y0, : x1 - x0]
        pred = full
    elif pred.shape != shape_before_crop:
        from scipy.ndimage import zoom

        factors = [shape_before_crop[i] / max(pred.shape[i], 1) for i in range(3)]
        pred = zoom(pred, factors, order=0).astype(np.int32, copy=False)

    itk_img = sitk.GetImageFromArray(pred.astype(np.uint8))
    itk_img.SetSpacing(tuple(float(x) for x in props["itk_spacing"]))
    itk_img.SetOrigin(tuple(float(x) for x in props["itk_origin"]))
    itk_img.SetDirection(tuple(float(x) for x in props["itk_direction"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(itk_img, str(out_path))


def _load_case_properties(npz_dir: Path, case_id: str) -> dict:
    pkl_path = npz_dir / f"{case_id}.pkl"
    with open(pkl_path, "rb") as fh:
        return pickle.load(fh)


def _gt_segmentations_dir(cfg: Mapping[str, Any]) -> Path:
    data = dict(cfg.get("data") or {})
    explicit = data.get("nnformer_gt_segmentations_dir") or data.get("gt_segmentations_dir")
    if explicit:
        return Path(str(explicit)).expanduser().resolve()
    import os

    env_gt = os.environ.get("DINOMIM_GT_SEGMENTATIONS_DIR", "").strip()
    if env_gt:
        p = Path(env_gt).expanduser().resolve()
        if p.is_dir():
            return p
    pre = data.get("nnformer_preprocessed_dir")
    if not pre:
        raise FileNotFoundError(
            "Set data.nnformer_preprocessed_dir, data.gt_segmentations_dir, "
            "or DINOMIM_GT_SEGMENTATIONS_DIR"
        )
    return Path(str(pre)).expanduser().resolve() / "gt_segmentations"


def _score_upstream_predictions(
    *,
    pred_dir: Path,
    gt_dir: Path,
    score_out: Path,
    case_ids: Sequence[str],
    dice_only: bool,
) -> dict:
    import os

    score_script = Path(os.environ.get("DINOMIM_UPSTREAM_SCORE_SCRIPT", "")).expanduser()
    if not score_script.is_file():
        raise FileNotFoundError(
            "Upstream paper scoring script not configured. "
            "Set DINOMIM_UPSTREAM_SCORE_SCRIPT to score_upstream_paper_metrics.py "
            "or use official_npz eval for downstream Synapse."
        )
    score_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(score_script),
        "--dataset",
        "synapse",
        "--pred_dir",
        str(pred_dir),
        "--gt_dir",
        str(gt_dir),
        "--out_dir",
        str(score_out),
        "--case_ids",
        *list(case_ids),
    ]
    if dice_only:
        cmd.append("--dice_only")
    subprocess.run(cmd, check=True)
    with open(score_out / "metrics_official_synapse_paper.json", encoding="utf-8") as fh:
        return json.load(fh)


def evaluate_upstream_nnformer(
    model: nn.Module,
    cfg: Mapping[str, Any],
    device: torch.device,
    *,
    out_dir: Path,
    overlap: Optional[float] = None,
    sw_batch_size: Optional[int] = None,
    max_cases: Optional[int] = None,
    dice_only: bool = False,
) -> dict:
    """Inference + upstream nifti export + paper metrics for fold val cases."""
    out_dir = Path(out_dir)
    raw_dir = out_dir / "validation_raw"
    score_dir = out_dir / "score_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    data = dict(cfg.get("data") or {})
    npz_dir = resolve_nnformer_npz_dir(data)
    if npz_dir is None:
        raise FileNotFoundError("Set data.nnformer_preprocessed_dir for upstream eval")
    npz_dir = Path(npz_dir).resolve()
    gt_dir = _gt_segmentations_dir(cfg)
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Missing gt_segmentations: {gt_dir}")

    infer_cfg = resolve_official_inference_config(cfg, overlap=overlap, sw_batch_size=sw_batch_size)
    fold = int(data.get("fold", data.get("nnformer_fold", 0)))
    case_ids = _val_case_ids(cfg, npz_dir)
    if max_cases is not None:
        case_ids = case_ids[: int(max_cases)]

    model.to(device).eval()

    for cid in tqdm(case_ids, desc=f"upstream-synapse fold={fold} n={len(case_ids)}"):
        image, _seg_gt, _spacing = load_npz_case(npz_dir, cid, data)
        props = _load_case_properties(npz_dir, cid)
        x = torch.from_numpy(image).unsqueeze(0)
        pred = sliding_window_predict_official(
            [model],
            x,
            roi_size=infer_cfg.roi_size,
            overlap=infer_cfg.overlap,
            sw_batch_size=infer_cfg.sw_batch_size,
            mode=infer_cfg.mode,
            device=device,
            tta_mirror=False,
            tta_axes=infer_cfg.tta_axes,
            tta_mode=infer_cfg.tta_mode,
            tta_task="synapse",
        )
        out_nii = raw_dir / f"{cid}.nii.gz"
        _export_upstream_nifti(pred, props, out_nii)

    payload = _score_upstream_predictions(
        pred_dir=raw_dir,
        gt_dir=gt_dir,
        score_out=score_dir,
        case_ids=case_ids,
        dice_only=dice_only,
    )
    payload["protocol"] = "upstream_nnformer_monai"
    payload["fold"] = fold
    payload["n_cases"] = len(case_ids)
    payload["case_ids"] = list(case_ids)
    payload["validation_raw"] = str(raw_dir)
    payload["overlap"] = float(infer_cfg.overlap)
    payload["roi_size"] = list(infer_cfg.roi_size)
    payload["sw_batch_size"] = int(infer_cfg.sw_batch_size)
    with open(out_dir / "upstream_eval_summary.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return payload
