"""
Save patch-level and full-volume segmentation figures (orthogonal middle slices + overlays).
Requires matplotlib.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch


def _ensure_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # noqa: BLE001
        raise ImportError("matplotlib is required for segmentation_vis (pip install matplotlib)") from exc


def _volume_to_gray_slice(
    vol: Union[torch.Tensor, np.ndarray], *, plane: str, index: Optional[int] = None
) -> np.ndarray:
    """Reduce [C,D,H,W] -> 2D float [0,1] display slice."""
    if isinstance(vol, torch.Tensor):
        x = vol.detach().float().cpu().numpy()
    else:
        x = np.asarray(vol, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected 4D [C,D,H,W], got {x.shape}")
    c, d, h, w = x.shape
    g = np.mean(x, axis=0) if c > 1 else x[0]
    if plane.lower() == "axial":
        z = index if index is not None else d // 2
        z = max(0, min(d - 1, z))
        sl = g[z, :, :]
    elif plane.lower() == "coronal":
        y = index if index is not None else h // 2
        y = max(0, min(h - 1, y))
        sl = g[:, y, :]
    elif plane.lower() == "sagittal":
        xi = index if index is not None else w // 2
        xi = max(0, min(w - 1, xi))
        sl = g[:, :, xi]
    else:
        raise ValueError(plane)
    sl = np.nan_to_num(sl, nan=0.0)
    lo, hi = float(sl.min()), float(sl.max())
    if hi - lo > 1e-6:
        sl = (sl - lo) / (hi - lo + 1e-8)
    else:
        sl = np.zeros_like(sl)
    return sl.astype(np.float32)


def mask_slice_correct(lab: Union[torch.Tensor, np.ndarray], plane: str, index: Optional[int] = None) -> np.ndarray:
    """Label volume [D,H,W] -> 2D slice."""
    if isinstance(lab, torch.Tensor):
        y = lab.detach().long().cpu().numpy()
    else:
        y = np.asarray(lab, dtype=np.int64)
    while y.ndim > 3:
        y = y[0]
    if y.ndim != 3:
        raise ValueError(f"Expected label [D,H,W], got {y.shape}")
    d, h, w = y.shape
    if plane.lower() == "axial":
        z = index if index is not None else d // 2
        z = max(0, min(d - 1, z))
        return y[z, :, :]
    if plane.lower() == "coronal":
        yi = index if index is not None else h // 2
        yi = max(0, min(h - 1, yi))
        return y[:, yi, :]
    if plane.lower() == "sagittal":
        xi = index if index is not None else w // 2
        xi = max(0, min(w - 1, xi))
        return y[:, :, xi]
    raise ValueError(plane)


def _class_colors_rgba(n_cls: int) -> np.ndarray:
    import matplotlib.cm as cm

    ncol = max(10, int(n_cls))
    cmap = cm.get_cmap("tab10", ncol)
    out: list = [np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)]
    for c in range(1, max(int(n_cls), 1)):
        rgb = cmap(c % ncol)[:3]
        out.append(np.array(list(rgb) + [1.0], dtype=np.float32))
    return np.stack(out, axis=0)


def _hex_to_rgb01(hex_code: str) -> Tuple[float, float, float]:
    h = str(hex_code).lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Expected 6-char hex color, got {hex_code!r}")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def synapse_organ_color_map() -> Dict[int, Tuple[float, float, float]]:
    """UNETR++ ``color_cycle`` hues mapped to Synapse 8-organ label ids."""
    from dinomim_pytorch.eval.official_npz import SYNAPSE_8_ORGANS

    hex_colors = (
        "4363d8",
        "f58231",
        "3cb44b",
        "e6194B",
        "911eb4",
        "ffe119",
        "42d4f4",
        "f032e6",
    )
    return {
        int(label_id): _hex_to_rgb01(hex_code)
        for (label_id, _), hex_code in zip(SYNAPSE_8_ORGANS, hex_colors)
    }


def synapse_organ_legend_entries() -> Tuple[List[Any], List[str]]:
    """Matplotlib Patch handles + organ names for Synapse foreground classes."""
    from matplotlib.patches import Patch

    from dinomim_pytorch.eval.official_npz import SYNAPSE_8_ORGANS

    colors = synapse_organ_color_map()
    handles: List[Any] = []
    labels: List[str] = []
    for label_id, name in SYNAPSE_8_ORGANS:
        rgb = colors[int(label_id)]
        handles.append(
            Patch(
                facecolor=rgb,
                edgecolor="#333333",
                linewidth=0.6,
                label=str(name),
            )
        )
        labels.append(str(name).replace("_", " "))
    return handles, labels


def overlay_synapse_organs_on_gray(
    gray: np.ndarray,
    mask_lab: np.ndarray,
    *,
    alpha: float = 0.45,
    label_colors: Optional[Mapping[int, Tuple[float, float, float]]] = None,
) -> np.ndarray:
    """Semi-transparent organ overlay on grayscale CT (Synapse label ids)."""
    colors = dict(label_colors or synapse_organ_color_map())
    g = np.clip(np.asarray(gray, dtype=np.float32), 0.0, 1.0)
    rgb = np.stack([g, g, g], axis=-1)
    lab = np.asarray(mask_lab, dtype=np.int64)
    overlay = rgb.copy()
    for label_id in sorted(colors):
        sel = lab == int(label_id)
        if not np.any(sel):
            continue
        col = colors[int(label_id)]
        for ch in range(3):
            overlay[..., ch][sel] = overlay[..., ch][sel] * (1 - alpha) + col[ch] * alpha
    return np.clip(overlay, 0, 1)


def axial_slice_index_max_foreground(lab_dhw: Union[torch.Tensor, np.ndarray]) -> int:
    """Axial slice with largest foreground voxel count."""
    if isinstance(lab_dhw, torch.Tensor):
        y = lab_dhw.detach().long().cpu().numpy()
    else:
        y = np.asarray(lab_dhw, dtype=np.int64)
    while y.ndim > 3:
        y = y[0]
    fg = (y > 0).astype(np.float64)
    sums = fg.sum(axis=(1, 2))
    return int(np.argmax(sums))


def overlay_mask_on_gray(
    gray: np.ndarray, mask_lab: np.ndarray, n_classes: int, *, alpha: float = 0.45
) -> np.ndarray:
    """gray [H,W] 0–1; mask_lab int [H,W] -> RGB [H,W,3]."""
    colors = _class_colors_rgba(n_classes + 4)
    gh, gw = gray.shape[-2:]
    rgb = np.stack([gray, gray, gray], axis=-1)
    mh, mw = mask_lab.shape[-2:]
    if (mh, mw) != (gh, gw):
        raise ValueError(f"Shape mismatch gray {gray.shape} mask {mask_lab.shape}")
    m = mask_lab.astype(np.int64).clip(0, len(colors) - 1)
    overlay = rgb.copy().astype(np.float32)
    for c in range(1, max(1, n_classes)):
        sel = m == c
        if not np.any(sel):
            continue
        col = colors[min(c, len(colors) - 1)]
        for ch in range(3):
            overlay[..., ch][sel] = overlay[..., ch][sel] * (1 - alpha) + col[ch] * alpha
    return np.clip(overlay, 0, 1)


def save_three_plane_panel(
    out_path: Union[str, Path],
    image_cdhw: Union[torch.Tensor, np.ndarray],
    pred_logits_cdhw: Union[torch.Tensor, np.ndarray],
    gt_dhw: Union[torch.Tensor, np.ndarray],
    n_classes: int,
    *,
    title_prefix: str = "",
) -> None:
    """
    Six-panel figure: axial / coronal / sagittal × (GT overlay | Pred overlay).
    ``pred_logits_cdhw``: [C,D,H,W]; ``gt_dhw``: [D,H,W].
    """
    plt = _ensure_matplotlib()
    planes = ["axial", "coronal", "sagittal"]

    pred = pred_logits_cdhw
    if isinstance(pred, torch.Tensor):
        pred_arg = torch.argmax(pred, dim=0).long().cpu().numpy() if pred.ndim == 4 else pred.long().cpu().numpy()
    else:
        pa = np.asarray(pred)
        if pa.ndim == 4:
            pred_arg = pa.argmax(axis=0).astype(np.int64)
        else:
            pred_arg = pa.astype(np.int64)
    while pred_arg.ndim > 3:
        pred_arg = pred_arg[0]

    if isinstance(gt_dhw, torch.Tensor):
        gn = gt_dhw.long().detach().cpu().numpy()
    else:
        gn = np.asarray(gt_dhw, dtype=np.int64)
    while gn.ndim > 3:
        gn = gn[0]

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.8))
    for j, plane in enumerate(planes):
        g = _volume_to_gray_slice(image_cdhw, plane=plane)
        m_gt = mask_slice_correct(gn, plane)
        m_pr = mask_slice_correct(pred_arg, plane)
        im_gt = overlay_mask_on_gray(g, m_gt, n_classes)
        im_pr = overlay_mask_on_gray(g, m_pr, n_classes)
        axes[0, j].imshow(im_gt, vmin=0, vmax=1)
        axes[0, j].set_title(f"{title_prefix}{plane}: GT overlay")
        axes[0, j].axis("off")
        axes[1, j].imshow(im_pr, vmin=0, vmax=1)
        axes[1, j].set_title(f"{title_prefix}{plane}: Pred overlay")
        axes[1, j].axis("off")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_gray_pred_gt_panel(
    out_path: Union[str, Path],
    gray_slice: np.ndarray,
    pred_slice_lab: np.ndarray,
    gt_slice_lab: np.ndarray,
    n_classes: int,
    *,
    subtitle: str = "",
) -> None:
    plt = _ensure_matplotlib()
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    g = np.asarray(gray_slice, dtype=np.float32)
    lo, hi = float(g.min()), float(g.max())
    if hi - lo > 1e-6:
        g = (g - lo) / (hi - lo + 1e-8)
    axes[0].imshow(np.stack([g, g, g], axis=-1), vmin=0, vmax=1)
    axes[0].set_title("Patch input (mean)")
    axes[1].imshow(overlay_mask_on_gray(g, pred_slice_lab, n_classes), vmin=0, vmax=1)
    axes[1].set_title("Pred")
    axes[2].imshow(overlay_mask_on_gray(g, gt_slice_lab, n_classes), vmin=0, vmax=1)
    axes[2].set_title("GT")
    diff = (pred_slice_lab != gt_slice_lab).astype(np.float32)
    err_rgb = np.stack([g + 0.4 * diff, g * (1 - 0.5 * diff), g * (1 - 0.5 * diff)], axis=-1)
    axes[3].imshow(np.clip(err_rgb, 0, 1))
    axes[3].set_title("Errors (red)")
    for ax in axes:
        ax.axis("off")
    if subtitle:
        fig.suptitle(subtitle)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=125, bbox_inches="tight")
    plt.close(fig)


def patch_axial_middle_slice(
    image_bcdhw: torch.Tensor,
    label_bdhw: torch.Tensor,
    pred_logits_bcdhw: torch.Tensor,
    batch_index: int,
    n_classes: int,
    *,
    pred_labels_bdhw: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Middle axial slice from one batch item (patch / validation crop)."""
    x = image_bcdhw[batch_index]
    if x.dim() == 5:
        x = x[0]
    y = label_bdhw[batch_index]
    if y.dim() == 4 and y.shape[0] == 1:
        y = y[0]
    d = int(x.shape[1])
    z = max(0, d // 2)
    gray = _volume_to_gray_slice(x, plane="axial", index=z)
    if pred_labels_bdhw is not None:
        pr = pred_labels_bdhw[batch_index]
        if pr.dim() == 4 and pr.shape[0] == 1:
            pr = pr[0]
    else:
        lg = pred_logits_bcdhw[batch_index]
        if lg.dim() == 5:
            lg = lg[0]
        pr = lg.argmax(dim=0) if lg.dim() == 4 else lg.long()
    prs = mask_slice_correct(pr, "axial", z)
    gts = mask_slice_correct(y, "axial", z)
    return gray, prs, gts


def save_paper_style_synapse_montage(
    out_path: Union[str, Path],
    rows: Sequence[Mapping[str, Any]],
    *,
    column_titles: Sequence[str] = ("Image", "GT", "Prediction"),
    dpi: int = 150,
    show_row_labels: bool = True,
) -> None:
    """
    Paper-style figure: rows = cases/slices, columns = Image | GT overlay | Pred overlay.

    Each row dict: ``gray_hw`` [H,W], ``gt_hw``, ``pred_hw`` (int labels), optional ``label``.
    """
    if not rows:
        return
    plt = _ensure_matplotlib()
    from matplotlib.gridspec import GridSpec

    n_rows = len(rows)
    fig_h = max(3.2, 2.55 * n_rows + 1.05)
    fig = plt.figure(figsize=(9.2, fig_h))
    gs = GridSpec(
        n_rows + 1,
        3,
        figure=fig,
        height_ratios=[1.0] * n_rows + [0.24],
        hspace=0.10,
        wspace=0.05,
    )

    for i, row in enumerate(rows):
        gray = np.asarray(row["gray_hw"], dtype=np.float32)
        gt = np.asarray(row["gt_hw"], dtype=np.int64)
        pred = np.asarray(row["pred_hw"], dtype=np.int64)
        row_label = str(row.get("label") or f"case {i + 1}")

        ax_img = fig.add_subplot(gs[i, 0])
        ax_gt = fig.add_subplot(gs[i, 1])
        ax_pr = fig.add_subplot(gs[i, 2])

        ax_img.imshow(np.clip(gray, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax_gt.imshow(overlay_synapse_organs_on_gray(gray, gt), vmin=0, vmax=1)
        ax_pr.imshow(overlay_synapse_organs_on_gray(gray, pred), vmin=0, vmax=1)

        if i == 0:
            ax_img.set_title(str(column_titles[0]), fontsize=11, pad=6)
            ax_gt.set_title(str(column_titles[1]), fontsize=11, pad=6)
            ax_pr.set_title(str(column_titles[2]), fontsize=11, pad=6)
        if show_row_labels:
            ax_img.set_ylabel(row_label, fontsize=9, rotation=0, labelpad=42, va="center")
        for ax in (ax_img, ax_gt, ax_pr):
            ax.set_xticks([])
            ax.set_yticks([])

    ax_leg = fig.add_subplot(gs[n_rows, :])
    handles, labels = synapse_organ_legend_entries()
    ax_leg.legend(handles=handles, labels=labels, loc="center", ncol=4, frameon=False, fontsize=9)
    ax_leg.axis("off")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def save_paper_style_synapse_row(
    out_path: Union[str, Path],
    *,
    gray_hw: np.ndarray,
    gt_hw: np.ndarray,
    pred_hw: np.ndarray,
    title: str = "",
    dpi: int = 150,
) -> None:
    """Single-row Image | GT | Pred with bottom legend."""
    row: Dict[str, Any] = {
        "gray_hw": gray_hw,
        "gt_hw": gt_hw,
        "pred_hw": pred_hw,
        "label": title or "",
    }
    save_paper_style_synapse_montage(
        out_path,
        [row],
        column_titles=("Image", "GT", "Prediction"),
        dpi=dpi,
        show_row_labels=bool(title),
    )


def save_paper_style_synapse_method_compare_montage(
    out_path: Union[str, Path],
    rows: Sequence[Mapping[str, Any]],
    *,
    column_titles: Sequence[str] = (
        "Image",
        "GT",
        "Paper pretrained",
        "Scratch",
        "Ours",
    ),
    pred_keys: Sequence[str] = ("pred_paper_hw", "pred_scratch_hw", "pred_s400_hw"),
    dpi: int = 150,
    show_row_labels: bool = True,
) -> None:
    """
    Paper-style method comparison: Image | GT | Paper | Scratch | s400.

    Each row dict: ``gray_hw``, ``gt_hw``, and pred slice arrays keyed by ``pred_keys``.
    """
    if not rows:
        return
    plt = _ensure_matplotlib()
    from matplotlib.gridspec import GridSpec

    pred_keys = tuple(pred_keys)
    n_pred = len(pred_keys)
    n_cols = 2 + n_pred
    titles = list(column_titles)
    if len(titles) < n_cols:
        titles = titles + [f"Pred {i + 1}" for i in range(n_cols - len(titles))]
    titles = titles[:n_cols]

    n_rows = len(rows)
    fig_w = max(12.0, 2.05 * n_cols)
    fig_h = max(3.2, 2.55 * n_rows + 1.05)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        n_rows + 1,
        n_cols,
        figure=fig,
        height_ratios=[1.0] * n_rows + [0.24],
        hspace=0.10,
        wspace=0.05,
    )

    for i, row in enumerate(rows):
        gray = np.asarray(row["gray_hw"], dtype=np.float32)
        gt = np.asarray(row["gt_hw"], dtype=np.int64)
        preds = [np.asarray(row[k], dtype=np.int64) for k in pred_keys]
        row_label = str(row.get("label") or f"case {i + 1}")

        ax_img = fig.add_subplot(gs[i, 0])
        ax_gt = fig.add_subplot(gs[i, 1])
        pred_axes = [fig.add_subplot(gs[i, j + 2]) for j in range(n_pred)]

        ax_img.imshow(np.clip(gray, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax_gt.imshow(overlay_synapse_organs_on_gray(gray, gt), vmin=0, vmax=1)
        for ax, pred in zip(pred_axes, preds):
            ax.imshow(overlay_synapse_organs_on_gray(gray, pred), vmin=0, vmax=1)

        if i == 0:
            ax_img.set_title(str(titles[0]), fontsize=10, pad=6)
            ax_gt.set_title(str(titles[1]), fontsize=10, pad=6)
            for j, ax in enumerate(pred_axes):
                ax.set_title(str(titles[j + 2]), fontsize=10, pad=6)
        if show_row_labels and row_label:
            ax_img.set_ylabel(row_label, fontsize=9, rotation=0, labelpad=42, va="center")
        for ax in (ax_img, ax_gt, *pred_axes):
            ax.set_xticks([])
            ax.set_yticks([])

    ax_leg = fig.add_subplot(gs[n_rows, :])
    handles, labels = synapse_organ_legend_entries()
    ax_leg.legend(handles=handles, labels=labels, loc="center", ncol=4, frameon=False, fontsize=9)
    ax_leg.axis("off")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def save_paper_style_synapse_method_compare_row(
    out_path: Union[str, Path],
    *,
    gray_hw: np.ndarray,
    gt_hw: np.ndarray,
    pred_paper_hw: np.ndarray,
    pred_scratch_hw: np.ndarray,
    pred_s400_hw: np.ndarray,
    title: str = "",
    dpi: int = 150,
) -> None:
    """Single-row Image | GT | Paper | Scratch | s400 with bottom legend."""
    row: Dict[str, Any] = {
        "gray_hw": gray_hw,
        "gt_hw": gt_hw,
        "pred_paper_hw": pred_paper_hw,
        "pred_scratch_hw": pred_scratch_hw,
        "pred_s400_hw": pred_s400_hw,
        "label": title or "",
    }
    save_paper_style_synapse_method_compare_montage(
        out_path,
        [row],
        dpi=dpi,
        show_row_labels=bool(title),
    )


__all__ = [
    "axial_slice_index_max_foreground",
    "overlay_synapse_organs_on_gray",
    "save_paper_style_synapse_montage",
    "save_paper_style_synapse_row",
    "save_paper_style_synapse_method_compare_montage",
    "save_paper_style_synapse_method_compare_row",
    "save_three_plane_panel",
    "save_gray_pred_gt_panel",
    "patch_axial_middle_slice",
    "overlay_mask_on_gray",
    "synapse_organ_color_map",
    "synapse_organ_legend_entries",
]
