"""Mirror TTA helpers for 3D sliding-window segmentation inference."""
from __future__ import annotations

from typing import List, Sequence, Tuple


def mirror_tta_flip_dims(
    tensor_ndim: int = 5,
    spatial_axes: Sequence[int] = (0, 1, 2),
    *,
    tta_mode: str = "mirror",
) -> List[Tuple[int, ...]]:
    """
    Return ``torch.flip`` dimension tuples for [B, C, D, H, W] tensors.

    ``spatial_axes`` are indices into (D, H, W), i.e. 0→D (dim 2), 1→H (dim 3), 2→W (dim 4).
    Identity is always included first.
    """
    mode = str(tta_mode).strip().lower()
    if mode not in ("mirror", "none", "off"):
        raise ValueError(f"Unsupported tta_mode={tta_mode!r} (use 'mirror')")
    if mode in ("none", "off"):
        return [tuple()]

    if tensor_ndim != 5:
        raise ValueError(f"Expected 5D tensor layout [B,C,D,H,W], got ndim={tensor_ndim}")
    base = 2  # spatial dims start at index 2
    dims = [base + int(a) for a in spatial_axes]
    if len(dims) == 0:
        return [tuple()]
    if len(dims) == 1:
        d0 = dims[0]
        return [(), (d0,)]
    if len(dims) == 2:
        d0, d1 = dims
        return [(), (d0,), (d1,), (d0, d1)]
    if len(dims) == 3:
        d, h, w = dims
        return [(), (d,), (h,), (w,), (d, h), (d, w), (h, w), (d, h, w)]
    raise ValueError(f"At most 3 spatial axes supported, got {spatial_axes!r}")


def parse_tta_axes(raw: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if not parts:
        return (0, 1, 2)
    axes = tuple(int(p) for p in parts)
    for a in axes:
        if a not in (0, 1, 2):
            raise ValueError(f"tta_axes values must be 0,1,2 (D,H,W); got {axes}")
    return axes


def permute_mirror_prob_channels(
    prob_kdhw: np.ndarray,
    flip_dims: Tuple[int, ...],
    *,
    task: str = "synapse",
) -> np.ndarray:
    """
    Remap class channels after mirror flip-back for asymmetric organs.

    For Synapse, left-right reflection (``W`` / tensor dim 4) swaps right/left kidney channels.
    """
    if not flip_dims:
        return prob_kdhw
    task_key = str(task).strip().lower()
    if task_key not in ("synapse", "task002", "task002_synapse", "btcv"):
        return prob_kdhw
    if 4 not in flip_dims:
        return prob_kdhw
    out = prob_kdhw.copy()
    out[2], out[3] = prob_kdhw[3], prob_kdhw[2]
    return out


__all__ = ["mirror_tta_flip_dims", "parse_tta_axes", "permute_mirror_prob_channels"]
