"""
MONAI 3D dict transforms for BraTS (multichannel MRI) and BTCV (CT) configs.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
    NormalizeIntensityd,
)

__all__ = [
    "build_brats_compose",
    "build_btcv_compose",
    "build_nnformer_npz_compose",
    "build_nnformer_npz_eval_fullvolume_compose",
    "val_sliding_compose",
    "build_eval_fullvolume_compose",
]


def _3(x: Any, default: Tuple[int, int, int] = (96, 96, 96)) -> Tuple:
    if x is None:
        return default
    if isinstance(x, (int, float)):
        v = int(x)
        return (v, v, v)
    if isinstance(x, (list, tuple)) and len(x) >= 3:
        return (int(x[0]), int(x[1]), int(x[2]))
    return default


def _rand_or_center(cfg: Dict, train: bool, img: str, lab: str) -> Any:
    roi = _3(cfg.get("image_size", (96, 96, 96)))
    if not train:
        return CenterSpatialCropd(keys=[img, lab], roi_size=roi)
    if bool(cfg.get("pos_neg_crop", True)):
        return RandCropByPosNegLabeld(
            keys=[img, lab],
            label_key=lab,
            spatial_size=roi,  # type: ignore[arg-type]
            pos=1,
            neg=1,
            num_samples=1,
        )
    return RandSpatialCropd(
        keys=[img, lab], roi_size=roi, random_size=False, random_center=True
    )


def build_brats_compose(
    data_cfg: Dict[str, Any], train: bool = True, keys: Optional[Dict[str, str]] = None
) -> Compose:
    c = data_cfg or {}
    img = (keys or {}).get("image", c.get("image_key", "image"))
    lab = (keys or {}).get("label", c.get("label_key", "label"))
    k = [img, lab]
    pix = _3(c.get("spacing", (1.0, 1.0, 1.0)), (1, 1, 1))  # type: ignore[arg-type]
    pixf = (float(pix[0]), float(pix[1]), float(pix[2]))
    # Loader gives numpy: image [C,D,H,W], label [D,H,W]. MONAI needs explicit channel_dim without MetaTensor meta.
    steps: List[Any] = [
        EnsureChannelFirstd(keys=[img], allow_missing_keys=True, channel_dim=0),
        EnsureChannelFirstd(keys=[lab], allow_missing_keys=True, channel_dim="no_channel"),
        Orientationd(keys=k, axcodes="RAS"),
        Spacingd(keys=k, pixdim=pixf, mode=("bilinear", "nearest")),
        NormalizeIntensityd(
            keys=[img], nonzero=bool(c.get("normalize_nonzero", True)), channel_wise=True
        ),
    ]
    steps.append(_rand_or_center(c, train, img, lab))
    if train:
        steps += [
            RandFlipd(keys=k, prob=0.1, spatial_axis=0),
            RandRotate90d(keys=k, prob=0.1, max_k=3, spatial_axes=(0, 1)),
            RandScaleIntensityd(keys=[img], factors=0.1, prob=0.1),
            RandShiftIntensityd(keys=[img], offsets=0.1, prob=0.1),
        ]
    steps.append(ToTensord(keys=k))
    return Compose(steps)


def build_btcv_compose(
    data_cfg: Dict[str, Any], train: bool = True, keys: Optional[Dict[str, str]] = None
) -> Compose:
    c = data_cfg or {}
    img = (keys or {}).get("image", c.get("image_key", "image"))
    lab = (keys or {}).get("label", c.get("label_key", "label"))
    k = [img, lab]
    pix = _3(c.get("spacing", (1.5, 1.5, 2.0)))
    pixf = (float(pix[0]), float(pix[1]), float(pix[2]))
    inty = c.get("intensity") or {}
    a_min, a_max = float(inty.get("a_min", -175)), float(inty.get("a_max", 250))
    b_min, b_max = float(inty.get("b_min", 0.0)), float(inty.get("b_max", 1.0))
    clip = bool(inty.get("clip", True))
    steps: List[Any] = [
        EnsureChannelFirstd(keys=[img], allow_missing_keys=True, channel_dim=0),
        EnsureChannelFirstd(keys=[lab], allow_missing_keys=True, channel_dim="no_channel"),
        Orientationd(keys=k, axcodes="RAS"),
        Spacingd(keys=k, pixdim=pixf, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=[img], a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=clip
        ),
    ]
    steps.append(_rand_or_center(c, train, img, lab))
    if train:
        steps += [
            RandFlipd(keys=k, prob=0.1, spatial_axis=0),
            RandRotate90d(keys=k, prob=0.1, max_k=3, spatial_axes=(0, 1)),
            RandScaleIntensityd(keys=[img], factors=0.05, prob=0.1),
        ]
    steps.append(ToTensord(keys=k))
    return Compose(steps)


def val_sliding_compose(
    data_cfg: Dict[str, Any], brats: bool, keys: Optional[Dict[str, str]] = None
) -> Compose:
    """Validation: orientation + spacing + norm/window, center crop, tensor (no random aug)."""
    return build_brats_compose(data_cfg, train=False, keys=keys) if brats else build_btcv_compose(  # type: ignore
        data_cfg, train=False, keys=keys
    )


def build_brats_eval_fullvolume_compose(
    data_cfg: Dict[str, Any], keys: Optional[Dict[str, str]] = None
) -> Compose:
    """
    BraTS-style preprocessing **without** center crop (full FOV tensor for sliding-window eval / figures).
    """
    c = data_cfg or {}
    img = (keys or {}).get("image", c.get("image_key", "image"))
    lab = (keys or {}).get("label", c.get("label_key", "label"))
    k = [img, lab]
    pix = _3(c.get("spacing", (1.0, 1.0, 1.0)), (1, 1, 1))  # type: ignore[arg-type]
    pixf = (float(pix[0]), float(pix[1]), float(pix[2]))
    steps: List[Any] = [
        EnsureChannelFirstd(keys=[img], allow_missing_keys=True, channel_dim=0),
        EnsureChannelFirstd(keys=[lab], allow_missing_keys=True, channel_dim="no_channel"),
        Orientationd(keys=k, axcodes="RAS"),
        Spacingd(keys=k, pixdim=pixf, mode=("bilinear", "nearest")),
        NormalizeIntensityd(
            keys=[img], nonzero=bool(c.get("normalize_nonzero", True)), channel_wise=True
        ),
        ToTensord(keys=k),
    ]
    return Compose(steps)


def build_btcv_eval_fullvolume_compose(
    data_cfg: Dict[str, Any], keys: Optional[Dict[str, str]] = None
) -> Compose:
    """BTCV-style preprocessing **without** center crop."""
    c = data_cfg or {}
    img = (keys or {}).get("image", c.get("image_key", "image"))
    lab = (keys or {}).get("label", c.get("label_key", "label"))
    k = [img, lab]
    pix = _3(c.get("spacing", (1.5, 1.5, 2.0)))
    pixf = (float(pix[0]), float(pix[1]), float(pix[2]))
    inty = c.get("intensity") or {}
    a_min, a_max = float(inty.get("a_min", -175)), float(inty.get("a_max", 250))
    b_min, b_max = float(inty.get("b_min", 0.0)), float(inty.get("b_max", 1.0))
    clip = bool(inty.get("clip", True))
    steps: List[Any] = [
        EnsureChannelFirstd(keys=[img], allow_missing_keys=True, channel_dim=0),
        EnsureChannelFirstd(keys=[lab], allow_missing_keys=True, channel_dim="no_channel"),
        Orientationd(keys=k, axcodes="RAS"),
        Spacingd(keys=k, pixdim=pixf, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=[img], a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=clip
        ),
        ToTensord(keys=k),
    ]
    return Compose(steps)


def build_nnformer_npz_compose(
    data_cfg: Dict[str, Any], train: bool = True, keys: Optional[Dict[str, str]] = None
) -> Compose:
    """
    nnFormer ``*.npz`` volumes are already resampled and intensity-normalized.

    Applies channel checks, random/center crop, light aug, and ``ToTensord`` only.
    """
    c = data_cfg or {}
    img = (keys or {}).get("image", c.get("image_key", "image"))
    lab = (keys or {}).get("label", c.get("label_key", "label"))
    k = [img, lab]
    steps: List[Any] = [
        EnsureChannelFirstd(keys=[img], allow_missing_keys=True, channel_dim=0),
        EnsureChannelFirstd(keys=[lab], allow_missing_keys=True, channel_dim="no_channel"),
    ]
    roi = _3(c.get("image_size", (96, 96, 96)))
    steps.append(_rand_or_center(c, train, img, lab))
    if train:
        # Official UNETR++ Synapse uses anisotropic ``(D,H,W)`` e.g. (64,128,128). Never rotate
        # axes (0,1): that swaps D/H and breaks decoder skip shapes.
        steps += [
            RandFlipd(keys=k, prob=0.1, spatial_axis=0),
            RandFlipd(keys=k, prob=0.1, spatial_axis=1),
            RandFlipd(keys=k, prob=0.1, spatial_axis=2),
            RandRotate90d(keys=k, prob=0.1, max_k=3, spatial_axes=(1, 2)),
            RandScaleIntensityd(keys=[img], factors=0.05, prob=0.1),
        ]
    steps.append(ResizeWithPadOrCropd(keys=k, spatial_size=roi))
    steps.append(ToTensord(keys=k))
    return Compose(steps)


def build_nnformer_npz_eval_fullvolume_compose(
    data_cfg: Dict[str, Any], keys: Optional[Dict[str, str]] = None
) -> Compose:
    """Full-volume eval on npz (no crop, no re-normalization)."""
    c = data_cfg or {}
    img = (keys or {}).get("image", c.get("image_key", "image"))
    lab = (keys or {}).get("label", c.get("label_key", "label"))
    k = [img, lab]
    return Compose(
        [
            EnsureChannelFirstd(keys=[img], allow_missing_keys=True, channel_dim=0),
            EnsureChannelFirstd(keys=[lab], allow_missing_keys=True, channel_dim="no_channel"),
            ToTensord(keys=k),
        ]
    )


def build_eval_fullvolume_compose(data_cfg: Dict[str, Any], dataset_name: str) -> Compose:
    c = data_cfg or {}
    loader = str(c.get("loader", "nifti_csv")).lower().replace("-", "_")
    if loader in ("nnformer", "nnformer_npz", "npz", "nnformer_preprocessed"):
        return build_nnformer_npz_eval_fullvolume_compose(c)
    n = str(dataset_name or "").lower()
    if n in ("btcv", "bmcv", "synapse"):
        return build_btcv_eval_fullvolume_compose(data_cfg)
    return build_brats_eval_fullvolume_compose(data_cfg)
