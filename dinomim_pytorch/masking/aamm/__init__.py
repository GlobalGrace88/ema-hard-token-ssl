"""Anatomy-Aware Multi-Zone Masking (AAMM), ported from MAE_v3/medical_mim."""

from dinomim_pytorch.masking.aamm.anatomy_priors import (
    compute_pseudo_anatomy_priors_2d,
    compute_pseudo_anatomy_priors_3d,
)
from dinomim_pytorch.masking.aamm.multi_zone_masking import (
    assign_patch_zones_2d,
    assign_patch_zones_3d,
    sample_aamm_mask_2d,
    sample_aamm_mask_3d,
)

__all__ = [
    "compute_pseudo_anatomy_priors_2d",
    "compute_pseudo_anatomy_priors_3d",
    "assign_patch_zones_2d",
    "assign_patch_zones_3d",
    "sample_aamm_mask_2d",
    "sample_aamm_mask_3d",
]
