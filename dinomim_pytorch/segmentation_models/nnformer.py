# Backward compatibility: local 3D nnFormer-style.

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from dinomim_pytorch.segmentation_models.local_nnformer3d import (
    LocalNNFormer3D,
    build_local_nnformer3d,
)

NnFormerSeg = LocalNNFormer3D  # historical name


def build_nnformer(c: Dict[str, Any]) -> nn.Module:
    return build_local_nnformer3d(c)
