# Local 3D UNETR++ stub (default) or official ``unetr_plus_plus-main`` when ``preferred_source: official``.

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from dinomim_pytorch.segmentation_models.local_unetrpp3d import build_local_unetrpp3d


def _use_official(c: Dict[str, Any]) -> bool:
    pref = str(c.get("preferred_source", "local")).lower().replace("-", "")
    return pref in ("official", "unetrpp", "unetr_plus_plus", "unetrplusplus", "paper") or bool(
        c.get("use_official_unetrpp", False)
    )


def build_unetrpp(c: Dict[str, Any]) -> nn.Module:
    if _use_official(c):
        from dinomim_pytorch.segmentation_models.official_unetrpp3d import build_official_unetrpp3d

        return build_official_unetrpp3d(c)
    return build_local_unetrpp3d(c)
