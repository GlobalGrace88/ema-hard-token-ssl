"""
Swin UNETR inpainting + masked EMA teacher-feature reconstruction.

Stage mapping (swinViT outputs, feature_size=48 default):
  stage2 -> index 1 (96 ch)
  stage3 -> index 2 (192 ch)
  stage4 -> index 3 (384 ch)
"""

from __future__ import annotations

import copy
import sys
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from dinomim_pytorch.segmentation_models.monai_models import build_swinunetr
from dinomim_pytorch.unetrpp_feature_reconstruction import (
    MultiStageFeaturePredictors,
    _stage_keys,
    resolve_feature_recon_version,
    resolve_feature_stages,
    tokens_from_stage_feat,
)

# UNETR++ stage numbers -> swinViT output list index (skip shallow stem index 0).
SWIN_STAGE_TO_HIDDEN_IDX: Dict[int, int] = {2: 1, 3: 2, 4: 3}
SWIN_STAGE_SOURCE_DESC: Dict[int, str] = {
    2: "swinViT output index 1 (96 ch @ feature_size=48)",
    3: "swinViT output index 2 (192 ch @ feature_size=48)",
    4: "swinViT output index 3 (384 ch @ feature_size=48)",
}


def _primary_logits(raw):
    if isinstance(raw, (list, tuple)):
        return raw[0]
    return raw


def _student_module(model: nn.Module) -> nn.Module:
    return model.student_net.module if hasattr(model.student_net, "module") else model.student_net


def _infer_swin_stage_dims(
    swin_vit: nn.Module,
    spatial: Tuple[int, int, int],
    stages: List[int],
) -> Dict[str, int]:
    with torch.no_grad():
        device = next(swin_vit.parameters()).device
        x = torch.zeros(1, 1, *spatial, device=device)
        feats = swin_vit(x)
    if not isinstance(feats, (list, tuple)):
        feats = [feats]
    dims: Dict[str, int] = {}
    for stage in stages:
        idx = SWIN_STAGE_TO_HIDDEN_IDX[int(stage)]
        if idx >= len(feats):
            raise RuntimeError(f"Swin stage {stage} index {idx} out of range (n={len(feats)})")
        dims[f"stage{stage}"] = int(feats[idx].shape[1])
    return dims


def _extract_swin_stage_tokens(
    hidden_states: List[torch.Tensor],
    stage: int,
    spatial: Tuple[int, int, int],
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    idx = SWIN_STAGE_TO_HIDDEN_IDX[int(stage)]
    if idx >= len(hidden_states):
        raise RuntimeError(f"Swin stage {stage} index {idx} out of range (n={len(hidden_states)})")
    return tokens_from_stage_feat(hidden_states[idx], spatial)


class SwinUNETRInpaintingFeatureReconstruction(nn.Module):
    """Inpainting + masked teacher-feature reconstruction for MONAI SwinUNETR."""

    def __init__(
        self,
        student_net: nn.Module,
        *,
        stages: List[int],
        stage_dims: Dict[str, int],
        predictor_hidden_dim: int = 512,
        version: str = "v4_hard_stage34_smoothl1",
    ) -> None:
        super().__init__()
        if not hasattr(student_net, "swinViT"):
            raise AttributeError("student_net must expose swinViT (MONAI SwinUNETR)")
        self.student_net = student_net
        self.version = str(version)
        self.stages = list(stages)
        self.stage_keys = _stage_keys(self.stages)
        self.stage_dims = dict(stage_dims)
        self.feature_predictors = MultiStageFeaturePredictors(stage_dims, hidden_dim=predictor_hidden_dim)
        self.teacher_encoder = copy.deepcopy(student_net.swinViT)
        self.teacher_encoder.load_state_dict(student_net.swinViT.state_dict())
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False

    def _capture_swin_features(
        self,
        net: nn.Module,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        captured: Dict[str, Any] = {}

        def _hook(_module, _inp, out) -> None:
            captured["feats"] = list(out) if isinstance(out, (list, tuple)) else [out]

        handle = net.swinViT.register_forward_hook(_hook)
        try:
            recon = _primary_logits(net(x))
        finally:
            handle.remove()
        hidden_states = captured.get("feats")
        if not hidden_states:
            raise RuntimeError("Student swinViT hook did not capture features")
        return recon, hidden_states

    def forward_student(
        self,
        x_masked: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Tuple[int, int, int]]]:
        recon, hidden_states = self._capture_swin_features(_student_module(self), x_masked)
        spatial = tuple(x_masked.shape[-3:])
        tokens: Dict[str, torch.Tensor] = {}
        grids: Dict[str, Tuple[int, int, int]] = {}
        for stage in self.stages:
            key = f"stage{stage}"
            tok, grid = _extract_swin_stage_tokens(hidden_states, stage, spatial)
            tokens[key] = tok
            grids[key] = grid
        return recon, tokens, grids

    @torch.no_grad()
    def encode_teacher_tokens(
        self,
        x_clean: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Tuple[int, int, int]]]:
        hidden = self.teacher_encoder(x_clean)
        if not isinstance(hidden, (list, tuple)):
            hidden = [hidden]
        spatial = tuple(x_clean.shape[-3:])
        tokens: Dict[str, torch.Tensor] = {}
        grids: Dict[str, Tuple[int, int, int]] = {}
        for stage in self.stages:
            key = f"stage{stage}"
            tok, grid = _extract_swin_stage_tokens(list(hidden), stage, spatial)
            tokens[key] = tok
            grids[key] = grid
        return tokens, grids

    @torch.no_grad()
    def update_teacher_ema(self, momentum: float) -> None:
        m = float(momentum)
        student = _student_module(self)
        for ps, pt in zip(student.swinViT.parameters(), self.teacher_encoder.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)


def build_swin_unetr_inpainting_feature_reconstruction(cfg: Dict[str, Any]) -> SwinUNETRInpaintingFeatureReconstruction:
    mcfg = dict((cfg or {}).get("model") or {})
    fcfg = (cfg or {}).get("feature_reconstruction") or {}
    version = resolve_feature_recon_version(fcfg)
    stages = resolve_feature_stages(fcfg)

    sp = mcfg.get("spatial_size") or mcfg.get("img_size") or [64, 128, 128]
    ut: Dict[str, Any] = {
        "in_channels": int(mcfg.get("in_channels", 1)),
        "out_channels": 1,
        "feature_size": int(mcfg.get("feature_size", 48)),
        "use_checkpoint": bool(mcfg.get("use_checkpoint", True)),
        "img_size": list(sp),
        "swin": dict(mcfg.get("swin") or {}),
    }
    for k in ("depths", "num_heads", "window_size", "patch_size", "drop_rate", "attn_drop_rate"):
        if mcfg.get(k) is not None:
            ut[k] = mcfg[k]

    net = build_swinunetr(ut)
    spatial = tuple(int(x) for x in sp)
    stage_dims = _infer_swin_stage_dims(net.swinViT, spatial, stages)

    print(
        f"[model] architecture=swin_unetr wrapper=SwinUNETRInpaintingFeatureReconstruction",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[feature_recon] stages={stages} stage_dims={stage_dims}",
        file=sys.stderr,
        flush=True,
    )
    for stage in stages:
        key = f"stage{stage}"
        print(
            f"[feature-recon-swin] {key} source: {SWIN_STAGE_SOURCE_DESC.get(stage, f'index {SWIN_STAGE_TO_HIDDEN_IDX[stage]}')} "
            f"(C={stage_dims[key]})",
            file=sys.stderr,
            flush=True,
        )
    print(f"[feature-recon] version={version} stages={stages}", file=sys.stderr, flush=True)

    return SwinUNETRInpaintingFeatureReconstruction(
        net,
        stages=stages,
        stage_dims=stage_dims,
        predictor_hidden_dim=int(fcfg.get("predictor_hidden_dim", 512)),
        version=version,
    )


__all__ = [
    "SWIN_STAGE_TO_HIDDEN_IDX",
    "SWIN_STAGE_SOURCE_DESC",
    "SwinUNETRInpaintingFeatureReconstruction",
    "build_swin_unetr_inpainting_feature_reconstruction",
]
