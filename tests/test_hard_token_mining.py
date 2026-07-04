"""Unit tests for adaptive hard-token mining helpers."""

from __future__ import annotations

import torch

from dinomim_pytorch.unetrpp_feature_reconstruction import (
    _hard_token_selection_mask,
    apply_objective_preset,
    merge_feature_reconstruction_config,
    resolve_hard_token_mining_for_epoch,
)


def test_error_mass_selects_fewer_than_all_when_mass_low() -> None:
    err = torch.tensor([[0.1, 0.2, 0.3, 0.9, 0.8, 0.05]])
    base = torch.ones_like(err)
    hard, count = _hard_token_selection_mask(
        err,
        base,
        hard_cfg={
            "mode": "error_mass",
            "error_mass_fraction": 0.65,
            "min_tokens": 1,
            "detach_weights": True,
        },
    )
    assert 0 < count < 6
    assert hard.sum() == count


def test_curriculum_relaxes_mining_over_epochs() -> None:
    feat = {
        "loss": "hard_cosine_smoothl1",
        "hard_token_mining": {
            "enabled": True,
            "mode": "topk_error",
            "topk_ratio": 0.5,
            "curriculum_epochs": 10,
            "curriculum_topk_start": 0.9,
        },
    }
    early = resolve_hard_token_mining_for_epoch(feat, 0)["topk_ratio"]
    late = resolve_hard_token_mining_for_epoch(feat, 9)["topk_ratio"]
    assert early > late


def test_objective_preset_boundary() -> None:
    cfg = merge_feature_reconstruction_config({"objective_preset": "boundary"})
    assert cfg["stages"] == [2, 3, 4]
    assert cfg["stage_weights"]["stage4"] == 0.5


def test_objective_preset_overlap() -> None:
    cfg = merge_feature_reconstruction_config({"objective_preset": "overlap"})
    assert cfg["stages"] == [3, 4]
