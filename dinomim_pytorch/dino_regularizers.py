"""Anti-collapse regularizers for small-batch / multi-view DINO."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def koleo_loss(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    KoLeo-style spread loss on L2-normalized rows of ``features`` [N, D].

    Encourages different views / samples to occupy different directions on the sphere.
    Useful when batch_size=1 but multiple student crops exist per step.
    """
    if features.dim() != 2 or features.shape[0] < 2:
        return features.new_tensor(0.0)
    x = F.normalize(features.float(), dim=-1, p=2, eps=eps)
    n = x.shape[0]
    sim = x @ x.t()
    sim.fill_diagonal_(-torch.inf)
    nn_idx = sim.argmax(dim=1)
    nn = x[nn_idx]
    dist = (x - nn).norm(dim=1).clamp_min(eps)
    return torch.log(dist).mean()


__all__ = ["koleo_loss"]
