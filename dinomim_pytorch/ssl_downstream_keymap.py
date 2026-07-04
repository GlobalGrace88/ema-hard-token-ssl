"""
Map SSL encoder checkpoint keys (MONAI adapter wrappers) to downstream segmentation modules.

Aligned with MAE_BYOL MAE_v3 strategy: separate SSL runs per encoder family (ViT/UNETR, CNN/U-Net,
Swin/SwinUNETR), then load the matching checkpoint into the downstream architecture.
"""

from __future__ import annotations

from typing import List


def map_ssl_key(ssl_k: str) -> str:
    """Strip common prefixes from SSL tensors (``student_backbone.*``)."""
    for p in ("backbone.", "net.", "model.", "swinViT.", "unetr."):
        if ssl_k.startswith(p):
            return ssl_k[len(p) :]
    return ssl_k


def downstream_key_candidates(ssl_k: str) -> List[str]:
    """
    Expand one SSL encoder key into names that may appear in a MONAI segmentation ``state_dict``.

    Handles ``m.*``, ``unet.*``, ``swin.*`` vs ``swinViT.*`` (same as BYOL path).
    """
    seen: list[str] = []
    pending: list[str] = [ssl_k]

    def add(x: str) -> None:
        if x and x not in seen:
            seen.append(x)

    while pending:
        k = pending.pop()
        add(k)
        add(map_ssl_key(k))
        if k.startswith("m.") and len(k) > 2:
            pending.append(k[2:])
        elif k.startswith("unet.") and len(k) > 5:
            pending.append(k[5:])
        elif k.startswith("m.unet.") and len(k) > 7:
            pending.append(k[7:])
        elif k.startswith("swin.") and len(k) > 5:
            pending.append("swinViT." + k[5:])
        elif k.startswith("m.swin.") and len(k) > 7:
            pending.append("swinViT." + k[7:])
    return seen


def pairing_candidates_for_load(ssl_k: str, vk: str) -> List[str]:
    mapped = map_ssl_key(vk)
    out: list[str] = []
    for cand in (
        vk,
        mapped,
        "backbone." + vk,
        "backbone." + mapped,
        "net." + mapped,
    ):
        if cand and cand not in out:
            out.append(cand)
    return out


__all__ = ["downstream_key_candidates", "map_ssl_key", "pairing_candidates_for_load"]
