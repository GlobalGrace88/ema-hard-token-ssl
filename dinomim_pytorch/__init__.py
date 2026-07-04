"""
DINOMiM: Multi-View Masked DINO for medical imaging (2D/3D).
Heavy imports are lazy so CLIs can run config checks without loading PyTorch first.
"""


def __getattr__(name: str):
    if name == "MultiViewMaskedDINO":
        from dinomim_pytorch.dino import MultiViewMaskedDINO

        return MultiViewMaskedDINO
    if name == "DINOLoss":
        from dinomim_pytorch.dino_loss import DINOLoss

        return DINOLoss
    if name == "build_ssl_backbone":
        from dinomim_pytorch.medical_backbones import build_ssl_backbone

        return build_ssl_backbone
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MultiViewMaskedDINO",
    "DINOLoss",
    "build_ssl_backbone",
]
