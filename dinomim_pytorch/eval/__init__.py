from dinomim_pytorch.eval.seg_postprocess import (
    apply_seg_postprocess,
    postprocess_enabled,
    resolve_postprocess_cfg,
)

__all__ = [
    "apply_seg_postprocess",
    "postprocess_enabled",
    "resolve_postprocess_cfg",
]


def __getattr__(name: str):
    if name == "logits_to_label_map":
        from dinomim_pytorch.eval.predict_postprocess import logits_to_label_map

        return logits_to_label_map
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
