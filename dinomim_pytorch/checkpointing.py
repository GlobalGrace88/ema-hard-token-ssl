"""
DINO checkpoint I/O and downstream weight transfer (student encoder by default).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn

from dinomim_pytorch.ssl_downstream_keymap import (
    downstream_key_candidates,
    pairing_candidates_for_load,
)

REPORT_REL = os.path.join("outputs", "logs", "dino_weight_loading_report.txt")
_SSL_STATE_FIELDS: Tuple[str, ...] = (
    "student_backbone",
    "student_net",
    "model",
    "state_dict",
    "backbone",
)


def _report_path(base: Optional[Union[str, Path]] = None) -> Path:
    p = Path(base or ".") / REPORT_REL
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _write_report(text: str, base: Optional[Union[str, Path]] = None) -> None:
    path = _report_path(base)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def _log_ssl(msg: str, *, report_path: Optional[Union[str, Path]] = None) -> None:
    print(msg, flush=True)
    _write_report(msg, report_path)


def _torch_load(path: Union[str, Path]) -> Dict[str, Any]:
    p = str(path)
    try:
        return torch.load(p, map_location="cpu", weights_only=False)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(p, map_location="cpu")


def summarize_key_groups(keys: Iterable[str]) -> Dict[str, int]:
    groups = {
        "unetr_pp_encoder": 0,
        "encoder1": 0,
        "decoder": 0,
        "out": 0,
        "seg_head": 0,
        "dino_head": 0,
        "teacher": 0,
        "student_head": 0,
        "projection": 0,
        "other": 0,
    }
    for k in keys:
        lk = k.lower()
        if "teacher" in lk:
            groups["teacher"] += 1
        elif "student_head" in lk or "dino_head" in lk or "projection_head" in lk:
            groups["student_head"] += 1
        elif "unetr_pp_encoder" in lk:
            groups["unetr_pp_encoder"] += 1
        elif "encoder1" in lk:
            groups["encoder1"] += 1
        elif "decoder" in lk:
            groups["decoder"] += 1
        elif ".out" in lk or lk.startswith("out") or "out1" in lk or "out2" in lk or "out3" in lk or "segmentation_head" in lk:
            groups["out"] += 1
        elif "seg_head" in lk:
            groups["seg_head"] += 1
        elif "head" in lk:
            groups["dino_head"] += 1
        elif "proj" in lk:
            groups["projection"] += 1
        else:
            groups["other"] += 1
    return groups


def _is_encoder_only_ssl_key(key: str) -> bool:
    """True when ``key`` may be loaded under ``load_encoder_only=True``."""
    lk = key.lower()
    if lk.endswith("num_batches_tracked"):
        return False
    if "teacher" in lk or "student_head" in lk or "dino_head" in lk or "projection_head" in lk:
        return False
    if "decoder" in lk:
        return False
    if lk.startswith("out") or "segmentation_head" in lk or "seg_head" in lk:
        return False
    if "unetr_pp_encoder" in lk:
        return True
    if "swinvit" in lk.replace(".", ""):
        return True
    if "encoder1" in lk:
        return True
    return False


def _ssl_key_skip_reason(key: str, *, load_encoder_only: bool) -> Optional[str]:
    """Return a skip reason for ``key``, or ``None`` if it may be considered for loading."""
    lk = key.lower()
    if lk.endswith("num_batches_tracked"):
        return "num_batches_tracked"
    if "teacher" in lk:
        return "teacher"
    if "feature_predictor" in lk:
        return "feature_predictor"
    if "student_head" in lk or "dino_head" in lk or "projection_head" in lk:
        return "dino_head"
    if "proj" in lk and ("head" in lk or "projection" in lk):
        return "projection"
    if load_encoder_only:
        if "decoder" in lk:
            return "decoder"
        if lk.startswith("out") or "segmentation_head" in lk or "seg_head" in lk:
            return "output_head"
        if not _is_encoder_only_ssl_key(key):
            return "non_encoder"
    return None


def _count_top_level_checkpoint(data: Mapping[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for name, val in data.items():
        if isinstance(val, dict):
            counts[name] = sum(1 for v in val.values() if isinstance(v, torch.Tensor))
        elif isinstance(val, torch.Tensor):
            counts[name] = 1
    return counts


def resolve_ssl_state_dict(
    checkpoint: Union[str, Path, Dict[str, Any]],
    *,
    from_teacher_encoder: bool = False,
) -> Tuple[Dict[str, torch.Tensor], str, Dict[str, Any]]:
    """
    Resolve tensor dict + source field from an SSL checkpoint.

    Returns ``(state_dict, field_name, meta)`` where ``meta`` includes top-level key counts.
    """
    ckpt_path = str(checkpoint) if not isinstance(checkpoint, dict) else None
    data = _torch_load(str(checkpoint)) if not isinstance(checkpoint, dict) else checkpoint
    if not isinstance(data, dict):
        return {}, "unknown", {"checkpoint_path": ckpt_path, "top_level_counts": {}}

    top_level_counts = _count_top_level_checkpoint(data)
    meta: Dict[str, Any] = {
        "checkpoint_path": ckpt_path,
        "top_level_counts": top_level_counts,
        "top_level_tensor_keys": sum(top_level_counts.values()),
    }

    if from_teacher_encoder:
        enc = data.get("teacher_backbone")
        if isinstance(enc, dict) and enc:
            sd = {k: v for k, v in enc.items() if isinstance(v, torch.Tensor)}
            return sd, "teacher_backbone", meta

    for field in _SSL_STATE_FIELDS:
        block = data.get(field)
        if isinstance(block, dict) and block and any(isinstance(v, torch.Tensor) for v in block.values()):
            sd = {k: v for k, v in block.items() if isinstance(v, torch.Tensor)}
            return sd, field, meta

    out: Dict[str, torch.Tensor] = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor) and k.startswith("student_backbone."):
            out[k.split("student_backbone.", 1)[-1]] = v
    if out:
        return out, "student_backbone.", meta

    bb = data.get("backbone")
    if (
        isinstance(bb, dict)
        and bb
        and data.get("student_backbone") is None
        and (
            isinstance(data.get("online_proj"), dict)
            or isinstance(data.get("online_pred"), dict)
            or isinstance(data.get("proj_tgt"), dict)
        )
    ):
        sd = {k: v for k, v in bb.items() if isinstance(v, torch.Tensor)}
        return sd, "backbone", meta

    return {}, "none", meta


def extract_student_encoder_state_dict(
    checkpoint: Union[str, Path, Dict[str, Any]], prefix: str = "student_backbone"
) -> Dict[str, torch.Tensor]:
    """
    Encoder tensors for downstream init.

    Supports **DINO** checkpoints (``student_backbone`` / ``student_backbone.*``) and
    **byol_paper** checkpoints (top-level ``backbone`` next to ``online_proj`` / ``proj_tgt``, etc.).
    """
    sd, _, _ = resolve_ssl_state_dict(checkpoint, from_teacher_encoder=False)
    return sd


def extract_teacher_encoder_state_dict(
    checkpoint: Union[str, Path, Dict[str, Any]]
) -> Dict[str, torch.Tensor]:
    sd, _, _ = resolve_ssl_state_dict(checkpoint, from_teacher_encoder=True)
    return sd


def _adapt_tensor_in_channels(tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Average or repeat input-channel dim for stem conv weights (e.g. 4ch SSL -> 1ch downstream)."""
    if tensor.shape == target.shape:
        return tensor
    if tensor.ndim != target.ndim or tensor.shape[0] != target.shape[0]:
        return tensor
    if tensor.shape[1] == target.shape[1]:
        return tensor
    # Conv3d weight: (out, in, d, h, w)
    src_in, dst_in = int(tensor.shape[1]), int(target.shape[1])
    if src_in > dst_in:
        if src_in % dst_in == 0:
            g = src_in // dst_in
            chunks = [tensor[:, i * g:(i + 1) * g] for i in range(dst_in)]
            return torch.stack([c.mean(dim=1) for c in chunks], dim=1)
        return tensor[:, :dst_in]
    # dst_in > src_in: repeat first channel
    reps = [tensor]
    while sum(t.shape[1] for t in reps) < dst_in:
        reps.append(tensor[:, :1])
    merged = torch.cat(reps, dim=1)[:, :dst_in]
    return merged


def load_dino_weights_into_downstream_model(
    model: nn.Module,
    checkpoint: Union[str, Path, Dict[str, Any]],
    *,
    from_teacher_encoder: bool = False,
    load_encoder_only: bool = True,
    strict_load: bool = False,
    adapt_input_channels: bool = False,
    report_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    use_strict = strict_load
    ckpt_path = str(checkpoint) if not isinstance(checkpoint, dict) else "<dict>"
    sd, field_used, meta = resolve_ssl_state_dict(
        checkpoint, from_teacher_encoder=from_teacher_encoder
    )

    raw_keys = list(sd.keys())
    raw_groups = summarize_key_groups(raw_keys)
    _log_ssl(f"[ssl-load] checkpoint path: {ckpt_path}", report_path=report_path)
    _log_ssl(f"[ssl-load] checkpoint field used: {field_used}", report_path=report_path)
    _log_ssl(
        f"[ssl-load] top-level checkpoint tensor counts: {meta.get('top_level_counts', {})}",
        report_path=report_path,
    )
    _log_ssl(f"[ssl-load] total raw checkpoint keys: {len(raw_keys)}", report_path=report_path)
    _log_ssl(f"[ssl-load] raw key groups: {raw_groups}", report_path=report_path)
    _log_ssl(f"[ssl-load] encoder_only={load_encoder_only}", report_path=report_path)

    if not sd:
        _write_report("extract student/teacher encoder: empty; no weights applied.", report_path)
        if use_strict:
            raise RuntimeError("No encoder state in DINO checkpoint")
        warnings.warn("DINO checkpoint had no loadable SSL state dict.", UserWarning, stacklevel=2)
        return {
            "missing": list(model.state_dict().keys()),
            "unexpected": [],
            "loaded": 0,
            "ok": False,
            "ssl_loaded": False,
            "field_used": field_used,
            "raw_key_groups": raw_groups,
            "loaded_key_groups": summarize_key_groups([]),
        }

    model_sd = model.state_dict()
    selected_keys: List[str] = []
    skipped_decoder: List[str] = []
    skipped_output: List[str] = []
    skipped_teacher_dino: List[str] = []
    skipped_non_encoder: List[str] = []
    skipped_bn_buffers: List[str] = []
    skipped_shape: List[str] = []
    skipped_not_in_model: List[str] = []

    to_load: Dict[str, torch.Tensor] = {}
    ssl_to_downstream: Dict[str, str] = {}

    for k, v in sd.items():
        reason = _ssl_key_skip_reason(k, load_encoder_only=load_encoder_only)
        if reason == "num_batches_tracked":
            skipped_bn_buffers.append(k)
            continue
        if reason == "decoder":
            skipped_decoder.append(k)
            continue
        if reason == "output_head":
            skipped_output.append(k)
            continue
        if reason in ("teacher", "dino_head", "projection", "feature_predictor"):
            skipped_teacher_dino.append(k)
            continue
        if reason == "non_encoder":
            skipped_non_encoder.append(k)
            continue

        selected_keys.append(k)
        placed = False
        matched_cand: Optional[str] = None
        for vk in downstream_key_candidates(k):
            for cand in pairing_candidates_for_load(k, vk):
                if cand in model_sd:
                    target = model_sd[cand]
                    load_v = v
                    if adapt_input_channels and load_v.shape != target.shape:
                        adapted = _adapt_tensor_in_channels(load_v, target)
                        if adapted.shape == target.shape:
                            _log_ssl(
                                f"[ssl-load] adapted channels {k} {tuple(load_v.shape)} -> {tuple(adapted.shape)}",
                                report_path=report_path,
                            )
                            load_v = adapted
                    if target.shape == load_v.shape:
                        to_load[cand] = load_v
                    ssl_to_downstream[k] = cand
                    placed = True
                    matched_cand = cand
                    break
            if placed:
                break
        if not placed:
            if any(cand in model_sd for vk in downstream_key_candidates(k) for cand in pairing_candidates_for_load(k, vk)):
                skipped_shape.append(k)
            else:
                skipped_not_in_model.append(k)

    selected_groups = summarize_key_groups(selected_keys)
    loaded_ssl_keys = list(ssl_to_downstream.keys())
    loaded_groups = summarize_key_groups(loaded_ssl_keys)

    _log_ssl(
        f"[ssl-load] selected key groups before shape matching: {selected_groups}",
        report_path=report_path,
    )
    _log_ssl(
        f"[ssl-load] loaded key groups after shape matching: {loaded_groups}",
        report_path=report_path,
    )
    _log_ssl(f"[ssl-load] selected keys for loading: {len(selected_keys)}", report_path=report_path)
    _log_ssl(f"[ssl-load] actually loaded keys: {len(to_load)}", report_path=report_path)
    _log_ssl(f"[ssl-load] skipped decoder keys: {len(skipped_decoder)}", report_path=report_path)
    _log_ssl(f"[ssl-load] skipped output/head keys: {len(skipped_output)}", report_path=report_path)
    _log_ssl(f"[ssl-load] skipped teacher/dino keys: {len(skipped_teacher_dino)}", report_path=report_path)
    _log_ssl(f"[ssl-load] skipped num_batches_tracked keys: {len(skipped_bn_buffers)}", report_path=report_path)
    _log_ssl(f"[ssl-load] skipped non-encoder keys: {len(skipped_non_encoder)}", report_path=report_path)
    _log_ssl(f"[ssl-load] skipped shape mismatch keys: {len(skipped_shape)}", report_path=report_path)
    _log_ssl(f"[ssl-load] skipped key not found in downstream: {len(skipped_not_in_model)}", report_path=report_path)

    preview_loaded = sorted(ssl_to_downstream.items())[:20]
    if preview_loaded:
        _log_ssl("[ssl-load] first loaded keys (ssl -> downstream):", report_path=report_path)
        for sk, dk in preview_loaded:
            _log_ssl(f"  {sk} -> {dk}", report_path=report_path)

    def _preview(title: str, keys: List[str]) -> None:
        if not keys:
            return
        _log_ssl(f"[ssl-load] first skipped {title} keys:", report_path=report_path)
        for k in sorted(keys)[:20]:
            _log_ssl(f"  {k}", report_path=report_path)

    _preview("decoder", skipped_decoder)
    _preview("output/head", skipped_output)
    _preview("shape mismatch", skipped_shape)

    if load_encoder_only and (loaded_groups.get("decoder", 0) or loaded_groups.get("out", 0) or loaded_groups.get("seg_head", 0)):
        msg = (
            f"encoder_only=True but loaded non-encoder groups: {loaded_groups}. "
            "This should not happen; check _is_encoder_only_ssl_key()."
        )
        _log_ssl(f"[ssl-load] ERROR: {msg}", report_path=report_path)
        if use_strict:
            raise RuntimeError(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

    if not to_load and not use_strict:
        msg = "No tensor shapes matched between SSL checkpoint and downstream model."
        _write_report(msg, report_path)
        warnings.warn(msg + " (strict_load=False; continuing).", UserWarning, stacklevel=2)
        return {
            "missing": list(model_sd.keys()),
            "unexpected": [],
            "loaded": 0,
            "ok": False,
            "ssl_loaded": False,
            "field_used": field_used,
            "raw_key_groups": raw_groups,
            "selected_key_groups": selected_groups,
            "loaded_key_groups": loaded_groups,
        }
    if not to_load and use_strict:
        raise RuntimeError("No keys matched; cannot load (strict).")

    miss, unexp = model.load_state_dict(to_load, strict=False)  # type: ignore[assignment]
    r = (
        f"load_dino: loaded={len(to_load)} missing={len(miss)} unexpected={len(unexp)} "
        f"field={field_used} encoder_only={load_encoder_only}"
    )
    _log_ssl(f"[checkpointing] {r}", report_path=report_path)
    _log_ssl(f"[ssl-load] missing keys in downstream: {len(miss)}", report_path=report_path)
    _log_ssl(f"[ssl-load] unexpected keys: {len(unexp)}", report_path=report_path)
    _write_report(
        f"{r} from_teacher={from_teacher_encoder} missing_keys={miss!r} unexpected_keys={unexp!r}",
        report_path,
    )
    return {
        "missing": list(miss),
        "unexpected": list(unexp),
        "loaded": len(to_load),
        "ok": len(to_load) > 0,
        "ssl_loaded": len(to_load) > 0,
        "field_used": field_used,
        "raw_key_groups": raw_groups,
        "selected_key_groups": selected_groups,
        "loaded_key_groups": loaded_groups,
        "skipped_decoder": len(skipped_decoder),
        "skipped_output_head": len(skipped_output),
        "skipped_shape_mismatch": len(skipped_shape),
    }


# Back-compat alias
load_dino_weights_into_downstream = load_dino_weights_into_downstream_model


def save_dino_checkpoint(
    path: Union[str, Path],
    *,
    student_backbone: nn.Module,
    student_head: nn.Module,
    teacher_backbone: Optional[nn.Module] = None,
    teacher_head: Optional[nn.Module] = None,
    dino_loss: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    global_step: Optional[int] = None,
    scaler_state: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "student_backbone": student_backbone.state_dict(),
        "student_head": student_head.state_dict(),
    }
    if teacher_backbone is not None:
        payload["teacher_backbone"] = teacher_backbone.state_dict()
    if teacher_head is not None:
        payload["teacher_head"] = teacher_head.state_dict()
    if dino_loss is not None and hasattr(dino_loss, "state_dict"):
        payload["dino_loss"] = dino_loss.state_dict()
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    if global_step is not None:
        payload["global_step"] = int(global_step)
    if scaler_state is not None:
        payload["scaler"] = scaler_state
    if extra:
        payload["extra"] = extra
    torch.save(payload, str(path))


def load_dino_checkpoint(
    path: Union[str, Path], map_location: Optional[str] = None
) -> Dict[str, Any]:
    m = str(map_location or "cpu")
    try:
        return torch.load(str(path), map_location=m, weights_only=False)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(str(path), map_location=m)


__all__ = [
    "save_dino_checkpoint",
    "load_dino_checkpoint",
    "extract_student_encoder_state_dict",
    "extract_teacher_encoder_state_dict",
    "resolve_ssl_state_dict",
    "summarize_key_groups",
    "load_dino_weights_into_downstream_model",
    "REPORT_REL",
]
