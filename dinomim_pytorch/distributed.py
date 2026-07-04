"""PyTorch distributed (DDP) helpers for DINO_MIM training scripts."""

from __future__ import annotations

import math
import os
import random
from datetime import timedelta
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0


def get_local_rank() -> int:
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return 0


def distributed_enabled_from_cfg(cfg: dict | None) -> bool:
    dcfg = dict((cfg or {}).get("distributed") or {})
    mode = str(dcfg.get("enabled", "auto")).strip().lower()
    if mode in ("false", "0", "off", "no"):
        return False
    if mode in ("true", "1", "on", "yes"):
        return True
    # auto: use env from torchrun
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed(cfg: dict | None = None) -> tuple[torch.device, bool]:
    """Init NCCL process group when launched via torchrun. Returns (device, use_ddp)."""
    use_ddp = distributed_enabled_from_cfg(cfg)
    if not use_ddp:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, False

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available but distributed.enabled is true")

    local_rank = get_local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if not dist.is_initialized():
        dcfg = dict((cfg or {}).get("distributed") or {})
        backend = str(dcfg.get("backend", "nccl"))
        timeout_min = int(dcfg.get("nccl_timeout_minutes", 30) or 30)
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=max(10, timeout_min)),
        )

    return device, True


def cleanup_distributed() -> None:
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if hasattr(module, "module") else module


def ddp_kwargs_from_cfg(cfg: dict | None) -> dict[str, Any]:
    dcfg = dict((cfg or {}).get("distributed") or {})
    return {
        "device_ids": [get_local_rank()] if torch.cuda.is_available() else None,
        "output_device": get_local_rank() if torch.cuda.is_available() else None,
        "find_unused_parameters": bool(dcfg.get("find_unused_parameters", False)),
        "broadcast_buffers": bool(dcfg.get("broadcast_buffers", False)),
    }


def seed_worker(worker_id: int, base_seed: int = 0) -> None:
    seed = int(base_seed) + worker_id + get_rank() * 1000
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))


def patch_samples_per_rank(global_samples: int, world_size: int) -> int:
    return int(math.ceil(float(global_samples) / float(max(1, world_size))))


def apply_patch_sampler_ddp_data_cfg(data_cfg: dict, *, rank: int, world_size: int) -> dict[str, Any]:
    """Shard random-patch ``samples_per_epoch`` across ranks (global total preserved)."""
    d = dict(data_cfg)
    global_spe = int(d.get("samples_per_epoch", d.get("ssl_samples_per_epoch", 500)))
    d["_global_samples_per_epoch"] = global_spe
    d["samples_per_epoch"] = patch_samples_per_rank(global_spe, world_size)
    d["seed"] = int(d.get("seed", 42)) + int(rank)
    return d


def log_ddp_data_plan(
    *,
    global_samples_per_epoch: int,
    batch_size: int,
    grad_accum_steps: int = 1,
    prefix: str = "[ddp]",
) -> None:
    if not is_main_process():
        return
    ws = get_world_size()
    spr = patch_samples_per_rank(global_samples_per_epoch, ws)
    bpr = max(1, spr // max(1, batch_size))
    eff_bs = batch_size * ws * max(1, grad_accum_steps)
    print(
        f"{prefix} rank={get_rank()}/{ws} local_rank={get_local_rank()} "
        f"global_samples_per_epoch={global_samples_per_epoch} samples_per_rank={spr} "
        f"batches_per_rank={bpr} effective_global_batch_size={eff_bs}",
        flush=True,
    )


def reduce_dict(values: Dict[str, float], average: bool = True) -> Dict[str, float]:
    if not is_dist_avail_and_initialized():
        return dict(values)
    out: Dict[str, float] = {}
    for k, v in values.items():
        t = torch.tensor(float(v), device=torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        out[k] = float(t.item() / get_world_size()) if average else float(t.item())
    return out


def all_reduce_mean(value: float) -> float:
    if not is_dist_avail_and_initialized():
        return float(value)
    t = torch.tensor(float(value), device=torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / get_world_size())


__all__ = [
    "all_reduce_mean",
    "apply_patch_sampler_ddp_data_cfg",
    "cleanup_distributed",
    "ddp_kwargs_from_cfg",
    "distributed_enabled_from_cfg",
    "get_local_rank",
    "get_rank",
    "get_world_size",
    "is_dist_avail_and_initialized",
    "is_main_process",
    "log_ddp_data_plan",
    "patch_samples_per_rank",
    "reduce_dict",
    "seed_worker",
    "setup_distributed",
    "unwrap_module",
]
