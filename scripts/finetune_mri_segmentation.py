from __future__ import annotations

import argparse
import math
from typing import Any
import csv
import itertools
import warnings
from contextlib import nullcontext
from pathlib import Path

import _bootstrap  # noqa: F401
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dinomim_pytorch.config_utils import load_yaml
from dinomim_pytorch.datasets import unwrap_monai_dict_batch
from dinomim_pytorch.datasets.nnformer_npz_patch import patch_sampler_enabled
from dinomim_pytorch.datasets.seg_dataset_factory import (
    build_segmentation_dataset,
    has_segmentation_data,
)
from dinomim_pytorch.segmentation_models import (
    build_segmentation_model,
    get_merged_model_config,
)

_DINO_MIM_ROOT = Path(__file__).resolve().parent.parent  # release repo root


def _resolve_ssl_checkpoint(cfg: dict[str, Any], *, cli_path: str | None) -> None:
    """Set ``model.ssl_checkpoint`` to an existing file path (absolute).

    Tries, in order: ``--ssl-checkpoint`` CLI; absolute YAML path; YAML path
    relative to cwd; same path relative to DINO_MIM repo root; in that directory,
    ``best.pt`` / ``last.pt`` if the named file is missing.
    """
    model = cfg.setdefault("model", {})
    raw = (cli_path or str(model.get("ssl_checkpoint") or "")).strip()
    if not raw:
        return
    p = Path(raw).expanduser()
    if p.is_absolute():
        if not p.is_file():
            raise FileNotFoundError(
                f"model.ssl_checkpoint not found: {p}\n"
                "Set an absolute path to best.pt or last.pt from byol_paper pretrain."
            )
        model["ssl_checkpoint"] = str(p.resolve())
        return
    tried: list[str] = []
    bases = [Path.cwd(), _DINO_MIM_ROOT]
    rel = p
    for base in bases:
        cand = (base / rel).resolve()
        parent = cand.parent
        for name in (cand.name, "best.pt", "last.pt"):
            c2 = parent / name
            key = str(c2)
            if key in tried:
                continue
            tried.append(key)
            if c2.is_file():
                model["ssl_checkpoint"] = str(c2)
                return
    raise FileNotFoundError(
        f"model.ssl_checkpoint not found: {raw!r}\n"
        f"  Tried under cwd={Path.cwd()} and DINO_MIM={_DINO_MIM_ROOT} (plus best.pt/last.pt in the same folder).\n"
        "  Run paper BYOL pretrain (byol_paper) or set --ssl-checkpoint / model.ssl_checkpoint to an absolute path."
    )
from dinomim_pytorch.segmentation_models.losses import (
    align_logits_to_labels,
    build_segmentation_loss,
    compute_segmentation_loss,
    primary_segmentation_logits,
)
from dinomim_pytorch.eval import logits_to_label_map, postprocess_enabled, resolve_postprocess_cfg
from dinomim_pytorch.segmentation_models.class_labels import (
    format_per_class_metric_line,
    resolve_segmentation_class_names,
)
from dinomim_pytorch.segmentation_models.metrics import (
    brats_wt_tc_et_dice,
    dice_iou_foreground_per_class,
)
from dinomim_pytorch.segmentation_models.val_metrics_accum import (
    val_confusion_update,
    val_dice_from_accumulated,
)

try:
    from tqdm.auto import tqdm
except ImportError:

    _HAS_REAL_TQDM = False

    def tqdm(it, **kwargs):  # type: ignore[misc,no-redef]
        return it

else:
    _HAS_REAL_TQDM = True


def _train_log(msg: str) -> None:
    if _HAS_REAL_TQDM:
        tqdm.write(msg)
    else:
        print(msg)


def _save_finetune_checkpoint(
    out_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch_done: int,
    global_step: int,
    filename: str,
) -> None:
    """Same payload shape as BYOL downstream (``model`` / ``optimizer`` / counters). Eval accepts ``model`` key."""
    from dinomim_pytorch.distributed import unwrap_module

    out_dir.mkdir(parents=True, exist_ok=True)
    core = unwrap_module(model)
    torch.save(
        {
            "model": core.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch_done,
            "global_step": global_step,
        },
        str(out_dir / filename),
    )


def _data_with_crop_from_model(cfg: dict) -> dict:
    """MONAI BraTS compose defaults ``image_size`` to 96³; UNETR ViT pos-embeddings need crop == ``model.img_size``."""
    d = dict((cfg or {}).get("data") or {})
    if d.get("image_size"):
        return d
    m = (cfg or {}).get("model") or {}
    sz = m.get("img_size") or m.get("spatial_size")
    if isinstance(sz, (list, tuple)) and len(sz) >= 3:
        d["image_size"] = [int(sz[0]), int(sz[1]), int(sz[2])]
    elif isinstance(sz, (int, float)) and int(sz) > 0:
        v = int(sz)
        d["image_size"] = [v, v, v]
    return d


def _val_metric_accumulated(cfg: dict) -> bool:
    data = (cfg or {}).get("data") or {}
    mode = str(data.get("val_metric_mode", "auto")).strip().lower()
    if mode in ("accumulated", "tpfpfn", "global", "mae"):
        return True
    if mode in ("per_batch", "legacy", "batch"):
        return False
    return patch_sampler_enabled(data)


def _val_postprocess_enabled(cfg: dict) -> bool:
    data = (cfg or {}).get("data") or {}
    if "val_postprocess" in data:
        return bool(data["val_postprocess"])
    if patch_sampler_enabled(data):
        return False
    return postprocess_enabled(cfg)


def _build_finetune_optimizer(
    params,
    train_cfg: dict[str, Any],
) -> torch.optim.Optimizer:
    name = str(train_cfg.get("optimizer", "adamw")).strip().lower()
    lr = float(train_cfg.get("lr", 1e-4))
    wd = float(train_cfg.get("weight_decay", 0.0))
    if name in ("sgd", "unetrpp", "unetr_pp"):
        return torch.optim.SGD(
            params,
            lr=lr,
            weight_decay=wd,
            momentum=float(train_cfg.get("momentum", 0.99)),
            nesterov=bool(train_cfg.get("nesterov", True)),
        )
    if name == "adam":
        betas = train_cfg.get("betas", (0.9, 0.999))
        return torch.optim.Adam(params, lr=lr, weight_decay=wd, betas=tuple(betas))
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)


def _poly_lr_epoch(
    epoch_idx: int,
    total_epochs: int,
    initial_lr: float,
    exponent: float,
) -> float:
    if total_epochs <= 1:
        return float(initial_lr)
    progress = float(epoch_idx) / float(max(1, total_epochs - 1))
    return float(initial_lr) * (1.0 - progress) ** float(exponent)


def _make_loader(
    cfg: dict,
    *,
    train: bool,
    index_csv_override: str | None = None,
    use_ddp: bool = False,
) -> DataLoader | None:
    if not has_segmentation_data(cfg, train=train, index_csv_override=index_csv_override):
        return None
    ds = build_segmentation_dataset(
        cfg, train=train, index_csv_override=index_csv_override
    )
    if ds is None:
        return None
    train_cfg = (cfg or {}).get("training") or {}
    data_cfg = (cfg or {}).get("data") or {}
    shuffle = bool(train) and not use_ddp and not patch_sampler_enabled(data_cfg)
    sampler = None
    if use_ddp and train and not patch_sampler_enabled(data_cfg):
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(ds, shuffle=True)
        shuffle = False
    return DataLoader(
        ds,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=shuffle,
        sampler=sampler,
        drop_last=bool(train_cfg.get("drop_last", False)) if train else False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def _finetune_amp_settings(
    train_cfg: dict[str, Any], device: torch.device
) -> tuple[bool, torch.dtype | None, bool]:
    """``(use_amp, autocast_dtype, use_grad_scaler)`` for CUDA training.

    ``use_grad_scaler`` is true only for float16 (bf16 does not need scaling).
    Validation always runs in full precision — fp16 autocast there was unstable
    for UNETR/ViT while the training loop previously stayed in FP32.
    """
    if device.type != "cuda" or not bool(train_cfg.get("mixed_precision", False)):
        return False, None, False
    raw = str(train_cfg.get("amp_dtype", "auto")).strip().lower()
    if raw in ("bf16", "bfloat16"):
        return True, torch.bfloat16, False
    if raw in ("fp16", "float16"):
        return True, torch.float16, True
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16, False
    return True, torch.float16, True


def _run_validation_accumulated(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    ik: str,
    lk: str,
    loss_fn: torch.nn.Module,
    n_classes: int,
    report_brats_regions: bool,
    report_per_class_dice: bool,
    class_names: list[str] | None,
) -> dict[str, float]:
    model.eval()
    sum_loss = 0.0
    n_batches = 0
    tp = torch.zeros(n_classes, dtype=torch.float64, device=device)
    fp = torch.zeros(n_classes, dtype=torch.float64, device=device)
    fn = torch.zeros(n_classes, dtype=torch.float64, device=device)
    sum_wt = sum_tc = sum_et = 0.0
    n_brats = 0

    with torch.no_grad():
        for batch in loader:
            batch = unwrap_monai_dict_batch(batch)
            if not isinstance(batch, dict):
                continue
            x = batch.get(ik, batch.get("image", batch.get("path")))
            y = batch.get(lk, batch.get("label", batch.get("label_path")))
            if x is None or y is None:
                continue
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x, dtype=torch.float32)
            if not isinstance(y, torch.Tensor):
                y = torch.as_tensor(y, dtype=torch.long)
            x, y = x.to(device), y.to(device)
            y = y.long()
            if y.dim() == 4:
                y_loss = y.unsqueeze(1)
            elif y.dim() == 5 and y.size(1) == 1:
                y_loss = y
                y = y[:, 0]
            else:
                raise ValueError(f"Expected label [B,D,H,W] or [B,1,D,H,W], got {tuple(y.shape)}")
            raw_out = model(x)
            out = align_logits_to_labels(primary_segmentation_logits(raw_out), y_loss)
            L = compute_segmentation_loss(loss_fn, raw_out, y_loss)
            sum_loss += float(L.detach().cpu())
            n_batches += 1
            pred = out.argmax(dim=1)
            val_confusion_update(pred, y, n_classes, tp, fp, fn)
            if report_brats_regions and n_classes == 4:
                y_b = y.unsqueeze(1) if y.dim() == 4 else y
                d_wt, d_tc, d_et = brats_wt_tc_et_dice(out, y_b, softmax=True)
                sum_wt += float(d_wt.detach().cpu())
                sum_tc += float(d_tc.detach().cpu())
                sum_et += float(d_et.detach().cpu())
                n_brats += 1

    mean_dice, per_class, note = val_dice_from_accumulated(
        tp, fp, fn, n_classes, class_names=class_names
    )
    out_d: dict[str, float] = {
        "val_loss": sum_loss / max(1, n_batches),
        "val_mean_dice": mean_dice,
    }
    if report_per_class_dice:
        out_d.update(per_class)
    out_d["val_macro_note"] = note  # type: ignore[assignment]
    if n_brats > 0:
        out_d["val_dice_wt"] = sum_wt / n_brats
        out_d["val_dice_tc"] = sum_tc / n_brats
        out_d["val_dice_et"] = sum_et / n_brats
    model.train()
    return out_d


def _run_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    ik: str,
    lk: str,
    loss_fn: torch.nn.Module,
    n_classes: int,
    report_brats_regions: bool,
    report_per_class_dice: bool = False,
    class_names: list[str] | None = None,
    cfg: dict[str, Any] | None = None,
    use_accumulated: bool = False,
) -> dict[str, float]:
    if use_accumulated:
        return _run_validation_accumulated(
            model,
            loader,
            device,
            ik=ik,
            lk=lk,
            loss_fn=loss_fn,
            n_classes=n_classes,
            report_brats_regions=report_brats_regions,
            report_per_class_dice=report_per_class_dice,
            class_names=class_names,
        )

    pp_cfg = resolve_postprocess_cfg(cfg or {})
    use_pp = _val_postprocess_enabled(cfg or {})
    model.eval()
    sum_loss = 0.0
    n_batches = 0
    sum_dice = 0.0
    sum_wt = sum_tc = sum_et = 0.0
    n_brats = 0
    n_fg = max(0, n_classes - 1)
    sum_dice_per_class: list[float] = [0.0] * n_fg
    # Full FP32/BF16 weights + loss in default dtypes; avoids fp16 autocast NaNs on ViT.
    with torch.no_grad():
        for batch in loader:
            batch = unwrap_monai_dict_batch(batch)
            if not isinstance(batch, dict):
                continue
            x = batch.get(ik, batch.get("image", batch.get("path")))
            y = batch.get(lk, batch.get("label", batch.get("label_path")))
            if x is None or y is None:
                continue
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x, dtype=torch.float32)
            if not isinstance(y, torch.Tensor):
                y = torch.as_tensor(y, dtype=torch.long)
            x, y = x.to(device), y.to(device)
            y = y.long()
            if y.dim() == 4:
                y = y.unsqueeze(1)
            elif y.dim() == 5 and y.shape[1] != 1:
                raise ValueError(f"Expected label [B,D,H,W] or [B,1,D,H,W], got {tuple(y.shape)}")
            with torch.no_grad():
                raw_out = model(x)
                out = align_logits_to_labels(primary_segmentation_logits(raw_out), y)
                L = compute_segmentation_loss(loss_fn, raw_out, y)
            sum_loss += float(L.detach().cpu())
            n_batches += 1
            y_s = y.squeeze(1) if y.dim() == 5 else y
            if use_pp:
                pred = logits_to_label_map(out, n_classes, pp_cfg)
                d_vec, _ = dice_iou_foreground_per_class(
                    pred, y_s, n_classes, softmax=False
                )
                sum_dice += float(d_vec.mean().detach().cpu())
                if report_brats_regions and n_classes == 4:
                    d_wt, d_tc, d_et = brats_wt_tc_et_dice(pred, y_s, softmax=False)
            else:
                d_vec, _ = dice_iou_foreground_per_class(out, y_s, n_classes, softmax=True)
                sum_dice += float(d_vec.mean().detach().cpu())
                if report_brats_regions and n_classes == 4:
                    d_wt, d_tc, d_et = brats_wt_tc_et_dice(out, y_s, softmax=True)
            if report_per_class_dice and n_fg > 0:
                for i in range(n_fg):
                    sum_dice_per_class[i] += float(d_vec[i].detach().cpu())
            if report_brats_regions and n_classes == 4:
                sum_wt += float(d_wt.detach().cpu())
                sum_tc += float(d_tc.detach().cpu())
                sum_et += float(d_et.detach().cpu())
                n_brats += 1
    out_d: dict[str, float] = {
        "val_loss": sum_loss / max(1, n_batches),
        "val_mean_dice": sum_dice / max(1, n_batches),
    }
    if n_brats > 0:
        out_d["val_dice_wt"] = sum_wt / n_brats
        out_d["val_dice_tc"] = sum_tc / n_brats
        out_d["val_dice_et"] = sum_et / n_brats
    if report_per_class_dice and n_fg > 0 and n_batches > 0:
        names = class_names or [f"class_{i}" for i in range(n_classes)]
        for i in range(n_fg):
            c = i + 1
            slug = (names[c] if c < len(names) else f"class_{c}").replace(" ", "_")
            out_d[f"val_dice_c{c:02d}_{slug}"] = sum_dice_per_class[i] / n_batches
    model.train()
    return out_d


def _log_validation_metrics(
    vm: dict[str, float],
    *,
    report_brats_regions: bool,
    report_per_class_dice: bool,
    class_names: list[str],
    n_classes: int,
) -> None:
    if report_brats_regions and "val_dice_wt" in vm:
        _train_log(
            "[finetune] val  "
            f"WT={vm['val_dice_wt']:.4f} | TC={vm['val_dice_tc']:.4f} | ET={vm['val_dice_et']:.4f} | "
            f"mean_dice={vm['val_mean_dice']:.4f} | loss={vm['val_loss']:.4f}"
        )
    else:
        note = str(vm.get("val_macro_note", "") or "")
        _train_log(
            "[finetune] val  "
            f"mean_dice={vm['val_mean_dice']:.4f} | loss={vm['val_loss']:.4f}{note}"
        )
    if not report_per_class_dice or n_classes < 3:
        return
    dice_vals: list[float] = []
    for c in range(1, n_classes):
        dkey = next((k for k in vm if k.startswith(f"val_dice_c{c:02d}_")), None)
        if dkey is None:
            continue
        dice_vals.append(float(vm[dkey]))
    if dice_vals:
        _train_log(
            format_per_class_metric_line("[finetune] val  Dice", dice_vals, class_names)
        )


_VAL_CSV_FIELDS = (
    "epoch",
    "val_loss",
    "val_mean_dice",
    "val_dice_wt",
    "val_dice_tc",
    "val_dice_et",
)


def _append_val_metrics_csv(out_dir: Path, epoch: int, metrics: dict[str, float]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "val_metrics.csv"
    row: dict[str, Any] = {"epoch": int(epoch)}
    for k in _VAL_CSV_FIELDS:
        if k == "epoch":
            continue
        row[k] = metrics.get(k, "")
    for k, v in sorted(metrics.items()):
        if k.startswith("val_dice_c"):
            row[k] = float(v)
    fieldnames = list(_VAL_CSV_FIELDS)
    for k in sorted(row):
        if k not in fieldnames:
            fieldnames.append(k)
    write_header = not path.is_file()
    if path.is_file() and not write_header:
        with open(path, newline="", encoding="utf-8") as fh:
            existing = (csv.DictReader(fh).fieldnames or [])
        for k in fieldnames:
            if k not in existing:
                fieldnames = list(existing) + [x for x in fieldnames if x not in existing]
                break
        else:
            fieldnames = list(existing)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def main():
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    from dinomim_pytorch.finetune_checkpointing import FinetuneCheckpointConfig, FinetuneCheckpointManager, load_finetune_resume_checkpoint
    from dinomim_pytorch.distributed import (
        all_reduce_mean,
        apply_patch_sampler_ddp_data_cfg,
        cleanup_distributed,
        ddp_kwargs_from_cfg,
        get_rank,
        get_world_size,
        is_main_process,
        log_ddp_data_plan,
        setup_distributed,
    )

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"monai\.utils\.deprecate_utils",
    )
    ap = argparse.ArgumentParser(
        description="Downstream 3D segmentation finetune (DINO_MIM). "
        "Matches BYOL ``finetune_mri_segmentation`` loop, checkpoints, and CSV batch keys.",
    )
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument(
        "--ssl-checkpoint",
        type=str,
        default=None,
        metavar="PATH",
        help="Override model.ssl_checkpoint (best.pt or last.pt from byol_paper pretrain).",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        metavar="N",
        help="Override training.epochs (full passes). Mutually exclusive with --steps smoke mode.",
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=None,
        metavar="N",
        help="Smoke: stop after N optimizer steps (single-epoch fragment).",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Override output.dir from YAML (checkpoints, val_metrics.csv, logs).",
    )
    ap.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="PATH",
        help="Resume from a finetune checkpoint (last_model.pt, epoch_*.pt, best_model.pt). "
        "Restores model, optimizer, scaler, epoch counter, and global_step.",
    )
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if args.output_dir:
        cfg.setdefault("output", {})["dir"] = str(args.output_dir)
    _resolve_ssl_checkpoint(cfg or {}, cli_path=args.ssl_checkpoint)

    device, use_ddp = setup_distributed(cfg)
    debug = dict((cfg or {}).get("debug") or {})
    debug_enabled = bool(debug.get("enabled", False))
    save_ckpt = (not debug_enabled or bool(debug.get("save_checkpoints", True))) and is_main_process()

    data_cfg = dict((cfg or {}).get("data") or {})
    global_train_samples = int(data_cfg.get("samples_per_epoch", data_cfg.get("ssl_samples_per_epoch", 250)))
    if use_ddp and patch_sampler_enabled(data_cfg):
        cfg["data"] = apply_patch_sampler_ddp_data_cfg(
            data_cfg, rank=get_rank(), world_size=get_world_size()
        )
        data_cfg = dict(cfg["data"])

    train_cfg = (cfg or {}).get("training") or {}
    yaml_epochs = int(train_cfg.get("epochs", 1))

    max_train_batches = None
    max_val_batches = None
    max_batches_per_epoch: int | None = None
    if debug_enabled:
        max_batches_per_epoch = int(debug.get("max_train_batches_per_epoch", 10))
        max_val_batches = debug.get("max_val_batches")
        if max_val_batches is not None:
            max_val_batches = int(max_val_batches)
        train_cfg["epochs"] = int(debug.get("epochs", train_cfg.get("epochs", 2)))
        cfg["training"] = train_cfg

    if args.steps is not None and args.epochs is None:
        epochs = 1
        max_batches = int(args.steps)
    elif args.epochs is not None:
        epochs = int(args.epochs)
        max_batches = None
    else:
        epochs = yaml_epochs
        max_batches = None if max_batches_per_epoch is not None else max_train_batches

    m = build_segmentation_model(cfg)
    dev = device
    m.to(dev)
    if use_ddp:
        m = DDP(m, **ddp_kwargs_from_cfg(cfg))
    m.train()
    loss_fn = build_segmentation_loss(cfg).to(dev)
    initial_lr = float(train_cfg.get("lr", 1e-4))
    opt = _build_finetune_optimizer(m.parameters(), train_cfg)
    use_poly_lr = bool(train_cfg.get("poly_lr", train_cfg.get("poly_lr_exponent") is not None))
    poly_exp = float(train_cfg.get("poly_lr_exponent", 0.9))
    L = None
    global_step = 0
    out_dir = Path((cfg or {}).get("output", {}).get("dir", "outputs/mri/seg/last"))
    log_cfg = (cfg or {}).get("logging") or {}
    best_metric = str(log_cfg.get("best_metric", "val_mean_dice") or "val_mean_dice").strip()
    best_mode_raw = str(log_cfg.get("best_mode", "auto") or "auto").strip().lower()
    ckpt_cfg = FinetuneCheckpointConfig.from_cfg(cfg)
    ckpt_manager: FinetuneCheckpointManager | None = None
    if ckpt_cfg is not None:
        if ckpt_cfg.monitor:
            best_metric = {
                "val_dice_mean": "val_mean_dice",
                "val_mean_dice": "val_mean_dice",
            }.get(str(ckpt_cfg.monitor), str(ckpt_cfg.monitor))
        if ckpt_cfg.mode in ("max", "min"):
            best_mode_raw = ckpt_cfg.mode

    merged = get_merged_model_config(cfg) or {}
    n_classes = int((cfg.get("data") or {}).get("num_classes") or merged.get("out_channels", 2))
    amp_on, amp_dtype, amp_use_scaler = _finetune_amp_settings(train_cfg, dev)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_use_scaler)
    if ckpt_cfg is not None and save_ckpt and is_main_process():
        ckpt_manager = FinetuneCheckpointManager(
            out_dir,
            cfg=cfg,
            model=m,
            optimizer=opt,
            scaler=scaler if amp_use_scaler else None,
            ckpt_cfg=ckpt_cfg,
        )
    start_epoch = 0
    resume_path = (args.resume or "").strip()
    if resume_path:
        rp = Path(resume_path)
        if not rp.is_file():
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")
        start_epoch, global_step = load_finetune_resume_checkpoint(
            rp,
            m,
            opt,
            scaler=scaler if amp_use_scaler else None,
            ckpt_manager=ckpt_manager if is_main_process() else None,
        )
        if is_main_process():
            print(
                f"[finetune] resumed from {rp.resolve()} | next_epoch={start_epoch + 1}/{epochs} "
                f"global_step={global_step}",
                flush=True,
            )
    grad_clip_raw = train_cfg.get("grad_clip_norm", train_cfg.get("clip_grad"))
    grad_clip_norm = float(grad_clip_raw) if grad_clip_raw is not None else 0.0
    if grad_clip_norm > 0:
        print(f"[finetune] grad_clip_norm={grad_clip_norm}", flush=True)
    early_stopping = bool(train_cfg.get("early_stopping", False))
    early_stop_patience = int(train_cfg.get("early_stopping_patience", 150) or 150)
    early_stop_min_epochs = int(train_cfg.get("early_stopping_min_epochs", 0) or 0)
    if early_stopping and is_main_process():
        print(
            f"[finetune] early_stopping: patience={early_stop_patience} "
            f"min_epochs={early_stop_min_epochs} monitor={best_metric}",
            flush=True,
        )
    data_cfg = (cfg.get("data") or {})
    report_brats_regions = bool(data_cfg.get("report_brats_regions", True)) and n_classes == 4
    report_per_class_dice = bool(data_cfg.get("report_per_class_dice", n_classes > 4))
    class_names = resolve_segmentation_class_names(data_cfg, n_classes)

    use_val_accum = _val_metric_accumulated(cfg)
    dl = _make_loader(cfg, train=True, use_ddp=use_ddp)
    val_dl = _make_loader(cfg, train=False, use_ddp=False) if is_main_process() else None
    if is_main_process():
        print(
            f"[finetune][ddp] rank={get_rank()}/{get_world_size()} device={dev} ddp={use_ddp}",
            flush=True,
        )
        if use_ddp and patch_sampler_enabled(data_cfg):
            log_ddp_data_plan(
                global_samples_per_epoch=global_train_samples,
                batch_size=int(train_cfg.get("batch_size", 1)),
                grad_accum_steps=1,
                prefix="[finetune][ddp]",
            )
    if val_dl is not None:
        vinfo = "accumulated TP/FP/FN" if use_val_accum else "per-batch Dice"
        ds_val = val_dl.dataset
        extra = ""
        if patch_sampler_enabled(data_cfg):
            n_cases = getattr(ds_val, "n_cases", "?")
            extra = f" | patch_sampler {n_cases} val cases"
        if is_main_process():
            print(
                f"[finetune] validation: {len(val_dl)} batches | metric={vinfo}{extra}",
                flush=True,
            )
            if use_ddp:
                print("[finetune][ddp] validation is rank-0 only", flush=True)
    elif is_main_process():
        print(
            "[finetune] no validation set (set data.index_val or data.csv_val to a CSV path)",
            flush=True,
        )
    if patch_sampler_enabled(data_cfg) and dl is not None and is_main_process():
        ds_tr = dl.dataset
        print(
            f"[finetune] train patch_sampler: {len(dl)} batches/epoch | "
            f"{getattr(ds_tr, 'n_cases', '?')} train cases | "
            f"samples_per_epoch={data_cfg.get('samples_per_epoch', 250)}",
            flush=True,
        )

    try:
        if dl and len(dl) > 0:
            n_batch_loader = len(dl)
            ds_cfg = getattr(dl.dataset, "cfg", {}) or {}
            ik = ds_cfg.get("image_key") or (cfg.get("data") or {}).get("image_key") or "image"
            lk = ds_cfg.get("label_key") or (cfg.get("data") or {}).get("label_key") or "label"

            def _best_maximize(metric: str, mode: str) -> bool:
                if mode == "max":
                    return True
                if mode == "min":
                    return False
                return not (metric == "val_loss" or metric.endswith("_loss"))

            maximize_best = _best_maximize(best_metric, best_mode_raw)
            if is_main_process():
                amp_msg = ""
                if amp_on and amp_dtype is not None:
                    amp_msg = f" | AMP train: {str(amp_dtype).split('.')[-1]}"
                    if amp_use_scaler:
                        amp_msg += " + GradScaler"
                print(
                    f"[finetune] {epochs} epoch(s), {n_batch_loader} batches/epoch"
                    + (f", stop after {max_batches} batches" if max_batches is not None else "")
                    + (f" | checkpoints: last_model.pt every epoch" if save_ckpt else " | checkpoints: disabled")
                    + (
                        f", best_model.pt when {best_metric} improves ({'higher' if maximize_best else 'lower'} is better)"
                        if val_dl is not None and save_ckpt
                        else ""
                    )
                    + amp_msg
                    + "; val in full precision",
                    flush=True,
                )
            best_score: float | None = ckpt_manager.best_score if ckpt_manager is not None else None
            for epoch in range(start_epoch, epochs):
                if use_ddp and getattr(dl, "sampler", None) is not None and hasattr(dl.sampler, "set_epoch"):
                    dl.sampler.set_epoch(epoch)
                if use_ddp:
                    dist.barrier()
                if use_poly_lr:
                    lr_ep = _poly_lr_epoch(epoch, epochs, initial_lr, poly_exp)
                    for pg in opt.param_groups:
                        pg["lr"] = lr_ep
                epoch_loss = 0.0
                n_b = 0
                budget = None
                if max_batches_per_epoch is not None:
                    budget = max_batches_per_epoch
                elif max_batches is not None:
                    budget = max(0, max_batches - global_step)
                if budget == 0:
                    break
                total_this = len(dl) if budget is None else min(len(dl), budget)
                batch_iter = itertools.islice(iter(dl), total_this)
                desc = f"Epoch {epoch + 1}/{epochs}"
                pbar = tqdm(
                    batch_iter,
                    total=total_this,
                    desc=desc,
                    unit="batch",
                    dynamic_ncols=True,
                    leave=True,
                    disable=not is_main_process(),
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                )
                for batch in pbar:
                    batch = unwrap_monai_dict_batch(batch)
                    if not isinstance(batch, dict):
                        raise TypeError(f"Expected dict batch after unwrap, got {type(batch)}")
                    x = batch.get(ik, batch.get("image", batch.get("path")))
                    y = batch.get(lk, batch.get("label", batch.get("label_path")))
                    if x is None or y is None:
                        raise KeyError(
                            f"Batch missing tensors (image_key={ik!r}, label_key={lk!r}); batch keys={list(batch.keys())!r}"
                        )
                    if not isinstance(x, torch.Tensor):
                        x = torch.as_tensor(x, dtype=torch.float32)
                    if not isinstance(y, torch.Tensor):
                        y = torch.as_tensor(y, dtype=torch.long)
                    x, y = x.to(dev), y.to(dev)
                    y = y.long()
                    if y.dim() == 4:
                        y = y.unsqueeze(1)
                    elif y.dim() == 5 and y.shape[1] != 1:
                        raise ValueError(f"Expected label shape [B,D,H,W] or [B,1,D,H,W], got {tuple(y.shape)}")
                    opt.zero_grad(set_to_none=True)
                    train_cm: Any = (
                        torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
                        if amp_on and amp_dtype is not None
                        else nullcontext()
                    )
                    with train_cm:
                        raw_out = m(x)
                        L = compute_segmentation_loss(loss_fn, raw_out, y)
                    loss_val = float(L.detach())
                    if not math.isfinite(loss_val):
                        _train_log(
                            f"[finetune] WARNING: non-finite train loss at step={global_step + 1} "
                            f"(skipping optimizer step; weights unchanged)"
                        )
                        opt.zero_grad(set_to_none=True)
                        continue
                    if amp_use_scaler:
                        scaler.scale(L).backward()
                        if grad_clip_norm > 0:
                            scaler.unscale_(opt)
                            torch.nn.utils.clip_grad_norm_(m.parameters(), grad_clip_norm)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        L.backward()
                        if grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(m.parameters(), grad_clip_norm)
                        opt.step()
                    epoch_loss += loss_val
                    n_b += 1
                    global_step += 1
                    if is_main_process():
                        if max_batches is not None:
                            rem = max_batches - global_step
                            pbar.set_postfix(loss=f"{float(L.detach()):.4f}", remaining_batches=str(rem))
                        else:
                            rem_eps = epochs - epoch - 1
                            rem_batches_this_ep = max(total_this - n_b, 0)
                            pbar.set_postfix(
                                loss=f"{float(L.detach()):.4f}",
                                batches_left_ep=str(rem_batches_this_ep),
                                epochs_after=str(rem_eps),
                            )
                    if max_batches is not None and global_step >= max_batches:
                        break
                if max_batches is not None and global_step >= max_batches:
                    break
                mean_ep = all_reduce_mean(epoch_loss / max(n_b, 1))
                ep_done = epoch + 1
                if is_main_process():
                    _train_log(f"[finetune] epoch {ep_done}/{epochs} mean_loss={mean_ep:.6f} batches={n_b}")
                if val_dl is not None and is_main_process():
                    val_source: Any = (
                        itertools.islice(val_dl, max_val_batches)
                        if max_val_batches is not None
                        else val_dl
                    )
                    vm = _run_validation(
                        m,
                        val_source,
                        dev,
                        ik=ik,
                        lk=lk,
                        loss_fn=loss_fn,
                        n_classes=n_classes,
                        report_brats_regions=report_brats_regions,
                        report_per_class_dice=report_per_class_dice,
                        class_names=class_names,
                        cfg=cfg,
                        use_accumulated=use_val_accum,
                    )
                    _log_validation_metrics(
                        vm,
                        report_brats_regions=report_brats_regions,
                        report_per_class_dice=report_per_class_dice,
                        class_names=class_names,
                        n_classes=n_classes,
                    )
                    if save_ckpt:
                        _append_val_metrics_csv(out_dir, ep_done, vm)
                    score = vm.get(best_metric)
                    if ckpt_manager is not None:
                        prev_best = ckpt_manager.best_score
                        ckpt_manager.on_epoch_end(epoch=ep_done, global_step=global_step, metrics=vm)
                        if ckpt_manager.best_score is not None and ckpt_manager.best_score != prev_best:
                            best_score = ckpt_manager.best_score
                            _train_log(
                                f"[finetune] new best {best_metric}={best_score:.6f} -> {ckpt_cfg.best_filename}"
                            )
                    elif save_ckpt and isinstance(score, (int, float)) and math.isfinite(float(score)):
                        s = float(score)
                        is_better = best_score is None or (
                            s > best_score if maximize_best else s < best_score
                        )
                        if is_better:
                            best_score = s
                            _save_finetune_checkpoint(
                                out_dir,
                                m,
                                opt,
                                epoch_done=ep_done,
                                global_step=global_step,
                                filename="best_model.pt",
                            )
                            _train_log(
                                f"[finetune] new best {best_metric}={best_score:.6f} -> best_model.pt"
                            )
                elif is_main_process() and ckpt_manager is not None:
                    ckpt_manager.on_epoch_end(epoch=ep_done, global_step=global_step, metrics={})
                if save_ckpt and is_main_process() and ckpt_manager is None:
                    _save_finetune_checkpoint(
                        out_dir,
                        m,
                        opt,
                        epoch_done=ep_done,
                        global_step=global_step,
                        filename="last_model.pt",
                    )
                if use_ddp:
                    dist.barrier()
                if max_batches is not None and global_step >= max_batches:
                    break
                should_stop = False
                if early_stopping and val_dl is not None:
                    best_ep = ckpt_manager.best_epoch if ckpt_manager is not None else None
                    if is_main_process() and best_ep is not None and ep_done >= early_stop_min_epochs:
                        stale_epochs = ep_done - best_ep
                        if stale_epochs >= early_stop_patience:
                            should_stop = True
                            _train_log(
                                f"[finetune] early stopping at epoch {ep_done}/{epochs}: "
                                f"no {best_metric} improvement for {stale_epochs} epochs "
                                f"(patience={early_stop_patience}, best epoch={best_ep})"
                            )
                    if use_ddp:
                        stop_flag = torch.tensor(int(should_stop), device=dev)
                        dist.broadcast(stop_flag, 0)
                        should_stop = bool(stop_flag.item())
                    if should_stop:
                        break

        if L is None:
            seg = get_merged_model_config(cfg) or {}
            ch = int(seg.get("in_channels", 1))
            nc = int(seg.get("out_channels", 2))
            x = torch.randn(1, ch, 96, 96, 96, device=dev)
            y = (torch.rand(1, 96, 96, 96, device=dev) * max(1, nc - 1)).long()
            opt.zero_grad()
            out = m(x)
            L = F.cross_entropy(out, y)
            L.backward()
            opt.step()
            global_step = 1
            if save_ckpt and is_main_process():
                _save_finetune_checkpoint(
                    out_dir,
                    m,
                    opt,
                    epoch_done=1,
                    global_step=global_step,
                    filename="last_model.pt",
                )

        if is_main_process():
            print("ok loss", float(L.detach() if L is not None else 0.0), "->", out_dir)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
