#!/usr/bin/env python3
"""
UNETR++ inpainting + masked EMA teacher-feature reconstruction.

  L_total = L_recon + lambda_feature * L_feature

Separate from global-DINO, patch-DINO, and existing inpainting+DINO trainers.

Example:
  python scripts/pretrain_unetrpp_inpainting_feature_reconstruction.py \\
    --config configs/pretrain/ct/volume3d/unetrpp_inpainting_feature_reconstruction_synapse_npz.yaml
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import _bootstrap  # noqa: F401


def _maybe_tqdm(enable: bool):
    try:
        from tqdm.auto import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    def wrap(iterable, **kw):
        if enable and _tqdm is not None:
            return _tqdm(iterable, **kw)
        return iterable

    return wrap


class _DeterministicPatchDataset:
    """Wrap patch dataset so each index uses a fixed RNG seed (debug fixed subset)."""

    def __init__(self, base, seed: int = 42) -> None:
        self.base = base
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import numpy as np

        random.seed(self.seed + int(idx))
        np.random.seed((self.seed + int(idx)) % (2**32 - 1))
        return self.base[idx]


def _train_step(
    model,
    batch: dict,
    device,
    use_amp: bool,
    spatial: Tuple[int, int, int],
    inpaint_cfg: dict,
    feat_cfg: dict,
    *,
    lambda_now: float,
) -> dict:
    import torch
    import torch.nn.functional as F

    from dinomim_pytorch.unetrpp_dino_inpainting import inpainting_recon_loss
    from dinomim_pytorch.unetrpp_feature_reconstruction import (
        masked_cosine_feature_loss,
        multiscale_feature_loss,
        random_block_mask_and_indicator,
        resolve_feature_recon_version,
        resolve_stage_weights,
        is_multiscale_feature_recon,
        token_std_raw,
        voxel_mask_to_token_mask,
    )

    d, h, w = spatial

    def _to_model(t: torch.Tensor) -> torch.Tensor:
        t = t.to(device, non_blocking=True)
        if tuple(t.shape[-3:]) != (d, h, w):
            return F.interpolate(t, size=(d, h, w), mode="trilinear", align_corners=False)
        return t

    x_clean = batch.get("volume")
    if x_clean is None:
        x_clean = batch["teacher_glob"][0]
    x_clean = _to_model(x_clean)

    mask_ratio = float(inpaint_cfg.get("mask_ratio", 0.75))
    patch_size = int(inpaint_cfg.get("patch_size", 16))
    mask_value = float(inpaint_cfg.get("mask_value", 0.0))
    only_masked = bool(inpaint_cfg.get("reconstruct_only_masked", True))
    masked_tokens_only = bool(feat_cfg.get("masked_tokens_only", True))
    token_mask_mode = str(feat_cfg.get("token_mask_mode", "any"))
    token_mask_threshold = float(feat_cfg.get("token_mask_threshold", 0.75))
    version = resolve_feature_recon_version(feat_cfg)
    stage_weights = resolve_stage_weights(feat_cfg, model.stages)

    x_masked, voxel_mask = random_block_mask_and_indicator(
        x_clean,
        mask_ratio=mask_ratio,
        patch_size=patch_size,
        mask_value=mask_value,
    )

    predictors = (
        model.feature_predictors.module
        if hasattr(model.feature_predictors, "module")
        else model.feature_predictors
    )

    use_cuda_amp = use_amp and device.type == "cuda"
    with torch.amp.autocast("cuda" if use_cuda_amp else "cpu", enabled=use_cuda_amp):
        with torch.no_grad():
            teacher_tokens, teacher_grids = model.encode_teacher_tokens(x_clean)

        recon, student_tokens, student_grids = model.forward_student(x_masked)

        l_recon = inpainting_recon_loss(
            recon,
            x_clean,
            x_masked,
            mask_value=mask_value,
            only_masked=only_masked,
        )

        if is_multiscale_feature_recon(version) or len(model.stages) > 1:
            student_preds = {}
            token_masks = {}
            for stage_key in model.stage_keys:
                if student_grids[stage_key] != teacher_grids[stage_key]:
                    raise ValueError(
                        f"Teacher/student token grids differ at {stage_key}: "
                        f"{teacher_grids[stage_key]} vs {student_grids[stage_key]}"
                    )
                student_preds[stage_key] = predictors.forward_stage(stage_key, student_tokens[stage_key])
                if masked_tokens_only:
                    token_masks[stage_key] = voxel_mask_to_token_mask(
                        voxel_mask,
                        student_grids[stage_key],
                        mode=token_mask_mode,
                        threshold=token_mask_threshold,
                    )
                else:
                    token_masks[stage_key] = None
            l_feat, feat_meta = multiscale_feature_loss(
                student_preds,
                teacher_tokens,
                token_masks=token_masks,
                stage_weights=stage_weights,
                feat_cfg=feat_cfg,
            )
            out: Dict[str, Any] = {
                "loss": l_recon + float(lambda_now) * l_feat,
                "mean_recon": float(l_recon.detach()),
                "mean_feature": float(l_feat.detach()),
                "feature_w": float(lambda_now) * float(l_feat.detach()),
                "lambda_feature_now": float(lambda_now),
            }
            for stage_key in model.stage_keys:
                out[f"s_std_{stage_key}"] = token_std_raw(student_tokens[stage_key])
                out[f"t_std_{stage_key}"] = token_std_raw(teacher_tokens[stage_key])
            out.update(feat_meta)
            out["mean_cosine_similarity"] = float(
                sum(float(feat_meta.get(f"cos_{k}", 0.0)) for k in model.stage_keys) / max(1, len(model.stage_keys))
            )
            out["masked_token_fraction"] = float(
                sum(float(feat_meta.get(f"mask_frac_{k}", 0.0)) for k in model.stage_keys) / max(1, len(model.stage_keys))
            )
            out["student_token_std_raw"] = float(out.get("s_std_stage4", out.get(f"s_std_{model.stage_keys[-1]}", 0.0)))
            out["teacher_token_std_raw"] = float(out.get("t_std_stage4", out.get(f"t_std_{model.stage_keys[-1]}", 0.0)))
            out["student_pred_std"] = float(
                sum(float(feat_meta.get(f"pred_std_{k}", 0.0)) for k in model.stage_keys) / max(1, len(model.stage_keys))
            )
            out["teacher_target_std"] = float(
                sum(float(feat_meta.get(f"target_std_{k}", 0.0)) for k in model.stage_keys) / max(1, len(model.stage_keys))
            )
            return out

        stage_key = model.stage_keys[-1]
        if student_grids[stage_key] != teacher_grids[stage_key]:
            raise ValueError(
                f"Teacher/student token grids differ: {teacher_grids[stage_key]} vs {student_grids[stage_key]}"
            )
        student_pred = predictors.forward_stage(stage_key, student_tokens[stage_key])
        token_mask = None
        if masked_tokens_only:
            token_mask = voxel_mask_to_token_mask(
                voxel_mask,
                student_grids[stage_key],
                mode=token_mask_mode,
                threshold=token_mask_threshold,
            )

        l_feat, feat_meta = masked_cosine_feature_loss(
            student_pred,
            teacher_tokens[stage_key],
            token_mask=token_mask,
        )
        loss = l_recon + float(lambda_now) * l_feat

    return {
        "loss": loss,
        "mean_recon": float(l_recon.detach()),
        "mean_feature": float(l_feat.detach()),
        "feature_w": float(lambda_now) * float(l_feat.detach()),
        "lambda_feature_now": float(lambda_now),
        "student_token_std_raw": token_std_raw(student_tokens[stage_key]),
        "teacher_token_std_raw": token_std_raw(teacher_tokens[stage_key]),
        "student_pred_std": feat_meta.get("student_pred_std", 0.0),
        "teacher_target_std": feat_meta.get("teacher_target_std", 0.0),
        "mean_cosine_similarity": feat_meta.get("mean_cosine_similarity", 0.0),
        "masked_token_fraction": feat_meta.get("masked_token_fraction", 0.0),
        f"feature_{stage_key}": float(l_feat.detach()),
        f"cos_{stage_key}": feat_meta.get("mean_cosine_similarity", 0.0),
        f"mask_frac_{stage_key}": feat_meta.get("masked_token_fraction", 0.0),
        f"s_std_{stage_key}": token_std_raw(student_tokens[stage_key]),
        f"t_std_{stage_key}": token_std_raw(teacher_tokens[stage_key]),
        f"pred_std_{stage_key}": feat_meta.get("student_pred_std", 0.0),
        f"target_std_{stage_key}": feat_meta.get("teacher_target_std", 0.0),
    }


def main() -> None:
    import torch
    import torch.distributed as dist
    from torch.cuda.amp import GradScaler
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader

    from dinomim_pytorch.config_utils import load_yaml
    from dinomim_pytorch.distributed import (
        all_reduce_mean,
        apply_patch_sampler_ddp_data_cfg,
        cleanup_distributed,
        ddp_kwargs_from_cfg,
        get_rank,
        get_world_size,
        is_main_process,
        log_ddp_data_plan,
        patch_samples_per_rank,
        seed_worker,
        setup_distributed,
    )
    from dinomim_pytorch.datasets.nnformer_npz_patch import patch_sampler_enabled
    from dinomim_pytorch.ssl_volume_dataset import (
        build_ssl_volume_dataset,
        has_ssl_volume_data,
        model_spatial_tuple_from_cfg,
        volume_multiview_collate_fn,
    )
    from dinomim_pytorch.training_schedules import cosine_schedule
    from dinomim_pytorch.inpainting_feature_reconstruction_factory import (
        build_inpainting_feature_reconstruction,
        ssl_pretrain_scheme,
    )
    from dinomim_pytorch.unetrpp_feature_reconstruction import (
        feature_collapse_warning,
        format_feature_recon_config_log,
        lambda_feature_for_epoch,
        predictor_grad_norm,
        predictor_grad_norm_per_stage,
        resolve_feature_recon_version,
        save_feature_recon_checkpoint,
    )

    ap = argparse.ArgumentParser(description="UNETR++ inpainting + feature reconstruction SSL")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--dry-batch", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(str(Path(args.config).expanduser().resolve())) or {}
    device, use_ddp = setup_distributed(cfg)

    debug = dict(cfg.get("debug") or {})
    debug_enabled = bool(debug.get("enabled", False))

    train = dict(cfg.get("training") or {})
    if debug_enabled:
        train["epochs"] = int(debug.get("epochs", 3))
    if args.epochs is not None:
        train["epochs"] = max(1, int(args.epochs))
    cfg["training"] = train

    data = dict(cfg.get("data") or {})
    global_samples_per_epoch = int(data.get("samples_per_epoch", data.get("ssl_samples_per_epoch", 500)))

    if debug_enabled and debug.get("fixed_subset", True):
        bs = int(train.get("batch_size", 2))
        max_b = int(debug.get("max_batches_per_epoch", 50))
        global_samples_per_epoch = max_b * bs * max(1, get_world_size() if use_ddp else 1)
        data["samples_per_epoch"] = global_samples_per_epoch
        data["seed"] = int(data.get("seed", cfg.get("experiment", {}).get("seed", 42)))
        cfg["data"] = data

    if use_ddp and patch_sampler_enabled(data):
        cfg["data"] = apply_patch_sampler_ddp_data_cfg(data, rank=get_rank(), world_size=get_world_size())
        data = dict(cfg["data"])

    feat_cfg = cfg.get("feature_reconstruction") or {}
    inpaint_cfg = cfg.get("inpainting") or {}
    log = cfg.get("logging") or {}
    exp = cfg.get("experiment") or {}

    out_dir = Path(exp.get("output_dir", "outputs/pretrain/ct/volume3d/unetrpp_inpainting_feature_reconstruction"))
    if debug_enabled:
        out_dir = out_dir / "debug"
    if is_main_process():
        out_dir.mkdir(parents=True, exist_ok=True)
    if use_ddp:
        dist.barrier()
    metrics_csv = out_dir / "metrics.csv"
    log_f = out_dir / "train.log"

    if is_main_process():
        print(format_feature_recon_config_log(feat_cfg), file=sys.stderr, flush=True)
        print(
            f"[feature-recon][ddp] rank={get_rank()}/{get_world_size()} "
            f"local_rank={device.index if device.type == 'cuda' else -1} device={device}",
            file=sys.stderr,
            flush=True,
        )
        log_ddp_data_plan(
            global_samples_per_epoch=global_samples_per_epoch,
            batch_size=int(train.get("batch_size", 2)),
            grad_accum_steps=int(train.get("grad_accum_steps", 1)),
            prefix="[feature-recon][ddp]",
        )

    spatial = model_spatial_tuple_from_cfg(cfg)

    model = build_inpainting_feature_reconstruction(cfg).to(device)
    pretrain_scheme = ssl_pretrain_scheme(cfg)
    if use_ddp:
        ddp_kw = ddp_kwargs_from_cfg(cfg)
        model.student_net = DDP(model.student_net, **ddp_kw)
        model.feature_predictors = DDP(model.feature_predictors, **ddp_kw)

    predictor_mod = (
        model.feature_predictors.module
        if hasattr(model.feature_predictors, "module")
        else model.feature_predictors
    )
    params = list(model.student_net.parameters()) + list(predictor_mod.parameters())
    opt = torch.optim.AdamW(
        params,
        lr=float(train.get("lr", 1e-4)),
        weight_decay=float(train.get("weight_decay", 0.04)),
    )
    use_amp = bool(train.get("mixed_precision", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    if not has_ssl_volume_data(cfg):
        cleanup_distributed()
        raise SystemExit("No SSL volume data configured")

    ds = build_ssl_volume_dataset(cfg)
    if debug_enabled and debug.get("fixed_subset", True):
        ds = _DeterministicPatchDataset(ds, seed=int(cfg.get("data", {}).get("seed", 42)))

    base_seed = int(cfg.get("data", {}).get("seed", cfg.get("experiment", {}).get("seed", 42)))
    loader = DataLoader(
        ds,
        batch_size=int(train.get("batch_size", 2)),
        shuffle=not debug_enabled and not use_ddp,
        num_workers=int(train.get("num_workers", 4 if not debug_enabled else 0)),
        collate_fn=volume_multiview_collate_fn,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=lambda wid: seed_worker(wid, base_seed),
    )

    max_batches = int(debug.get("max_batches_per_epoch", 50)) if debug_enabled else None
    n_batches = max(1, min(len(loader), max_batches) if max_batches else len(loader))
    epochs = int(train.get("epochs", 200))
    total_steps = max(1, n_batches * epochs)
    grad_accum = max(1, int(train.get("grad_accum_steps", 1)))
    max_grad = float(train.get("clip_grad", 3.0))
    t_base = float(feat_cfg.get("teacher_momentum_base", 0.996))
    t_end = float(feat_cfg.get("teacher_momentum_final", 1.0))
    lr0 = float(train.get("lr", 1e-4))
    lr_e = float(train.get("min_lr", 1e-6))
    wd_s = float(train.get("weight_decay", 0.04))
    wd_e = float(train.get("weight_decay_end", wd_s))
    save_ckpt = (not debug_enabled or bool(debug.get("save_checkpoints", False))) and is_main_process()

    feat_version = resolve_feature_recon_version(feat_cfg)
    metrics_header = [
        "step", "epoch", "mean_loss", "mean_recon", "mean_feature", "feature_w",
        "lambda_feature_now", "student_token_std_raw", "teacher_token_std_raw",
        "student_pred_std", "teacher_target_std", "mean_cosine_similarity",
        "masked_token_fraction", "teacher_momentum", "predictor_grad_norm", "learning_rate",
        "epoch_seconds",
        "feature_stage2", "feature_stage3", "feature_stage4",
        "cos_stage2", "cos_stage3", "cos_stage4",
        "s_std_stage2", "s_std_stage3", "s_std_stage4",
        "t_std_stage2", "t_std_stage3", "t_std_stage4",
        "pred_std_stage2", "pred_std_stage3", "pred_std_stage4",
        "target_std_stage2", "target_std_stage3", "target_std_stage4",
        "mask_frac_stage2", "mask_frac_stage3", "mask_frac_stage4",
        "hard_frac_stage2", "hard_frac_stage3", "hard_frac_stage4",
        "hard_tokens_stage2", "hard_tokens_stage3", "hard_tokens_stage4",
        "smooth_l1_stage2", "smooth_l1_stage3", "smooth_l1_stage4",
        "pred_grad_stage2", "pred_grad_stage3", "pred_grad_stage4",
    ]
    first_row = not metrics_csv.is_file() or metrics_csv.stat().st_size == 0
    best_ep_loss = None
    global_step = 0

    if is_main_process():
        spr = patch_samples_per_rank(global_samples_per_epoch, get_world_size()) if use_ddp else int(
            cfg.get("data", {}).get("samples_per_epoch", global_samples_per_epoch)
        )
        print(
            f"[feature-recon] debug={debug_enabled} batches/epoch={n_batches} epochs={epochs} "
            f"save_ckpt={save_ckpt} spatial={spatial} ddp={use_ddp} "
            f"global_samples_per_epoch={global_samples_per_epoch} samples_per_rank={spr}",
            file=sys.stderr,
            flush=True,
        )

    if args.dry_batch:
        batch = next(iter(loader))
        _, lam = lambda_feature_for_epoch(feat_cfg, 0)
        st = _train_step(model, batch, device, use_amp, spatial, inpaint_cfg, feat_cfg, lambda_now=lam)
        if is_main_process():
            print(
                f"dry-batch loss={float(st['loss'].detach()):.6f} recon={st['mean_recon']:.6f} "
                f"feature={st['mean_feature']:.6f} cos={st['mean_cosine_similarity']:.4f} "
                f"s_std={st['student_token_std_raw']:.4f} t_std={st['teacher_token_std_raw']:.4f}",
                file=sys.stderr,
            )
        cleanup_distributed()
        return

    show_prog = bool(log.get("show_progress", True)) and is_main_process()
    tqdm_wrap = _maybe_tqdm(show_prog)

    try:
        for ep in range(epochs):
            t_ep0 = time.perf_counter()
            model.student_net.train()
            model.feature_predictors.train()
            _, lambda_now = lambda_feature_for_epoch(feat_cfg, ep)

            ep_loss = ep_recon = ep_feat = ep_cos = 0.0
            ep_s_std = ep_t_std = ep_sp_std = ep_tt_std = 0.0
            ep_mask_frac = ep_pred_grad = 0.0
            ep_batches = 0
            accum_idx = 0
            pred_grad_last = 0.0
            pred_grad_stage_last: Dict[str, float] = {}
            lr_last = lr0
            ep_stage_acc: Dict[str, float] = {}

            for bi, batch in enumerate(tqdm_wrap(loader, total=n_batches, desc=f"epoch {ep+1}/{epochs}", leave=False)):
                if max_batches and bi >= max_batches:
                    break
                if accum_idx == 0:
                    opt.zero_grad(set_to_none=True)
                st = _train_step(
                    model, batch, device, use_amp, spatial, inpaint_cfg, feat_cfg, lambda_now=lambda_now,
                )
                loss = st["loss"] / grad_accum
                scaler.scale(loss).backward()
                accum_idx += 1
                lf = float(st["loss"].detach())
                ep_loss += lf
                ep_recon += float(st["mean_recon"])
                ep_feat += float(st["mean_feature"])
                ep_cos += float(st["mean_cosine_similarity"])
                ep_s_std += float(st["student_token_std_raw"])
                ep_t_std += float(st["teacher_token_std_raw"])
                ep_sp_std += float(st["student_pred_std"])
                ep_tt_std += float(st["teacher_target_std"])
                ep_mask_frac += float(st["masked_token_fraction"])
                ep_batches += 1
                for k, v in st.items():
                    if k.startswith((
                        "feature_stage", "cos_stage", "mask_frac_stage",
                        "hard_frac_stage", "hard_tokens_stage", "smooth_l1_stage",
                        "s_std_stage", "t_std_stage", "pred_std_stage", "target_std_stage",
                    )):
                        ep_stage_acc[k] = ep_stage_acc.get(k, 0.0) + float(v)

                if accum_idx < grad_accum:
                    continue
                scaler.unscale_(opt)
                pred_grad_stage_last = predictor_grad_norm_per_stage(predictor_mod)
                pred_grad_last = predictor_grad_norm(predictor_mod)
                ep_pred_grad += pred_grad_last
                torch.nn.utils.clip_grad_norm_(params, max_grad)
                scaler.step(opt)
                scaler.update()
                accum_idx = 0

                m_ema = cosine_schedule(global_step, total_steps, t_base, t_end)
                model.update_teacher_ema(m_ema)
                lr_last = cosine_schedule(global_step, total_steps, lr0, lr_e)
                wd = cosine_schedule(global_step, total_steps, wd_s, wd_e)
                for g in opt.param_groups:
                    g["lr"] = lr_last
                    g["weight_decay"] = wd

                if is_main_process() and global_step % int(log.get("log_every", 10) or 10) == 0:
                    row = {
                        "step": global_step,
                        "epoch": ep,
                        "mean_loss": lf,
                        "mean_recon": st["mean_recon"],
                        "mean_feature": st["mean_feature"],
                        "feature_w": st["feature_w"],
                        "lambda_feature_now": lambda_now,
                        "student_token_std_raw": st["student_token_std_raw"],
                        "teacher_token_std_raw": st["teacher_token_std_raw"],
                        "student_pred_std": st["student_pred_std"],
                        "teacher_target_std": st["teacher_target_std"],
                        "mean_cosine_similarity": st["mean_cosine_similarity"],
                        "masked_token_fraction": st["masked_token_fraction"],
                        "teacher_momentum": m_ema,
                        "predictor_grad_norm": pred_grad_last,
                        "learning_rate": lr_last,
                        "epoch_seconds": 0.0,
                    }
                    for key in metrics_header:
                        if key.startswith((
                            "feature_stage", "cos_stage", "mask_frac_stage",
                            "hard_frac_stage", "hard_tokens_stage", "smooth_l1_stage",
                            "s_std_stage", "t_std_stage", "pred_std_stage", "target_std_stage",
                        )):
                            row[key] = float(st.get(key, 0.0))
                    for key, val in pred_grad_stage_last.items():
                        row[key] = val
                    with open(metrics_csv, "a", newline="", encoding="utf-8") as fh:
                        w = csv.DictWriter(fh, fieldnames=metrics_header)
                        if first_row:
                            w.writeheader()
                            first_row = False
                        w.writerow(row)

                global_step += 1

            ep_sec = time.perf_counter() - t_ep0
            n_ep = max(1, ep_batches)
            ep_mean = all_reduce_mean(ep_loss / n_ep)
            ep_recon_m = all_reduce_mean(ep_recon / n_ep)
            ep_feat_m = all_reduce_mean(ep_feat / n_ep)
            ep_cos_m = all_reduce_mean(ep_cos / n_ep)
            ep_s_std_m = all_reduce_mean(ep_s_std / n_ep)
            ep_t_std_m = all_reduce_mean(ep_t_std / n_ep)
            ep_sp_std_m = all_reduce_mean(ep_sp_std / n_ep)
            ep_tt_std_m = all_reduce_mean(ep_tt_std / n_ep)
            ep_mask_m = all_reduce_mean(ep_mask_frac / n_ep)
            ep_pred_grad_m = all_reduce_mean(ep_pred_grad / n_ep)

            if is_main_process():
                summary = {
                    "student_token_std_raw": ep_s_std_m,
                    "teacher_token_std_raw": ep_t_std_m,
                }
                for key, val in ep_stage_acc.items():
                    summary[key] = val / n_ep
                feature_collapse_warning(ep, summary)

            if save_ckpt:
                save_feature_recon_checkpoint(
                    out_dir / "last.pt",
                    model=model,
                    scheme=pretrain_scheme,
                    epoch=ep,
                    best_loss=best_ep_loss,
                    cfg=cfg,
                    optimizer=opt,
                    global_step=global_step,
                )
                if best_ep_loss is None or ep_mean < best_ep_loss:
                    best_ep_loss = ep_mean
                    save_feature_recon_checkpoint(
                        out_dir / "best.pt",
                        model=model,
                        scheme=pretrain_scheme,
                        epoch=ep,
                        best_loss=best_ep_loss,
                        cfg=cfg,
                        optimizer=opt,
                        global_step=global_step,
                    )

            if is_main_process():
                stage_bits = " ".join(
                    f"{k.split('_', 1)[-1]}={ep_stage_acc.get(k, 0.0)/n_ep:.4f}"
                    for k in sorted(ep_stage_acc)
                    if k.startswith("cos_stage")
                )
                line = (
                    f"epoch={ep+1} mean_loss={ep_mean:.6f} mean_recon={ep_recon_m:.6f} "
                    f"mean_feature={ep_feat_m:.6f} feature_w={lambda_now*ep_feat_m:.6f} "
                    f"lambda_now={lambda_now:.4f} s_std={ep_s_std_m:.4f} t_std={ep_t_std_m:.4f} "
                    f"pred_std={ep_sp_std_m:.4f} target_std={ep_tt_std_m:.4f} "
                    f"cos={ep_cos_m:.4f} mask_frac={ep_mask_m:.4f} "
                    f"pred_grad={ep_pred_grad_m:.4f} version={feat_version} epoch_sec={ep_sec:.1f}"
                )
                if stage_bits:
                    line += f" | {stage_bits}"
                with open(log_f, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                print(f"[feature-recon] {line}", file=sys.stderr, flush=True)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
