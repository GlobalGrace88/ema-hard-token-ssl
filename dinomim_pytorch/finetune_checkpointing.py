"""Periodic and top-k checkpoint management for downstream segmentation finetune."""
from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn

from dinomim_pytorch.distributed import unwrap_module


def _resolve_monitor_key(monitor: str, metrics: Mapping[str, Any]) -> Optional[float]:
    """Map ``monitor`` config to a scalar in validation metrics."""
    key = str(monitor).strip()
    aliases = {
        "val_dice_mean": "val_mean_dice",
        "val_mean_dice": "val_mean_dice",
        "val_loss": "val_loss",
    }
    resolved = aliases.get(key, key)
    val = metrics.get(resolved)
    if isinstance(val, (int, float)) and math.isfinite(float(val)):
        return float(val)
    return None


@dataclass
class FinetuneCheckpointConfig:
    save_last: bool = True
    save_best: bool = True
    save_every_epochs: int = 0
    save_top_k: int = 0
    monitor: str = "val_dice_mean"
    mode: str = "max"
    keep_epoch_checkpoints: bool = True
    epoch_checkpoint_pattern: str = "epoch_{epoch:03d}.pt"
    checkpoint_index_file: str = "checkpoint_index.json"
    topk_epochs_dir: str = "topk_epochs"
    topk_epochs_pattern: str = "epoch_{epoch:04d}.pt"
    best_filename: str = "best_model.pt"
    last_filename: str = "last_model.pt"

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any]) -> Optional["FinetuneCheckpointConfig"]:
        raw = (cfg or {}).get("checkpointing")
        if not raw:
            return None
        return cls(
            save_last=bool(raw.get("save_last", True)),
            save_best=bool(raw.get("save_best", True)),
            save_every_epochs=int(raw.get("save_every_epochs", 0) or 0),
            save_top_k=int(raw.get("save_top_k", 0) or 0),
            monitor=str(raw.get("monitor", "val_dice_mean")),
            mode=str(raw.get("mode", "max")).strip().lower(),
            keep_epoch_checkpoints=bool(raw.get("keep_epoch_checkpoints", True)),
            epoch_checkpoint_pattern=str(raw.get("epoch_checkpoint_pattern", "epoch_{epoch:03d}.pt")),
            checkpoint_index_file=str(raw.get("checkpoint_index_file", "checkpoint_index.json")),
            topk_epochs_dir=str(raw.get("topk_epochs_dir", "topk_epochs")),
            topk_epochs_pattern=str(raw.get("topk_epochs_pattern", "epoch_{epoch:04d}.pt")),
        )


@dataclass
class _TopKEntry:
    epoch: int
    path: str
    score: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    is_best: bool = False


class FinetuneCheckpointManager:
    """Save last/best, periodic epoch checkpoints, and top-k by validation metric."""

    def __init__(
        self,
        out_dir: Path,
        *,
        cfg: Mapping[str, Any],
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
        scaler: Any = None,
        ckpt_cfg: Optional[FinetuneCheckpointConfig] = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.cfg = dict(cfg)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.ckpt_cfg = ckpt_cfg or FinetuneCheckpointConfig()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._topk: List[_TopKEntry] = []
        self._best_score: Optional[float] = None
        self._best_epoch: Optional[int] = None
        self._index_path = self.out_dir / self.ckpt_cfg.checkpoint_index_file
        self._load_index()

    def _load_index(self) -> None:
        if not self._index_path.is_file():
            return
        try:
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for item in data.get("checkpoints", []):
            try:
                self._topk.append(
                    _TopKEntry(
                        epoch=int(item["epoch"]),
                        path=str(item["path"]),
                        score=float(item.get(self.ckpt_cfg.monitor, item.get("val_dice_mean", item.get("score", 0.0)))),
                        metrics={k: v for k, v in item.items() if k not in ("epoch", "path", "is_best")},
                        is_best=bool(item.get("is_best", False)),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        self._topk.sort(key=lambda e: e.score, reverse=(self.ckpt_cfg.mode == "max"))

    def _build_payload(
        self,
        *,
        epoch: int,
        global_step: int,
        metrics: Optional[Mapping[str, Any]] = None,
        best_metric: Optional[float] = None,
    ) -> Dict[str, Any]:
        core = unwrap_module(self.model)
        payload: Dict[str, Any] = {
            "model": core.state_dict(),
            "model_state_dict": core.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "best_metric": best_metric,
            "current_metrics": dict(metrics or {}),
            "config": self.cfg,
        }
        if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
            payload["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.scaler is not None and hasattr(self.scaler, "state_dict"):
            payload["scaler_state_dict"] = self.scaler.state_dict()
        return payload

    def _save_file(self, filename: str, payload: Dict[str, Any]) -> Path:
        path = self.out_dir / filename
        try:
            from dinomim_pytorch.checkpoint_metadata import attach_to_payload, build_metadata, save_sidecar

            exp = dict((self.cfg or {}).get("experiment") or {})
            data = dict((self.cfg or {}).get("data") or {})
            meta = build_metadata(
                self.cfg,
                phase="finetune",
                task_name=str(exp.get("task_name") or exp.get("dataset") or "synapse"),
                model_name=str(exp.get("model") or (self.cfg.get("model") or {}).get("architecture") or "unetrpp"),
                fold=int(data.get("fold", exp.get("fold", 0))),
                method=str(exp.get("init_type") or "scratch"),
            )
            payload = attach_to_payload(payload, meta)
            torch.save(payload, str(path))
            save_sidecar(path, meta)
        except Exception:
            torch.save(payload, str(path))
        return path

    def _is_better(self, score: float) -> bool:
        if self._best_score is None:
            return True
        if self.ckpt_cfg.mode == "min":
            return score < self._best_score
        return score > self._best_score

    def _topk_archive_path(self, epoch: int) -> Path:
        name = self.ckpt_cfg.topk_epochs_pattern.format(epoch=int(epoch))
        return self.out_dir / self.ckpt_cfg.topk_epochs_dir / name

    def _archive_topk_payload(self, epoch: int, payload: Dict[str, Any]) -> Path:
        path = self._topk_archive_path(epoch)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(path))
        return path

    def _resolve_topk_source(self, entry: _TopKEntry) -> Optional[Path]:
        archive = self._topk_archive_path(entry.epoch)
        if archive.is_file():
            return archive
        candidate = self.out_dir / entry.path
        if candidate.is_file():
            return candidate
        return None

    def _write_index(self) -> None:
        entries: List[Dict[str, Any]] = []
        for e in sorted(self._topk, key=lambda x: x.epoch):
            row: Dict[str, Any] = {
                "epoch": e.epoch,
                "path": e.path,
                "is_best": e.is_best,
            }
            row.update(e.metrics)
            entries.append(row)
        self._index_path.write_text(
            json.dumps({"checkpoints": entries}, indent=2),
            encoding="utf-8",
        )

    def _update_topk(self, epoch: int, score: float, payload: Dict[str, Any], metrics: Mapping[str, Any]) -> None:
        k = int(self.ckpt_cfg.save_top_k)
        if k <= 0:
            return

        metric_row: Dict[str, Any] = {
            self.ckpt_cfg.monitor: score,
            "val_dice_mean": metrics.get("val_mean_dice", metrics.get("val_dice_mean")),
            "val_hd95_mean": metrics.get("val_hd95_mean"),
            "val_loss": metrics.get("val_loss"),
        }
        metric_row = {key: val for key, val in metric_row.items() if val is not None}

        self._archive_topk_payload(epoch, payload)
        self._topk.append(
            _TopKEntry(
                epoch=epoch,
                path=str(self._topk_archive_path(epoch).relative_to(self.out_dir)),
                score=score,
                metrics=metric_row,
                is_best=(self._best_epoch == epoch),
            )
        )
        reverse = self.ckpt_cfg.mode == "max"
        self._topk.sort(key=lambda e: e.score, reverse=reverse)

        keep = self._topk[:k]
        drop = self._topk[k:]
        keep_epochs = {e.epoch for e in keep}
        for e in drop:
            (self.out_dir / e.path).unlink(missing_ok=True)
            self._topk_archive_path(e.epoch).unlink(missing_ok=True)
        self._topk = keep

        # Two-phase rename: never unlink a destination that another kept entry still points to.
        swap_paths: List[Path] = []
        for idx, e in enumerate(self._topk):
            src = self._resolve_topk_source(e)
            swap = self.out_dir / f"_topk_swap_{idx}.pt"
            if swap.is_file():
                swap.unlink(missing_ok=True)
            if src is None or not src.is_file():
                swap_paths.append(swap)
                continue
            shutil.copy2(src, swap)
            swap_paths.append(swap)

        for rank in range(1, k + 1):
            (self.out_dir / f"topk_{rank}.pt").unlink(missing_ok=True)
        for rank, (e, swap) in enumerate(zip(self._topk, swap_paths), start=1):
            target = f"topk_{rank}.pt"
            if swap.is_file():
                swap.rename(self.out_dir / target)
            e.path = target
            e.is_best = self._best_epoch is not None and e.epoch == self._best_epoch

        for swap in swap_paths:
            if swap.is_file():
                swap.unlink(missing_ok=True)

        self._write_index()

    def on_epoch_end(
        self,
        *,
        epoch: int,
        global_step: int,
        metrics: Optional[Mapping[str, Any]] = None,
    ) -> Optional[float]:
        """Save checkpoints for completed ``epoch``. Returns monitor score if finite."""
        metrics = metrics or {}
        score = _resolve_monitor_key(self.ckpt_cfg.monitor, metrics)
        payload = self._build_payload(
            epoch=epoch,
            global_step=global_step,
            metrics=metrics,
            best_metric=self._best_score,
        )

        if self.ckpt_cfg.save_last:
            self._save_file(self.ckpt_cfg.last_filename, payload)

        every = int(self.ckpt_cfg.save_every_epochs)
        if every > 0 and epoch % every == 0:
            epoch_name = self.ckpt_cfg.epoch_checkpoint_pattern.format(epoch=epoch)
            self._save_file(epoch_name, payload)
            if not self.ckpt_cfg.keep_epoch_checkpoints:
                prev = epoch - every
                if prev > 0:
                    prev_name = self.ckpt_cfg.epoch_checkpoint_pattern.format(epoch=prev)
                    (self.out_dir / prev_name).unlink(missing_ok=True)

        if score is not None and self.ckpt_cfg.save_best and self._is_better(score):
            self._best_score = score
            self._best_epoch = epoch
            payload_best = self._build_payload(
                epoch=epoch,
                global_step=global_step,
                metrics=metrics,
                best_metric=self._best_score,
            )
            self._save_file(self.ckpt_cfg.best_filename, payload_best)

        if score is not None:
            self._update_topk(epoch, score, payload, metrics)

        return score

    @property
    def best_score(self) -> Optional[float]:
        return self._best_score

    @property
    def best_epoch(self) -> Optional[int]:
        return self._best_epoch


def _checkpoint_epoch(path: Path) -> Optional[int]:
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    epoch = ckpt.get("epoch")
    if isinstance(epoch, int):
        return epoch
    return None


def _scan_epoch_sources(out_dir: Path) -> Dict[int, Path]:
    """Map training epoch -> checkpoint path for recoverable files in ``out_dir``."""
    sources: Dict[int, Path] = {}
    patterns = ("best_model.pt", "last_model.pt", "topk_*.pt")
    for pattern in patterns:
        for path in sorted(out_dir.glob(pattern)):
            epoch = _checkpoint_epoch(path)
            if epoch is None:
                continue
            sources.setdefault(epoch, path)
    archive_dir = out_dir / "topk_epochs"
    if archive_dir.is_dir():
        for path in sorted(archive_dir.glob("epoch_*.pt")):
            epoch = _checkpoint_epoch(path)
            if epoch is None:
                continue
            sources.setdefault(epoch, path)
    return sources


def recover_topk_slot_files(
    out_dir: Path,
    *,
    index_file: str = "checkpoint_index.json",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Rebuild ``topk_1.pt``..``topk_k.pt`` from ``checkpoint_index.json`` and on-disk weights."""
    out_dir = Path(out_dir)
    index_path = out_dir / index_file
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint index: {index_path}")

    data = json.loads(index_path.read_text(encoding="utf-8"))
    entries = list(data.get("checkpoints", []))
    if not entries:
        return {"restored": [], "missing": [], "out_dir": str(out_dir)}

    reverse = True
    monitor = "val_dice_mean"
    for row in entries:
        if "val_dice_mean" in row:
            monitor = "val_dice_mean"
            break
        if "val_loss" in row:
            monitor = "val_loss"
            reverse = False
            break

    def _score(row: Mapping[str, Any]) -> float:
        val = row.get(monitor)
        return float(val) if isinstance(val, (int, float)) else float("-inf" if reverse else "inf")

    ranked = sorted(entries, key=_score, reverse=reverse)
    sources = _scan_epoch_sources(out_dir)
    restored: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []

    for rank, row in enumerate(ranked, start=1):
        epoch = int(row["epoch"])
        target = out_dir / f"topk_{rank}.pt"
        src = sources.get(epoch)
        if src is None or not src.is_file():
            missing.append({"rank": rank, "epoch": epoch, "target": target.name})
            continue
        if src.resolve() == target.resolve():
            restored.append({"rank": rank, "epoch": epoch, "source": str(src), "target": target.name})
            continue
        if dry_run:
            restored.append({"rank": rank, "epoch": epoch, "source": str(src), "target": target.name})
            continue
        shutil.copy2(src, target)
        restored.append({"rank": rank, "epoch": epoch, "source": str(src), "target": target.name})

    return {"restored": restored, "missing": missing, "out_dir": str(out_dir)}


def load_finetune_resume_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scaler: Any = None,
    ckpt_manager: Optional["FinetuneCheckpointManager"] = None,
) -> Tuple[int, int]:
    """Load model/optimizer/scaler from a finetune checkpoint.

    Returns ``(start_epoch, global_step)`` where ``start_epoch`` is the next
    0-indexed training-loop index (checkpoint stores completed 1-indexed epoch).
    """
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict") or ckpt.get("model")
    if not isinstance(sd, dict):
        raise RuntimeError(f"Resume checkpoint has no model weights: {path}")
    core = unwrap_module(model)
    core.load_state_dict(sd, strict=False)
    opt_sd = ckpt.get("optimizer_state_dict") or ckpt.get("optimizer")
    if isinstance(opt_sd, dict):
        optimizer.load_state_dict(opt_sd)
    scaler_sd = ckpt.get("scaler_state_dict")
    if scaler is not None and isinstance(scaler_sd, dict) and hasattr(scaler, "load_state_dict"):
        scaler.load_state_dict(scaler_sd)
    completed = int(ckpt.get("epoch", 0))
    # ep_done stored: next loop index equals completed epoch count
    start_epoch = max(0, completed)
    global_step = int(ckpt.get("global_step", 0))
    if ckpt_manager is not None:
        bm = ckpt.get("best_metric")
        if isinstance(bm, (int, float)) and math.isfinite(float(bm)):
            ckpt_manager._best_score = float(bm)
            ckpt_manager._best_epoch = completed
    return start_epoch, global_step


__all__ = [
    "FinetuneCheckpointConfig",
    "FinetuneCheckpointManager",
    "_resolve_monitor_key",
    "recover_topk_slot_files",
    "load_finetune_resume_checkpoint",
]
