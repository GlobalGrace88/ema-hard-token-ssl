"""nnFormer / UNETR++ ``splits_final.pkl`` train/val case IDs."""

from __future__ import annotations

import pickle
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_PATIENT_ID_RE = re.compile(r"^(patient\d+)", re.I)


def patient_id_from_case_id(case_id: str) -> str:
    """Map ``patient004_frame01`` -> ``patient004``."""
    s = str(case_id).strip()
    m = _PATIENT_ID_RE.match(s)
    if m:
        return m.group(1).lower()
    if "_frame" in s.lower():
        return s.split("_frame")[0].split("_Frame")[0].lower()
    return s.lower()


def restrict_case_ids_to_num_patients(
    case_ids: Sequence[str],
    num_patients: int,
    *,
    seed: int = 42,
) -> List[str]:
    """
    Keep all frames/volumes for ``num_patients`` randomly sampled subjects.

    When ``num_patients`` exceeds the number of unique patients, all patients are kept.
    """
    if num_patients <= 0:
        return sorted(str(c) for c in case_ids)
    by_patient: Dict[str, List[str]] = {}
    for cid in case_ids:
        pid = patient_id_from_case_id(cid)
        by_patient.setdefault(pid, []).append(str(cid))
    patients = sorted(by_patient.keys())
    n = min(int(num_patients), len(patients))
    rng = random.Random(int(seed))
    chosen = sorted(rng.sample(patients, n))
    out: List[str] = []
    for pid in chosen:
        out.extend(sorted(by_patient[pid]))
    return sorted(out)


def finetune_num_patients_requested(data_cfg: Dict) -> Optional[int]:
    raw = data_cfg.get("finetune_num_patients", data_cfg.get("finetune_num_subjects"))
    if raw is None:
        return None
    return int(raw)


def apply_finetune_patient_cap(case_ids: List[str], data_cfg: Dict) -> List[str]:
    """Apply ``data.finetune_num_patients`` (train split only)."""
    m = finetune_num_patients_requested(data_cfg)
    if m is None:
        return case_ids
    seed = int(data_cfg.get("finetune_patient_seed", data_cfg.get("seed", 42)))
    return restrict_case_ids_to_num_patients(case_ids, m, seed=seed)


def load_case_ids_from_txt(path: str | Path) -> List[str]:
    """One case id per line (``#`` comments and blank lines ignored)."""
    path = Path(path)
    if not path.is_file():
        return []
    ids: List[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s and not s.startswith("#"):
                ids.append(s)
    return ids


def resolve_explicit_train_case_ids(data_cfg: Dict) -> Optional[List[str]]:
    """Train-case override from YAML list or ``train_case_ids_file``."""
    explicit = data_cfg.get("nnformer_train_cases")
    if explicit:
        return [str(x) for x in explicit]
    fpath = data_cfg.get("train_case_ids_file")
    if fpath:
        ids = load_case_ids_from_txt(fpath)
        if ids:
            return ids
    return None


def load_splits_final(pkl_path: str | Path, fold: int = 0) -> Tuple[List[str], List[str]]:
    pkl_path = Path(pkl_path)
    if not pkl_path.is_file():
        raise FileNotFoundError(f"splits_final.pkl not found: {pkl_path}")
    with open(pkl_path, "rb") as fh:
        splits = pickle.load(fh)
    if not isinstance(splits, list) or not splits:
        raise ValueError(f"Expected non-empty list of fold dicts in {pkl_path}")
    fold = int(fold)
    if fold < 0 or fold >= len(splits):
        raise IndexError(f"fold {fold} out of range (0..{len(splits) - 1})")
    entry = splits[fold]
    train_ids = list(entry.get("train", entry.get("training", [])))
    val_ids = list(entry.get("val", entry.get("validation", [])))
    if not train_ids and not val_ids:
        raise ValueError(f"Fold {fold} has empty train and val in {pkl_path}")
    return train_ids, val_ids


def resolve_splits_pkl(data_cfg: Dict, task: str = "") -> Optional[Path]:
    explicit = data_cfg.get("splits_pkl") or data_cfg.get("nnformer_splits_pkl")
    if explicit:
        p = Path(str(explicit)).expanduser()
        return p if p.is_file() else None

    for key in ("nnformer_preprocessed_dir", "preprocessed_dir", "nnformer_npz_dir", "npz_dir"):
        raw = data_cfg.get(key)
        if not raw:
            continue
        root = Path(str(raw)).expanduser().resolve()
        for cand in (root / "splits_final.pkl", root.parent / "splits_final.pkl"):
            if cand.is_file():
                return cand

    task_l = (task or str(data_cfg.get("dataset_name", ""))).lower()
    task_globs = {
        "synapse": ["*Synapse*", "*synapse*"],
        "tumor": ["*tumor*", "*Tumor*", "*Brain*"],
        "acdc": ["*ACDC*", "*acdc*"],
        "lung": ["*Lung*", "*lung*"],
    }
    for key in ("nnformer_preprocessed_dir", "preprocessed_dir"):
        raw = data_cfg.get(key)
        if not raw:
            continue
        root = Path(str(raw)).expanduser().resolve()
        for pat in task_globs.get(task_l, []):
            for hit in root.parent.glob(pat):
                cand = hit / "splits_final.pkl"
                if cand.is_file():
                    return cand
    return None


def resolve_npz_case_ids_for_split(
    data_cfg: Dict,
    *,
    train: bool,
    npz_dir: Path,
) -> List[str]:
    """Case IDs for train or val from ``splits_final.pkl`` or explicit lists."""
    task = str(data_cfg.get("dataset_name", data_cfg.get("task", ""))).lower()
    fold = int(data_cfg.get("fold", data_cfg.get("nnformer_fold", 0)))
    pkl = resolve_splits_pkl(data_cfg, task)
    if pkl is not None:
        tr, va = load_splits_final(pkl, fold=fold)
        case_ids = list(va if not train else tr)
        if train:
            explicit_tr = resolve_explicit_train_case_ids(data_cfg)
            if explicit_tr:
                want = set(explicit_tr)
                case_ids = [c for c in case_ids if c in want]
            case_ids = apply_finetune_patient_cap(case_ids, data_cfg)
        return case_ids

    key = "nnformer_val_cases" if not train else "nnformer_train_cases"
    explicit = data_cfg.get(key)
    if explicit:
        case_ids = [str(x) for x in explicit]
        if train:
            case_ids = apply_finetune_patient_cap(case_ids, data_cfg)
        return case_ids

    from dinomim_pytorch.datasets.nnformer_npz import case_id_from_npz, list_npz_files, split_npz_paths_train_val

    all_paths = list_npz_files(npz_dir)
    paths = split_npz_paths_train_val(all_paths, data_cfg, train=train)
    case_ids = [case_id_from_npz(p) for p in paths]
    if train:
        case_ids = apply_finetune_patient_cap(case_ids, data_cfg)
    return case_ids


__all__ = [
    "apply_finetune_patient_cap",
    "finetune_num_patients_requested",
    "load_case_ids_from_txt",
    "load_splits_final",
    "patient_id_from_case_id",
    "resolve_explicit_train_case_ids",
    "resolve_splits_pkl",
    "resolve_npz_case_ids_for_split",
    "restrict_case_ids_to_num_patients",
]
