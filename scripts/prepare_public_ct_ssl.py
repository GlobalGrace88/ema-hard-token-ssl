#!/usr/bin/env python3
"""
Prepare public 3D CT datasets for SSL pretraining.

Unified output: processed NIfTI volumes + manifest_ssl_ct.csv.

Example:
  python tools/datasets/prepare_public_ct_ssl.py \\
    --config configs/datasets/public_ct_ssl.yaml \\
    --root /media/user/DATA/vibe/selfMiM/DATASET_SSL_CT \\
    --download --preprocess --make-manifest
"""
from __future__ import annotations

import argparse
import fnmatch
import gzip
import json
import logging
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

try:
    import nibabel as nib
except ImportError as exc:  # pragma: no cover
    raise SystemExit("nibabel is required: pip install nibabel") from exc

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pandas is required: pip install pandas") from exc

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(x, **kwargs):  # type: ignore[misc]
        return x


LOG = logging.getLogger("prepare_public_ct_ssl")

DATASET_RAW_DIRS: Dict[str, str] = {
    "abdomenct1k": "AbdomenCT-1K",
    "amos": "AMOS",
    "flare": "FLARE",
    "flare_task4_ct_fm": "FLARE-MedFM/FLARE-Task4-CT-FM",
    "totalsegmentator": "TotalSegmentator",
    "msd": "MSD",
}

OUTPUT_PREFIX: Dict[str, str] = {
    "abdomenct1k": "abdomenct1k",
    "amos": "amos",
    "flare": "flare",
    "flare_task4_ct_fm": "flare_task4",
    "totalsegmentator": "totalseg",
    "msd": "msd",
}

MSD_S3_BUCKET = "s3://medical-segmentation-decathlon"

MANUAL_DOWNLOAD_INSTRUCTIONS: Dict[str, str] = {
    "abdomenct1k": (
        "AbdomenCT-1K requires registration/agreement.\n"
        "  1. Register at https://abdomenct-1k.media.mit.edu/ (or official project page).\n"
        "  2. Download the dataset after approval.\n"
        "  3. Place zip(s) or extracted folder under:\n"
        "       {raw_dir}\n"
        "  4. Set datasets.abdomenct1k.local_archives or local_root in the YAML config."
    ),
    "amos": (
        "AMOS requires registration on the AMOS challenge site.\n"
        "  1. Register at https://amos22.grand-challenge.org/\n"
        "  2. Download CT volumes after agreement.\n"
        "  3. Place archives or extracted data under:\n"
        "       {raw_dir}\n"
        "  4. Set datasets.amos.local_archives or local_root in the YAML config."
    ),
    "flare": (
        "FLARE requires registration (MICCAI FLARE challenge).\n"
        "  1. Register via the official FLARE22/FLARE23 challenge pages.\n"
        "  2. Download after agreement.\n"
        "  3. Place data under:\n"
        "       {raw_dir}\n"
        "  4. Set datasets.flare.local_archives or local_root in the YAML config."
    ),
    "totalsegmentator": (
        "TotalSegmentator: download from the official release (Zenodo / project page).\n"
        "  https://github.com/wasserth/TotalSegmentator\n"
        "  Place archives or extracted cases under:\n"
        "       {raw_dir}\n"
        "  Or set datasets.totalsegmentator.download_url to a URL you obtained legally."
    ),
    "msd": (
        "Medical Segmentation Decathlon (MSD) CT tasks can be fetched via AWS Open Data:\n"
        "  aws s3 cp s3://medical-segmentation-decathlon/Task03_Liver.tar . --no-sign-request\n"
        "  Or enable datasets.msd.use_aws_open_data in config (requires AWS CLI).\n"
        "  Manual alternative: place Task*.tar or extracted Task folders under:\n"
        "       {raw_dir}\n"
    ),
}


@dataclass
class IndexedCase:
    dataset: str
    case_id: str
    image_path: Path
    label_path: Optional[Path] = None
    modality: str = "CT"
    anatomy: str = ""
    source_split: str = ""


@dataclass
class ProcessResult:
    ok: bool
    row: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


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


def sanitize_case_id(case_id: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(case_id).strip())
    return s[:120] if s else "case"


def _canonical_ssl_key(case: IndexedCase) -> str:
    """Normalize filenames for cross-folder dedup (e.g. FLARE labeled50 ⊂ unlabeled)."""
    stem = case.image_path.name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    elif stem.endswith(".nii"):
        stem = stem[:-4]
    key = stem.lower()
    key = re.sub(r"_0000$", "", key)
    key = re.sub(r"^flarets_", "flares_", key)
    key = re.sub(r"^flare22_tr_", "flare22tr_", key)
    return f"{case.dataset}:{key}"


def dedupe_cases(
    cases: List[IndexedCase],
    *,
    priority: Optional[List[str]] = None,
) -> Tuple[List[IndexedCase], List[Dict[str, str]]]:
    """Drop redundant volumes; keep first occurrence per canonical key (dataset priority order)."""
    order = {name: i for i, name in enumerate(priority or [])}

    def _rank(c: IndexedCase) -> Tuple[int, str]:
        return (order.get(c.dataset, 999), str(c.image_path))

    ranked = sorted(cases, key=_rank)
    seen: set = set()
    kept: List[IndexedCase] = []
    dropped: List[Dict[str, str]] = []
    for case in ranked:
        key = _canonical_ssl_key(case)
        if key in seen:
            dropped.append(
                {
                    "dataset": case.dataset,
                    "case_id": case.case_id,
                    "canonical_key": key,
                    "path": str(case.image_path),
                    "reason": "duplicate_canonical_key",
                }
            )
            continue
        seen.add(key)
        kept.append(case)
    return kept, dropped


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    fh = logging.FileHandler(log_dir / "prepare_public_ct_ssl.log", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg


def _is_label_path(path: Path) -> bool:
    s = str(path).lower().replace("\\", "/")
    markers = (
        "/labels",
        "/label",
        "/segmentations/",
        "/seg/",
        "labelsTr",
        "labelsVa",
        "labelsTs",
        "segmentation",
    )
    name = path.name.lower()
    if name.startswith("label") or name.startswith("seg") or name == "segmentations.nii.gz":
        return True
    return any(m in s for m in markers)


def _glob_match(path: Path, root: Path, patterns: Sequence[str]) -> bool:
    rel = str(path.relative_to(root)).replace("\\", "/")
    return any(fnmatch.fnmatch(rel, pat) for pat in patterns)


_SKIP_PATH_PARTS = ("__macosx", "/.extracted/", "/.git/")


def _should_skip_path(p: Path) -> bool:
    s = str(p).replace("\\", "/").lower()
    if any(part in s for part in _SKIP_PATH_PARTS):
        return True
    if p.name.startswith("._"):
        return True
    return False


def _iter_nifti(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return
    for p in sorted(root.rglob("*")):
        if _should_skip_path(p):
            continue
        if p.is_file() and p.suffix == ".gz" and p.name.endswith(".nii.gz"):
            yield p
        elif p.is_file() and p.suffix == ".nii":
            yield p


def _case_id_from_path(path: Path) -> str:
    stem = path.name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    elif stem.endswith(".nii"):
        stem = stem[:-4]
    return sanitize_case_id(stem)


def _find_label_for_image(image_path: Path, label_paths: List[Path]) -> Optional[Path]:
    img_id = _case_id_from_path(image_path)
    img_lower = image_path.name.lower()
    for lp in label_paths:
        lab_id = _case_id_from_path(lp)
        if lab_id == img_id:
            return lp
        if lab_id in img_lower or img_id in lp.name.lower():
            return lp
    parent = image_path.parent.name.lower()
    for lp in label_paths:
        if lp.parent.name.lower() == parent.replace("image", "label"):
            return lp
    return None


def _iter_scan_roots(name: str, ds_cfg: dict, raw_dir: Path) -> List[Path]:
    """Collect all roots to scan (primary local_root + additional_scan_roots)."""
    roots: List[Path] = []
    lr = ds_cfg.get("local_root")
    if lr:
        p = Path(str(lr)).expanduser().resolve()
        if p.is_dir() and p not in roots:
            roots.append(p)
    for extra in ds_cfg.get("additional_scan_roots") or []:
        p = Path(str(extra)).expanduser().resolve()
        if p.is_dir() and p not in roots:
            roots.append(p)
    if ds_cfg.get("huggingface_repo"):
        hf_dir = ds_cfg.get("huggingface_local_dir")
        if hf_dir:
            p = Path(str(hf_dir)).expanduser().resolve()
            if p.is_dir() and p not in roots:
                roots.append(p)
    if not roots and raw_dir.is_dir():
        roots.append(raw_dir)
    return roots


def _resolve_scan_root(name: str, ds_cfg: dict, raw_dir: Path) -> Path:
    roots = _iter_scan_roots(name, ds_cfg, raw_dir)
    return roots[0] if roots else raw_dir


def _is_flare_image_candidate(path: Path) -> bool:
    """Fast path filter: FLARE pseudo-label masks live under dedicated label folders."""
    s = str(path).replace("\\", "/").lower()
    if "pseudo" in s or "/labels/" in s or "labelsva" in s or "labelstr" in s:
        return False
    return True


def _detect_amos_modality(path: Path, raw_root: Path) -> str:
    s = str(path).lower().replace("\\", "/")
    if "mri_unlabeled" in s or "/amos_mri" in s or "unlabeled_mri" in s:
        return "MR"
    if "unlabeled_ct" in s or "unlabeled_ct_" in s or "unlabled_part" in s:
        return "CT"
    if "/ct/" in s or "_ct" in s or "ct_" in path.name.lower():
        return "CT"
    if "/mr/" in s or "_mr" in s or "mr_" in path.name.lower():
        return "MR"
    for js in raw_root.rglob("dataset.json"):
        try:
            meta = json.loads(js.read_text(encoding="utf-8"))
            name = path.name
            for key in ("training", "validation", "test"):
                for ent in meta.get(key, []) or []:
                    img = str(ent.get("image", ""))
                    if img.endswith(path.name) or path.name in img:
                        mod = str(ent.get("modality", ent.get("mod", ""))).upper()
                        if mod:
                            return mod
        except Exception:  # noqa: BLE001
            continue
    return "CT_unknown_source"


def _index_scan_root(
    name: str,
    ds_cfg: dict,
    scan_root: Path,
    *,
    image_globs: List[str],
    label_globs: List[str],
) -> List[IndexedCase]:
    if not scan_root.is_dir():
        LOG.warning("[%s] scan root missing: %s", name, scan_root)
        return []
    LOG.info("[%s] indexing from %s", name, scan_root)

    all_nii = list(_iter_nifti(scan_root))
    label_candidates = [p for p in all_nii if _glob_match(p, scan_root, label_globs) or _is_label_path(p)]
    label_set = set(label_candidates)

    images: List[Path] = []
    for p in all_nii:
        if p in label_set:
            continue
        if _is_label_path(p):
            continue
        if _glob_match(p, scan_root, image_globs):
            images.append(p)

    cases: List[IndexedCase] = []
    ct_only = bool(ds_cfg.get("ct_only", False))
    for img in images:
        if name == "flare" and not _is_flare_image_candidate(img):
            continue
        mod = "CT"
        if name == "amos":
            mod = _detect_amos_modality(img, scan_root)
            if ct_only and mod == "MR":
                continue
        split = ""
        for tag in ("imagesTr", "imagesVa", "imagesTs", "images", "unlabeled", "pseudolabel"):
            if tag in str(img):
                split = tag
                break
        lab = _find_label_for_image(img, label_candidates)
        anatomy = ""
        if name == "msd":
            for part in img.parts:
                if part.startswith("Task"):
                    anatomy = part.replace("Task", "").split("_", 1)[-1]
        cases.append(
            IndexedCase(
                dataset=name,
                case_id=_case_id_from_path(img),
                image_path=img,
                label_path=lab,
                modality=mod,
                anatomy=anatomy,
                source_split=split,
            )
        )
    return cases


def index_dataset(name: str, ds_cfg: dict, raw_dir: Path) -> List[IndexedCase]:
    if not ds_cfg.get("enabled", True):
        return []

    image_globs = list(ds_cfg.get("image_globs") or ["**/*.nii.gz"])
    label_globs = list(ds_cfg.get("label_globs") or [])

    cases: List[IndexedCase] = []
    seen_ids: set = set()
    for scan_root in _iter_scan_roots(name, ds_cfg, raw_dir):
        root_cases = _index_scan_root(
            name, ds_cfg, scan_root, image_globs=image_globs, label_globs=label_globs
        )
        for case in root_cases:
            key = (case.dataset, case.case_id)
            if key in seen_ids:
                continue
            seen_ids.add(key)
            cases.append(case)

    max_cases = ds_cfg.get("max_cases_per_dataset")
    if max_cases is not None:
        cases = cases[: int(max_cases)]
    LOG.info("[%s] indexed %d image volumes (raw scan)", name, len(cases))
    return cases


def _extract_archive(archive: Path, dest: Path, *, force: bool = False) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".extracted" / archive.name
    if marker.is_file() and not force:
        LOG.info("Skip extract (exists): %s -> %s", archive, dest)
        return
    LOG.info("Extracting %s -> %s", archive, dest)
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dest)
    elif name.endswith(".tar.gz") or name.endswith(".tgz") or name.endswith(".tar"):
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(dest)
    elif name.endswith(".gz") and not name.endswith(".nii.gz"):
        out = dest / archive.stem
        with gzip.open(archive, "rb") as fin, open(out, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    else:
        shutil.copy2(archive, dest / archive.name)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok\n", encoding="utf-8")


def _stage_local_root(local_root: Path, dest: Path, *, force: bool = False) -> None:
    if not local_root.exists():
        LOG.warning("local_root not found: %s", local_root)
        return
    if local_root.is_file():
        _extract_archive(local_root, dest, force=force)
        return
    if dest.resolve() == local_root.resolve():
        return
    if any(dest.iterdir()) and not force:
        LOG.info("Skip copy local_root (dest non-empty): %s", dest)
        return
    LOG.info("Copy local_root %s -> %s", local_root, dest)
    shutil.copytree(local_root, dest, dirs_exist_ok=True)


def _aws_available() -> bool:
    try:
        r = subprocess.run(["aws", "--version"], capture_output=True, check=False, timeout=10)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def _download_huggingface_dataset(
    repo_id: str,
    local_dir: Path,
    *,
    force: bool = False,
    include_patterns: Optional[List[str]] = None,
) -> None:
    local_dir = local_dir.expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    marker = local_dir / ".hf_download_complete"
    if marker.is_file() and not force and any(local_dir.rglob("*.nii.gz")):
        LOG.info("[huggingface] skip download (exists): %s", local_dir)
        return
    LOG.info("[huggingface] downloading %s -> %s", repo_id, local_dir)
    if include_patterns:
        LOG.info("[huggingface] include patterns: %s", include_patterns)
    try:
        from huggingface_hub import snapshot_download

        kwargs: Dict[str, Any] = {
            "repo_id": repo_id,
            "repo_type": "dataset",
            "local_dir": str(local_dir),
            "local_dir_use_symlinks": False,
        }
        if include_patterns:
            kwargs["allow_patterns"] = list(include_patterns)
        snapshot_download(**kwargs)
        marker.write_text("ok\n", encoding="utf-8")
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required: pip install -U huggingface_hub"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        err_name = type(exc).__name__
        if err_name == "GatedRepoError" or "GatedRepoError" in str(exc):
            raise RuntimeError(
                f"HuggingFace dataset {repo_id} is gated. Log in (hf auth login), accept terms at "
                f"https://huggingface.co/datasets/{repo_id}, then retry."
            ) from exc
        hf_bin = shutil.which("hf") or "hf"
        cmd = [
            hf_bin,
            "download",
            repo_id,
            "--repo-type",
            "dataset",
            "--local-dir",
            str(local_dir),
        ]
        for pat in include_patterns or []:
            cmd.extend(["--include", pat])
        LOG.warning("[huggingface] snapshot_download failed (%s); trying CLI", exc)
        r = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"HuggingFace download failed for {repo_id}: {r.stderr.strip()}") from exc
        marker.write_text("ok\n", encoding="utf-8")


def _download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        LOG.info("Skip download (exists): %s", dest)
        return
    LOG.info("Downloading %s -> %s", url, dest)
    try:
        import urllib.request

        urllib.request.urlretrieve(url, dest)  # noqa: S310
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Download failed: {url}") from exc


def download_msd_task(task: str, raw_dir: Path, *, force: bool = False) -> None:
    task_dir = raw_dir / task
    if task_dir.is_dir() and any(task_dir.rglob("*.nii.gz")) and not force:
        LOG.info("[msd] %s already present", task)
        return
    tar_name = f"{task}.tar"
    tar_path = raw_dir / tar_name
    if _aws_available():
        cmd = ["aws", "s3", "cp", f"{MSD_S3_BUCKET}/{tar_name}", str(tar_path), "--no-sign-request"]
        LOG.info("[msd] %s", " ".join(cmd))
        r = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if r.returncode != 0:
            LOG.warning("[msd] AWS download failed for %s: %s", task, r.stderr.strip())
        elif tar_path.is_file():
            _extract_archive(tar_path, raw_dir, force=force)
            return
    if tar_path.is_file():
        _extract_archive(tar_path, raw_dir, force=force)


def download_dataset(name: str, ds_cfg: dict, raw_dir: Path, *, force: bool = False) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    manual = bool(ds_cfg.get("manual_required", True))

    for arc in ds_cfg.get("local_archives") or []:
        p = Path(str(arc)).expanduser()
        if p.is_file():
            _extract_archive(p, raw_dir, force=force)
        elif p.is_dir():
            _stage_local_root(p, raw_dir, force=force)
        else:
            LOG.warning("[%s] local_archives path not found: %s", name, p)

    local_root = ds_cfg.get("local_root")
    if local_root:
        lr = Path(str(local_root)).expanduser()
        # When indexing in-place, only extract archives; do not copy the whole tree.
        if ds_cfg.get("index_from_local_root", True):
            if lr.is_file():
                _extract_archive(lr, raw_dir, force=force)
        else:
            _stage_local_root(lr, raw_dir, force=force)

    if name == "msd" and ds_cfg.get("use_aws_open_data", False):
        for task in ds_cfg.get("tasks") or []:
            download_msd_task(str(task), raw_dir, force=force)

    url = ds_cfg.get("download_url")
    if url:
        dest = raw_dir / Path(str(url)).name
        _download_url(str(url), dest)
        if dest.is_file():
            _extract_archive(dest, raw_dir, force=force)

    hf_repo = ds_cfg.get("huggingface_repo")
    if hf_repo:
        hf_dir = Path(
            str(
                ds_cfg.get("huggingface_local_dir")
                or raw_dir.parent / "FLARE-MedFM" / "FLARE-Task3-DomainAdaption"
            )
        ).expanduser()
        include = list(ds_cfg.get("huggingface_include") or [])
        try:
            _download_huggingface_dataset(str(hf_repo), hf_dir, force=force, include_patterns=include or None)
        except RuntimeError as exc:
            LOG.warning("[%s] HuggingFace download skipped: %s", name, exc)
            LOG.warning(
                "[%s] Use tools/datasets/download_flare_unlabeled_hf.py after `hf auth login --force`, "
                "or bulk zips from https://flare22.grand-challenge.org/Dataset/",
                name,
            )
        copy_to = ds_cfg.get("huggingface_copy_images_to") or ds_cfg.get("huggingface_copy_to")
        if copy_to and hf_dir.is_dir():
            dest = Path(str(copy_to)).expanduser().resolve()
            dest.mkdir(parents=True, exist_ok=True)
            n = 0
            for src in hf_dir.rglob("*.nii.gz"):
                if "imagesTr" not in str(src):
                    continue
                out = dest / src.name
                if out.is_file() and not force:
                    continue
                shutil.copy2(src, out)
                n += 1
            if n:
                LOG.info("[%s] copied %d CT images from HF staging -> %s", name, n, dest)

    has_nifti = False
    for scan_root in _iter_scan_roots(name, ds_cfg, raw_dir):
        if any(scan_root.rglob("*.nii.gz")) or any(scan_root.rglob("*.nii")):
            has_nifti = True
            break
    if manual and not has_nifti:
        instr = MANUAL_DOWNLOAD_INSTRUCTIONS.get(name, "Place data under {raw_dir}").format(raw_dir=raw_dir)
        LOG.warning("[%s] manual download required:\n%s", name, instr)


def _sitk_available() -> bool:
    try:
        import SimpleITK  # noqa: F401

        return True
    except ImportError:
        return False


def _reorient_ras_nib(data: np.ndarray, affine: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    img = nib.Nifti1Image(data, affine)
    reoriented = nib.as_closest_canonical(img)
    return np.asanyarray(reoriented.dataobj), reoriented.affine


def _read_volume(path: Path, pp: dict) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    if _sitk_available():
        import SimpleITK as sitk

        img = sitk.ReadImage(str(path))
        spacing = tuple(float(x) for x in img.GetSpacing()[::-1])  # sitk xyz -> dhw approx
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        if pp.get("reorient_to_ras", True):
            img = sitk.DICOMOrient(img, "RAS")
            arr = sitk.GetArrayFromImage(img).astype(np.float32)
            spacing = tuple(float(x) for x in img.GetSpacing()[::-1])
        # SimpleITK array is z,y,x; treat as D,H,W
        affine = np.eye(4, dtype=np.float64)
        return arr, affine, (spacing[0], spacing[1], spacing[2])

    data, affine = load_nifti_array(path)
    data = np.asarray(data, dtype=np.float32)
    if pp.get("reorient_to_ras", True):
        data, affine = _reorient_ras_nib(data, affine)
    zooms = nib.load(str(path)).header.get_zooms()[:3]
    spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    return data, affine, spacing


def load_nifti_array(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    nii = nib.load(str(path))
    return np.asanyarray(nii.dataobj, dtype=np.float32), nii.affine


def _select_channel(data: np.ndarray, pp: dict) -> np.ndarray:
    if data.ndim == 4:
        if not pp.get("allow_4d_nifti", True):
            raise ValueError(f"4D NIfTI not allowed: {data.shape}")
        if pp.get("select_first_channel_if_4d", True):
            return data[..., 0] if data.shape[-1] <= data.shape[0] else data[0]
        return data[0]
    if data.ndim == 3:
        return data
    raise ValueError(f"Unsupported ndim={data.ndim} shape={data.shape}")


def _clip_normalize(data: np.ndarray, pp: dict) -> np.ndarray:
    lo = float(pp.get("ct_clip_min", -175))
    hi = float(pp.get("ct_clip_max", 250))
    out = np.clip(data, lo, hi)
    mode = str(pp.get("normalize", "zero_mean_unit_std_after_clip")).lower()
    if mode in ("none", "off"):
        return out.astype(np.float32)
    if mode == "minmax_after_clip":
        denom = max(hi - lo, 1e-6)
        return ((out - lo) / denom).astype(np.float32)
    if mode in ("zero_mean_unit_std_after_clip", "zscore"):
        mu = float(out.mean())
        sd = float(out.std())
        if sd < 1e-6:
            sd = 1.0
        return ((out - mu) / sd).astype(np.float32)
    raise ValueError(f"Unknown normalize mode: {mode!r}")


def _resample_sitk(data: np.ndarray, spacing: Tuple[float, float, float], out_spacing: Sequence[float]) -> np.ndarray:
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(data.astype(np.float32))
    img.SetSpacing((float(spacing[2]), float(spacing[1]), float(spacing[0])))
    new_sp = (float(out_spacing[2]), float(out_spacing[1]), float(out_spacing[0]))
    resampled = sitk.Resample(
        img,
        sitk.Cast(img, sitk.sitkFloat32),
        sitk.Transform(),
        sitk.sitkLinear,
        img.GetOrigin(),
        new_sp,
        img.GetDirection(),
        0.0,
        img.GetPixelID(),
    )
    return sitk.GetArrayFromImage(resampled).astype(np.float32)


def _save_nifti(path: Path, data: np.ndarray, affine: np.ndarray, dtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dt = np.float32 if dtype == "float32" else np.float32
    img = nib.Nifti1Image(np.ascontiguousarray(data.astype(dt)), affine)
    nib.save(img, str(path))


def _preview_png(path: Path, data: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    vol = data
    if vol.ndim == 4:
        vol = vol[0]
    if vol.ndim != 3:
        return
    z = vol.shape[0] // 2
    sl = vol[z]
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(sl, cmap="gray")
    plt.axis("off")
    plt.title(path.stem)
    plt.tight_layout()
    plt.savefig(str(path), dpi=80, bbox_inches="tight")
    plt.close()


def output_name(case: IndexedCase, msd_task: str = "") -> str:
    prefix = OUTPUT_PREFIX.get(case.dataset, case.dataset)
    cid = sanitize_case_id(case.case_id)
    if case.dataset == "msd" and msd_task:
        task_slug = msd_task.replace("Task", "").split("_", 1)[-1].lower()
        return f"{prefix}_{task_slug}_{cid}.nii.gz"
    return f"{prefix}_{cid}.nii.gz"


def _msd_task_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("Task"):
            return part
    return ""


def preprocess_case(
    case: IndexedCase,
    *,
    out_images: Path,
    out_preview: Path,
    pp: dict,
    root: Path,
) -> ProcessResult:
    try:
        out_name = output_name(case, _msd_task_from_path(case.image_path))
        out_path = out_images / out_name
        if out_path.is_file() and pp.get("skip_if_exists", True):
            data, affine, spacing = _read_volume(out_path, pp)
            vol = _select_channel(data, pp)
            row = _manifest_row_from_array(case, out_path, vol, spacing, root, label_saved=None)
            return ProcessResult(ok=True, row=row)

        data, affine, spacing = _read_volume(case.image_path, pp)
        vol = _select_channel(data, pp)
        if vol.size == 0:
            return ProcessResult(ok=False, error="empty volume")
        if not np.isfinite(vol).all():
            return ProcessResult(ok=False, error="NaN/Inf in volume")
        if min(vol.shape) < 16:
            LOG.warning("[%s/%s] small shape %s", case.dataset, case.case_id, vol.shape)

        out_sp = pp.get("output_spacing")
        if out_sp and _sitk_available():
            vol = _resample_sitk(vol, spacing, out_sp)

        vol = _clip_normalize(vol, pp)
        _save_nifti(out_path, vol, affine, str(pp.get("save_dtype", "float32")))

        label_saved = None
        if case.label_path and case.label_path.is_file():
            label_out = out_images / out_name.replace(".nii.gz", "_label.nii.gz")
            if not label_out.is_file() or not pp.get("skip_if_exists", True):
                ld, la, lsp = _read_volume(case.label_path, pp)
                ld = _select_channel(ld, pp)
                if out_sp and _sitk_available():
                    ld = _resample_sitk(ld, lsp, out_sp)
                _save_nifti(label_out, ld.astype(np.int16), la, "float32")
            label_saved = label_out

        if pp.get("create_preview_png", True):
            _preview_png(out_preview / out_name.replace(".nii.gz", ".png"), vol)

        row = _manifest_row_from_array(case, out_path, vol, spacing, root, label_saved)
        return ProcessResult(ok=True, row=row)
    except Exception as exc:  # noqa: BLE001
        return ProcessResult(ok=False, error=str(exc))


def _manifest_row_from_array(
    case: IndexedCase,
    out_path: Path,
    vol: np.ndarray,
    spacing: Tuple[float, float, float],
    root: Path,
    label_saved: Optional[Path],
) -> Dict[str, Any]:
    rel = out_path.resolve()
    try:
        rel = out_path.resolve().relative_to(root.resolve())
    except ValueError:
        rel = out_path.resolve()
    lab_rel = ""
    if label_saved and label_saved.is_file():
        try:
            lab_rel = str(label_saved.resolve().relative_to(root.resolve()))
        except ValueError:
            lab_rel = str(label_saved.resolve())
    d, h, w = (int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2]))
    return {
        "image_path": str(rel),
        "dataset": case.dataset,
        "case_id": case.case_id,
        "modality": case.modality,
        "anatomy": case.anatomy or _msd_task_from_path(case.image_path).replace("Task", ""),
        "source_split": case.source_split,
        "has_label": bool(lab_rel),
        "label_path": lab_rel,
        "spacing_x": spacing[0],
        "spacing_y": spacing[1],
        "spacing_z": spacing[2],
        "shape_x": d,
        "shape_y": h,
        "shape_z": w,
        "intensity_min": float(vol.min()),
        "intensity_max": float(vol.max()),
        "intensity_mean": float(vol.mean()),
        "intensity_std": float(vol.std()),
    }


MANIFEST_COLUMNS = [
    "image_path",
    "dataset",
    "case_id",
    "modality",
    "anatomy",
    "source_split",
    "has_label",
    "label_path",
    "spacing_x",
    "spacing_y",
    "spacing_z",
    "shape_x",
    "shape_y",
    "shape_z",
    "intensity_min",
    "intensity_max",
    "intensity_mean",
    "intensity_std",
]


def run_preprocess(
    cases: List[IndexedCase],
    *,
    root: Path,
    pp: dict,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    out_images = root / "processed" / "images"
    out_preview = root / "processed" / "preview"
    out_images.mkdir(parents=True, exist_ok=True)
    out_preview.mkdir(parents=True, exist_ok=True)

    max_cases = pp.get("max_cases_per_dataset")
    if max_cases is not None:
        cases = cases[: int(max_cases)]

    rows: List[Dict[str, Any]] = []
    failed: List[Dict[str, str]] = []
    workers = max(1, int(pp.get("num_workers", 1)))

    def _job(c: IndexedCase) -> ProcessResult:
        return preprocess_case(c, out_images=out_images, out_preview=out_preview, pp=pp, root=root)

    if workers <= 1:
        it = tqdm(cases, desc="preprocess")
        for c in it:
            r = _job(c)
            if r.ok:
                rows.append(r.row)
            else:
                failed.append({"dataset": c.dataset, "case_id": c.case_id, "path": str(c.image_path), "error": r.error})
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_job, c): c for c in cases}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="preprocess"):
                c = futs[fut]
                r = fut.result()
                if r.ok:
                    rows.append(r.row)
                else:
                    failed.append({"dataset": c.dataset, "case_id": c.case_id, "path": str(c.image_path), "error": r.error})
    return rows, failed


def write_manifest(rows: List[Dict[str, Any]], path: Path, *, verify: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if verify:
        ok_mask = []
        for p in df["image_path"]:
            fp = Path(str(p))
            if not fp.is_absolute():
                fp = (path.parent.parent / fp).resolve()
            ok_mask.append(fp.is_file())
        df = df[np.array(ok_mask, dtype=bool)]
    df.to_csv(path, index=False)
    LOG.info("Wrote manifest: %s (%d rows)", path, len(df))


def write_failed(failed: List[Dict[str, str]], path: Path) -> None:
    if not failed:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failed).to_csv(path, index=False)
    LOG.info("Wrote failed cases: %s (%d rows)", path, len(failed))


def write_summary(rows: List[Dict[str, Any]], indexed: Dict[str, int], failed: List[Dict[str, str]], path: Path) -> None:
    proc_counts: Dict[str, int] = {}
    for r in rows:
        proc_counts[r["dataset"]] = proc_counts.get(r["dataset"], 0) + 1
    fail_counts: Dict[str, int] = {}
    for f in failed:
        fail_counts[f["dataset"]] = fail_counts.get(f["dataset"], 0) + 1
    names = sorted(set(indexed) | set(proc_counts) | set(fail_counts))
    out = []
    for n in names:
        out.append(
            {
                "dataset": n,
                "raw_indexed": indexed.get(n, 0),
                "processed": proc_counts.get(n, 0),
                "failed": fail_counts.get(n, 0),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(path, index=False)
    LOG.info("Wrote dataset summary: %s", path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare public 3D CT datasets for SSL pretraining")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--root", type=str, default=None, help="Override config root")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--preprocess", action="store_true")
    ap.add_argument("--make-manifest", action="store_true")
    ap.add_argument("--force", action="store_true", help="Re-extract archives and reprocess existing outputs")
    args = ap.parse_args()

    if not (args.download or args.preprocess or args.make_manifest):
        ap.error("Specify at least one of --download, --preprocess, --make-manifest")

    cfg = load_config(Path(args.config).expanduser().resolve())
    root = Path(args.root or cfg.get("root") or ".").expanduser().resolve()
    setup_logging(root / "logs")

    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "processed" / "images").mkdir(parents=True, exist_ok=True)
    (root / "processed" / "preview").mkdir(parents=True, exist_ok=True)

    ds_cfg_all = cfg.get("datasets") or {}
    pp = dict(cfg.get("preprocess") or {})
    manifest_cfg = dict(cfg.get("manifest") or {})

    if args.download:
        LOG.info("=== DOWNLOAD / STAGE ===")
        for name, ds_cfg in ds_cfg_all.items():
            if not ds_cfg.get("enabled", True):
                continue
            raw_name = DATASET_RAW_DIRS.get(name, name)
            raw_dir = root / "raw" / raw_name
            download_dataset(name, ds_cfg, raw_dir, force=args.force)

    all_cases: List[IndexedCase] = []
    indexed_counts: Dict[str, int] = {}
    LOG.info("=== INDEX RAW ===")
    for name, ds_cfg in ds_cfg_all.items():
        if not ds_cfg.get("enabled", True):
            continue
        raw_name = DATASET_RAW_DIRS.get(name, name)
        raw_dir = root / "raw" / raw_name
        cases = index_dataset(name, ds_cfg, raw_dir)
        indexed_counts[name] = len(cases)
        all_cases.extend(cases)

    dedupe_priority = list(cfg.get("dedupe_dataset_priority") or ["amos", "flare", "msd", "abdomenct1k", "totalsegmentator"])
    all_cases, dropped = dedupe_cases(all_cases, priority=dedupe_priority)
    if dropped:
        drop_path = root / "logs" / "deduped_cases.csv"
        drop_path.parent.mkdir(parents=True, exist_ok=True)
        import csv

        with open(drop_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=["dataset", "case_id", "canonical_key", "path", "reason"],
            )
            w.writeheader()
            w.writerows(dropped)
        LOG.info("Deduped %d redundant cases -> %s", len(dropped), drop_path)
    LOG.info("Total unique cases after dedup: %d", len(all_cases))

    manifest_rows: List[Dict[str, Any]] = []
    failed_rows: List[Dict[str, str]] = []

    if args.preprocess:
        LOG.info("=== PREPROCESS (%d cases) ===", len(all_cases))
        manifest_rows, failed_rows = run_preprocess(all_cases, root=root, pp=pp)
        write_failed(failed_rows, root / "logs" / "failed_cases.csv")

    manifest_path = root / "processed" / str(manifest_cfg.get("filename", "manifest_ssl_ct.csv"))

    if args.make_manifest:
        LOG.info("=== MANIFEST ===")
        if not manifest_rows and manifest_path.is_file() and not args.preprocess:
            LOG.info("Loading existing processed files into manifest from index...")
            for case in tqdm(all_cases, desc="manifest-scan"):
                out_name = output_name(case, _msd_task_from_path(case.image_path))
                out_path = root / "processed" / "images" / out_name
                if not out_path.is_file():
                    continue
                try:
                    data, _, spacing = _read_volume(out_path, pp)
                    vol = _select_channel(data, pp)
                    manifest_rows.append(
                        _manifest_row_from_array(case, out_path, vol, spacing, root, label_saved=None)
                    )
                except Exception as exc:  # noqa: BLE001
                    failed_rows.append(
                        {"dataset": case.dataset, "case_id": case.case_id, "path": str(out_path), "error": str(exc)}
                    )
        write_manifest(
            manifest_rows,
            manifest_path,
            verify=bool(manifest_cfg.get("verify_readable", True)),
        )

    write_summary(manifest_rows, indexed_counts, failed_rows, root / "logs" / "dataset_summary.csv")

    LOG.info("=== SUMMARY ===")
    for name, n in sorted(indexed_counts.items()):
        proc = sum(1 for r in manifest_rows if r.get("dataset") == name)
        fail = sum(1 for f in failed_rows if f.get("dataset") == name)
        LOG.info("  %s: raw=%d processed=%d failed=%d", name, n, proc, fail)
    LOG.info("Manifest: %s", manifest_path if manifest_path.is_file() else "(not written)")


if __name__ == "__main__":
    main()
