"""CXR classification metrics: accuracy, AUC, sensitivity, specificity, precision, recall, F1, confusion matrix."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> np.ndarray:
    m = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.flat, y_pred.flat):
        m[int(t), int(p)] += 1
    return m


def accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean())


def auc_binary(probs: np.ndarray, labels: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:  # noqa: BLE001
        return 0.0
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(roc_auc_score(labels, probs))


def precision_recall_f1(preds: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    from sklearn.metrics import f1_score, precision_score, recall_score

    p = float(precision_score(labels, preds, average="binary", zero_division=0))
    r = float(recall_score(labels, preds, average="binary", zero_division=0))
    f = float(f1_score(labels, preds, average="binary", zero_division=0))
    return p, r, f


def sensitivity_specificity(preds: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    t = (labels == 1).astype(np.float64)
    p = (preds == 1).astype(np.float64)
    tp = ((p * t) > 0).sum()
    fn = (((1 - p) * t) > 0).sum()
    fp = ((p * (1 - t)) > 0).sum()
    tn = (((1 - p) * (1 - t)) > 0).sum()
    se = tp / (tp + fn + 1e-6)
    sp = tn / (tn + fp + 1e-6)
    return float(se), float(sp)


__all__ = [
    "confusion_matrix",
    "accuracy",
    "auc_binary",
    "precision_recall_f1",
    "sensitivity_specificity",
]
