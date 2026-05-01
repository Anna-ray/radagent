"""
radagent.utils.metrics
----------------------
Multi-label medical metrics, computed safely (handles classes that have
zero positives in a given batch/split).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score


def per_class_auc(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """[C] vector of AUCs; returns NaN for classes with <2 unique labels."""
    C = y_true.shape[1]
    out = np.full(C, np.nan, dtype=np.float64)
    for c in range(C):
        if len(np.unique(y_true[:, c])) >= 2:
            out[c] = roc_auc_score(y_true[:, c], y_prob[:, c])
    return out


def mean_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    aucs = per_class_auc(y_true, y_prob)
    return float(np.nanmean(aucs))


def find_optimal_thresholds(
    y_true: np.ndarray, y_prob: np.ndarray
) -> np.ndarray:
    """Per-class threshold maximizing F1 on the validation set.

    Critical for the downstream agent: the LLM consumes binary findings,
    so calibrated thresholds matter more than any single AUC point.
    """
    C = y_true.shape[1]
    thresholds = np.full(C, 0.5, dtype=np.float64)
    for c in range(C):
        if y_true[:, c].sum() == 0:
            continue
        prec, rec, thr = precision_recall_curve(y_true[:, c], y_prob[:, c])
        f1 = (2 * prec * rec) / np.clip(prec + rec, 1e-9, None)
        # thr has length len(prec)-1; align by indexing safely
        if len(thr) == 0:
            continue
        best_idx = int(np.nanargmax(f1[:-1]))
        thresholds[c] = float(thr[best_idx])
    return thresholds


def per_class_f1(
    y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    y_pred = (y_prob >= thresholds[None, :]).astype(np.int32)
    C = y_true.shape[1]
    out = np.zeros(C, dtype=np.float64)
    for c in range(C):
        if y_true[:, c].sum() == 0:
            out[c] = np.nan
        else:
            out[c] = f1_score(y_true[:, c], y_pred[:, c], zero_division=0)
    return out
