"""
radagent.utils.bootstrap
------------------------
Bootstrap confidence intervals for per-class multi-label metrics.

Why this matters: a "mean AUC = 0.832" line in your submission is fine,
but a "mean AUC = 0.832 [0.825, 0.839]" line is *publishable*. CIs let
reviewers compare your number to literature numbers honestly.

We resample at the *image* level (not per-class), preserving the
correlation structure between classes — the right way to bootstrap
multi-label problems.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)


def _safe_auc(y_true_c: np.ndarray, y_prob_c: np.ndarray) -> float:
    if len(np.unique(y_true_c)) < 2:
        return float("nan")
    return roc_auc_score(y_true_c, y_prob_c)


def _safe_ap(y_true_c: np.ndarray, y_prob_c: np.ndarray) -> float:
    if y_true_c.sum() == 0:
        return float("nan")
    return average_precision_score(y_true_c, y_prob_c)


def _safe_f1(y_true_c: np.ndarray, y_pred_c: np.ndarray) -> float:
    if y_true_c.sum() == 0:
        return float("nan")
    return f1_score(y_true_c, y_pred_c, zero_division=0)


def per_class_metrics_with_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    """
    y_true:     [N, C]
    y_prob:     [N, C]
    thresholds: [C]
    Returns a dict with point estimates + bootstrap CI for AUC, AP, F1,
    sensitivity, specificity per class, and macro/micro AUC.
    """
    assert y_true.shape == y_prob.shape
    N, C = y_true.shape
    y_pred = (y_prob >= thresholds[None, :]).astype(np.int32)

    # ---- point estimates ----
    auc = np.array([_safe_auc(y_true[:, c], y_prob[:, c]) for c in range(C)])
    ap  = np.array([_safe_ap (y_true[:, c], y_prob[:, c]) for c in range(C)])
    f1  = np.array([_safe_f1 (y_true[:, c], y_pred[:, c]) for c in range(C)])
    # sens = TP / (TP + FN), spec = TN / (TN + FP)
    sens = np.array([
        (y_pred[:, c][y_true[:, c] == 1].sum() / max(1, (y_true[:, c] == 1).sum()))
        if (y_true[:, c] == 1).sum() > 0 else float("nan")
        for c in range(C)
    ])
    spec = np.array([
        ((1 - y_pred[:, c])[y_true[:, c] == 0].sum() / max(1, (y_true[:, c] == 0).sum()))
        if (y_true[:, c] == 0).sum() > 0 else float("nan")
        for c in range(C)
    ])

    # micro-AUC (treats every (sample, class) pair as one binary problem)
    micro_auc_point = _safe_auc(y_true.ravel(), y_prob.ravel())
    macro_auc_point = float(np.nanmean(auc))

    # ---- bootstrap ----
    rng = np.random.default_rng(seed)
    boot_auc = np.zeros((n_bootstrap, C), dtype=np.float64)
    boot_ap  = np.zeros((n_bootstrap, C), dtype=np.float64)
    boot_f1  = np.zeros((n_bootstrap, C), dtype=np.float64)
    boot_macro_auc = np.zeros(n_bootstrap, dtype=np.float64)
    boot_micro_auc = np.zeros(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        idx = rng.integers(0, N, size=N)
        yt = y_true[idx]
        yp = y_prob[idx]
        ypd = y_pred[idx]
        per_c_auc = np.array([_safe_auc(yt[:, c], yp[:, c]) for c in range(C)])
        per_c_ap  = np.array([_safe_ap (yt[:, c], yp[:, c]) for c in range(C)])
        per_c_f1  = np.array([_safe_f1 (yt[:, c], ypd[:, c]) for c in range(C)])
        boot_auc[b] = per_c_auc
        boot_ap[b]  = per_c_ap
        boot_f1[b]  = per_c_f1
        boot_macro_auc[b] = np.nanmean(per_c_auc)
        boot_micro_auc[b] = _safe_auc(yt.ravel(), yp.ravel())

    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q

    def _ci(arr: np.ndarray) -> tuple[float, float]:
        if np.all(np.isnan(arr)):
            return float("nan"), float("nan")
        return (
            float(np.nanpercentile(arr, lo_q * 100)),
            float(np.nanpercentile(arr, hi_q * 100)),
        )

    auc_ci = [_ci(boot_auc[:, c]) for c in range(C)]
    ap_ci  = [_ci(boot_ap[:, c])  for c in range(C)]
    f1_ci  = [_ci(boot_f1[:, c])  for c in range(C)]
    macro_auc_ci = _ci(boot_macro_auc)
    micro_auc_ci = _ci(boot_micro_auc)

    return {
        "n_samples": int(N),
        "n_classes": int(C),
        "n_bootstrap": int(n_bootstrap),
        "ci_level": float(ci),
        "per_class": {
            "auc":      auc.tolist(),
            "auc_ci":   auc_ci,
            "ap":       ap.tolist(),
            "ap_ci":    ap_ci,
            "f1":       f1.tolist(),
            "f1_ci":    f1_ci,
            "sens":     sens.tolist(),
            "spec":     spec.tolist(),
            "thresholds": thresholds.tolist(),
        },
        "macro_auc":     float(macro_auc_point),
        "macro_auc_ci":  list(macro_auc_ci),
        "micro_auc":     float(micro_auc_point),
        "micro_auc_ci":  list(micro_auc_ci),
        "mean_f1":       float(np.nanmean(f1)),
        "mean_ap":       float(np.nanmean(ap)),
    }
