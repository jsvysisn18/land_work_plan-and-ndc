"""Core two-stage Random Forest transition probability model.

Implements the two-stage classification and probability calibration framework
described in Section 2.4 and Supporting Information Text S4:
1. Stage 1: Binary change probability among nonurban systems.
2. Stage 2: Conditional destination-system probability on changed pixels.
3. Calibration: Prior-odds correction, Platt scaling, or isotonic regression
   selected via inner spatial-block cross-validation.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupKFold


DEFAULT_RF_PARAMS = {
    "n_estimators": 160,
    "max_depth": 18,
    "min_samples_leaf": 5,
    "class_weight": "balanced_subsample",
    "max_features": "sqrt",
    "random_state": 20350803,
    "n_jobs": -1,
}


def build_rf_classifier(**overrides) -> RandomForestClassifier:
    """Instantiate a Random Forest classifier with study hyperparameters."""
    params = dict(DEFAULT_RF_PARAMS)
    params.update(overrides)
    return RandomForestClassifier(**params)


def prior_odds_correction(
    p_sample: np.ndarray,
    pi_true: float,
    pi_sample: float,
    eps: float = 1e-7,
) -> np.ndarray:
    """Apply prior-odds correction to sampled binary change probabilities."""
    p = np.clip(np.asarray(p_sample, dtype=np.float64), eps, 1.0 - eps)
    pi_t = np.clip(pi_true, eps, 1.0 - eps)
    pi_s = np.clip(pi_sample, eps, 1.0 - eps)
    logit_p = np.log(p / (1.0 - p))
    logit_adj = logit_p + np.log(pi_t / (1.0 - pi_t)) - np.log(pi_s / (1.0 - pi_s))
    return 1.0 / (1.0 + np.exp(-logit_adj))


def fit_and_select_calibrator(
    p_raw: np.ndarray,
    y_true: np.ndarray,
    groups: np.ndarray,
    sample_weight: np.ndarray | None = None,
    n_splits: int = 3,
):
    """Select probability calibration method via spatial-block inner CV."""
    gkf = GroupKFold(n_splits=n_splits)
    candidates = ("prior_odds", "platt", "isotonic")
    losses = {name: [] for name in candidates}

    pi_true = float(np.average(y_true, weights=sample_weight))
    pi_sample = float(np.mean(y_true))

    for tr_idx, va_idx in gkf.split(p_raw, y_true, groups):
        p_tr, y_tr = p_raw[tr_idx], y_true[tr_idx]
        p_va, y_va = p_raw[va_idx], y_true[va_idx]
        w_tr = None if sample_weight is None else sample_weight[tr_idx]
        w_va = None if sample_weight is None else sample_weight[va_idx]

        # 1. Prior-odds correction
        cal_prior = prior_odds_correction(p_va, pi_true=pi_true, pi_sample=pi_sample)
        losses["prior_odds"].append(brier_score_loss(y_va, cal_prior, sample_weight=w_va))

        # 2. Platt scaling
        lr = LogisticRegression(solver="lbfgs")
        lr.fit(p_tr.reshape(-1, 1), y_tr, sample_weight=w_tr)
        cal_platt = lr.predict_proba(p_va.reshape(-1, 1))[:, 1]
        losses["platt"].append(brier_score_loss(y_va, cal_platt, sample_weight=w_va))

        # 3. Isotonic regression
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_tr, y_tr, sample_weight=w_tr)
        cal_iso = iso.predict(p_va)
        losses["isotonic"].append(brier_score_loss(y_va, cal_iso, sample_weight=w_va))

    mean_losses = {k: float(np.mean(v)) for k, v in losses.items()}
    best_method = min(candidates, key=lambda k: mean_losses[k])
    return best_method, mean_losses


def renormalize_destination_probabilities(
    p_change: np.ndarray,
    p_dest: np.ndarray,
    current_class: np.ndarray,
    allowed_mask: np.ndarray,
) -> np.ndarray:
    """Compute stay and masked-renormalized transition probabilities (Eq. S4)."""
    n_cells, n_classes = p_dest.shape
    p_eff = np.where(allowed_mask, p_dest, 0.0)
    idx = np.arange(n_cells)
    p_eff[idx, current_class] = 0.0

    row_sums = p_eff.sum(axis=1, keepdims=True)
    p_cond = np.divide(p_eff, row_sums, out=np.zeros_like(p_eff), where=row_sums > 0)

    probs = p_change[:, None] * p_cond
    probs[idx, current_class] = 1.0 - p_change
    return probs
