"""Core GCAM-to-LUCC quota reconciliation and spatial allocation solver.

Implements the quantitative reconciliation and competitive spatial allocation
mechanisms described in Sections 2.2 and 2.4 and Supporting Information Texts S5-S6:
1. Relative-change compositional reconciliation with pseudocount smoothing (Eq. 1 / S7-S8).
2. Largest-remainder integer rounding for strict GLU pixel closure.
3. Competitive utility scoring, shadow-price iteration, and minimum-utility-loss
   residual quota repair (Eq. 3 / S5-S6).
"""

from __future__ import annotations

import numpy as np


def reconcile_glu_quotas(
    baseline_counts: np.ndarray,
    gcam_base: np.ndarray,
    gcam_target: np.ndarray,
    total_pool: int,
    unused_floor: int = 0,
    pseudocount: float = 0.5,
) -> np.ndarray:
    """Reconcile GCAM physical-area trajectories to baseline pixel counts."""
    b = np.asarray(baseline_counts, dtype=np.float64)
    g0 = np.asarray(gcam_base, dtype=np.float64)
    gt = np.asarray(gcam_target, dtype=np.float64)

    ratio = (gt + pseudocount) / (g0 + pseudocount)
    raw_shares = (b + pseudocount) * ratio
    raw_shares /= raw_shares.sum()
    continuous = raw_shares * float(total_pool)

    # Enforce unused-land release floor on the last nonurban class (index -1)
    if continuous[-1] < unused_floor and continuous[:-1].sum() > 0:
        deficit = unused_floor - continuous[-1]
        continuous[-1] = float(unused_floor)
        continuous[:-1] *= max(0.0, (continuous[:-1].sum() - deficit) / continuous[:-1].sum())

    # Largest-remainder rounding for exact integer closure
    floored = np.floor(continuous).astype(np.int64)
    remainder = int(total_pool - floored.sum())
    if remainder > 0:
        order = np.argsort(-(continuous - floored))
        floored[order[:remainder]] += 1
    return floored


def allocate_with_shadow_prices(
    base_utility: np.ndarray,
    target_quotas: np.ndarray,
    max_iter: int = 200,
    step_scale: float = 1e-3,
) -> np.ndarray:
    """Assign cells to classes using iterative shadow prices and residual repair."""
    n_cells, n_classes = base_utility.shape
    targets = np.asarray(target_quotas, dtype=np.int64)
    if int(targets.sum()) != n_cells:
        raise ValueError("Target quotas must sum to the number of allocatable cells.")

    lam = np.zeros(n_classes, dtype=np.float64)
    best_assign = None
    best_gap = n_cells + 1

    for _ in range(max_iter):
        adj_utility = base_utility + lam[None, :]
        assign = np.argmax(adj_utility, axis=1)
        counts = np.bincount(assign, minlength=n_classes)
        diff = targets - counts
        gap = int(np.abs(diff).sum())
        if gap < best_gap:
            best_gap = gap
            best_assign = assign.copy()
        if gap == 0:
            return assign
        lam += step_scale * diff
        lam -= lam.mean()

    # Deterministic minimum-utility-loss residual repair
    assign = best_assign
    counts = np.bincount(assign, minlength=n_classes)
    surplus = np.where(counts > targets)[0]
    deficit = np.where(counts < targets)[0]

    while len(surplus) > 0 and len(deficit) > 0:
        best_cost = np.inf
        best_move = None
        for src in surplus:
            src_cells = np.where(assign == src)[0]
            for dst in deficit:
                losses = base_utility[src_cells, src] - base_utility[src_cells, dst]
                idx = int(np.argmin(losses))
                if losses[idx] < best_cost:
                    best_cost = float(losses[idx])
                    best_move = (int(src_cells[idx]), int(src), int(dst))
        cell, src, dst = best_move
        assign[cell] = dst
        counts[src] -= 1
        counts[dst] += 1
        surplus = np.where(counts > targets)[0]
        deficit = np.where(counts < targets)[0]

    return assign
