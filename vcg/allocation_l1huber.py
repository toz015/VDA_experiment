"""vcg/allocation_l1huber.py — L1-Huber multiplicative VCG allocation.

Value function:
    v_k(d, theta^k) = prod_t k_c(d_t - theta^k_t)
where the per-step kernel is the truncated linear
    k_c(x) = 1 - min(|x|, c)

This is the L1 analogue of allocation_huber (which uses 1 - min(x^2, c^2)).
The first-order condition for the coordinate update at step t becomes:

    sum_{k in A_t} w^t_k * sign(d_t - theta^k_t) = 0
    =>  d_t = WeightedMedian({theta^k_t}, weights {w^t_k}) over k in A_t

where A_t(d) = {k : |d_t - theta^k_t| < c} is the active set.

The cross-step reputation w^t_k = prod_{t' != t} k_c(d_{t'} - theta^k_{t'})
is unchanged in structure — the change is ONLY in how the K within-step
reports get combined for the local update: weighted median, not weighted mean.

Motivation: empirically median outperforms mean on Hand-Crafted Who&When
because LLM discriminators exhibit heavy-tail confidently-wrong errors.
Weighted-median update preserves the cross-step reputation machinery of
multiplicative VCG while making the local combination rule outlier-robust.

Kernel range: k_c(x) in [1-c, 1] for x in [-1, 1]. For c in (0, 1) the
kernel stays strictly positive, so reputation weights never collapse to zero.

Convergence: V is concave in each d_t (sum of piecewise-linear concave
functions). Coordinate-wise weighted median is the exact maximiser conditional
on the active set; a per-coordinate revert guards against rare active-set
transition cases where the conditional optimum doesn't increase global V.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class L1HuberAllocationResult:
    d: np.ndarray
    w: np.ndarray
    V_history: np.ndarray
    converged: bool
    c: float


def _l1huber_factors(d: np.ndarray, theta: np.ndarray, c: float) -> np.ndarray:
    """k_c(d_t - theta^k_t) = 1 - min(|d_t - theta^k_t|, c). Returns (K, T)."""
    return 1.0 - np.minimum(np.abs(d[None, :] - theta), c)


def social_welfare_l1huber(d: np.ndarray, theta: np.ndarray, c: float) -> float:
    return float(_l1huber_factors(d, theta, c).prod(axis=1).sum())


def compute_weights_l1huber(d: np.ndarray, theta: np.ndarray, c: float) -> np.ndarray:
    factors = _l1huber_factors(d, theta, c)
    v = factors.prod(axis=1, keepdims=True)
    safe = factors > 1e-30
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(safe, v / np.where(safe, factors, 1.0), 0.0)
    if not safe.all():
        for k, t in zip(*np.where(~safe)):
            mask = np.ones(theta.shape[1], dtype=bool)
            mask[t] = False
            w[k, t] = float(np.prod(factors[k, mask])) if mask.any() else 1.0
    return w


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median: smallest v such that cumulative weight reaches half of total."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return float("nan")
    if values.size == 1:
        return float(values[0])
    total = weights.sum()
    if total <= 0:
        return float(np.median(values))
    order = np.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    cumw = np.cumsum(w_sorted)
    target = total / 2.0
    idx = int(np.searchsorted(cumw, target))
    if idx >= v_sorted.size:
        return float(v_sorted[-1])
    # Interpolate when cumulative weight is exactly at half (well-defined median).
    if idx > 0 and abs(cumw[idx - 1] - target) < 1e-12:
        return 0.5 * (v_sorted[idx - 1] + v_sorted[idx])
    return float(v_sorted[idx])


def solve_allocation_l1huber(
    theta: np.ndarray,
    *,
    c: float = 0.3,
    L: int = 100,
    tau: float = 1e-10,
    eps: float = 1e-6,
    d_init: Optional[np.ndarray] = None,
) -> L1HuberAllocationResult:
    """L1-Huber multiplicative VCG allocation: weighted-median coordinate update."""
    if c <= 0:
        raise ValueError(f"c must be positive; got {c}")
    theta = np.clip(np.asarray(theta, dtype=np.float64), eps, 1.0 - eps)
    K, T = theta.shape
    if d_init is None:
        d = np.median(theta, axis=0).astype(np.float64)   # median init (matches the kernel)
    else:
        d = np.asarray(d_init, dtype=np.float64).copy()
    d = np.clip(d, eps, 1.0 - eps)

    V_hist = [social_welfare_l1huber(d, theta, c)]
    converged = False
    for _ in range(L):
        V_old = V_hist[-1]
        for t in range(T):
            mask = np.ones(T, dtype=bool)
            mask[t] = False
            factors_excl = _l1huber_factors(d[mask], theta[:, mask], c)   # (K, T-1)
            w_t = factors_excl.prod(axis=1)                               # (K,)

            # Active set at current d[t].
            dev = np.abs(d[t] - theta[:, t])
            active = dev < c

            if active.any():
                theta_a = theta[active, t]
                w_t_a = w_t[active]
                d_new = _weighted_median(theta_a, w_t_a)
                d_new = float(np.clip(d_new, eps, 1.0 - eps))
                # Per-coordinate monotonicity guard.
                d_old_t = float(d[t])
                d[t] = d_new
                V_after = social_welfare_l1huber(d, theta, c)
                if V_after < V_old - 1e-9:
                    d[t] = d_old_t
            # If no active discriminator at step t, leave d[t] unchanged.

        V_new = social_welfare_l1huber(d, theta, c)
        if V_new < V_old - 1e-9:
            break
        V_hist.append(V_new)
        if V_new - V_old < tau:
            converged = True
            break

    w = compute_weights_l1huber(d, theta, c)
    return L1HuberAllocationResult(
        d=d, w=w, V_history=np.array(V_hist),
        converged=converged, c=c,
    )