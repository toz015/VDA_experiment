"""Generalized power-kernel allocation solver.

Kernel:  kappa_{c, p}(x) = 1 - min(|x|^p, c^p),   p >= 1.

Special cases:
  p = 1                : L1-Huber (truncated absolute), our existing allocation_l1huber.
  p = 2, c large       : original VCG squared kernel.
  p = 2, c < 1         : truncated squared (Huber).
  p ∈ (1, 2)           : smoother than L1, more robust than L2.
                         Coordinate update is a weighted M-estimator.

Value function:
  v_k(d, theta^k)  =  prod_t kappa_{c, p}(d_t - theta^k_t)
  V(d)             =  sum_k v_k(d, theta^k).

Reputation:
  w_k^t(d)  =  v_k(d, theta^k) / kappa_{c, p}(d_t - theta^k_t).

Coordinate FOC at step t. Within the active set
  A_t(d) = { k : |d_t - theta^k_t| < c },
the partial derivative is
  d/d(d_t) V(d)  =  - sum_{k in A_t} w_k^t · p · |d_t - theta^k_t|^{p-1}
                                          · sgn(d_t - theta^k_t),
yielding the M-estimator FOC
  sum_{k in A_t} w_k^t · |d_t - theta^k_t|^{p-1} · sgn(d_t - theta^k_t)  =  0.

This is monotone non-decreasing in d_t (the LHS is a weighted sum of
non-decreasing functions of d_t), so we solve it by bisection on
[min_k theta_k_t, max_k theta_k_t]. Inactive discriminators (|deviation| >= c)
contribute zero to the FOC at this step, so we ignore them when solving;
the solver is monotone (V never decreases on accept).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PowerAllocationResult:
    d: np.ndarray
    w: np.ndarray
    V_history: np.ndarray
    converged: bool


def power_factors(d: np.ndarray, theta: np.ndarray, c: float, p: float) -> np.ndarray:
    """kappa_{c, p}(d - theta) elementwise. d: (T,); theta: (K, T) -> (K, T)."""
    return 1.0 - np.minimum(np.abs(d[None, :] - theta) ** p, c ** p)


def _per_disc_products(d: np.ndarray, theta: np.ndarray, c: float, p: float) -> np.ndarray:
    """v_k = prod_t kappa_{c, p}(d_t - theta^k_t), shape (K,)."""
    return power_factors(d, theta, c, p).prod(axis=1)


def social_welfare(d: np.ndarray, theta: np.ndarray, c: float, p: float) -> float:
    return float(_per_disc_products(d, theta, c, p).sum())


def _weighted_median(values, weights):
    """Exact weighted median (lower median on ties)."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if v.size == 0:
        return 0.5
    order = np.argsort(v)
    v_s = v[order]
    w_s = w[order]
    cumw = np.cumsum(w_s)
    total = cumw[-1] if cumw.size > 0 else 0.0
    if total <= 0:
        return float(np.median(v))
    idx = int(np.searchsorted(cumw, total / 2.0))
    if idx >= v_s.size:
        idx = v_s.size - 1
    return float(v_s[idx])


def _solve_coordinate_foc(
    theta_t: np.ndarray,       # (K,) reports at step t
    w_t: np.ndarray,           # (K,) reputation weights at step t
    c: float, p: float,
    *,
    n_iter: int = 30,
    tol: float = 1e-8,
) -> float:
    """Find d_t solving sum_k w_k * |d_t - theta_k|^{p-1} * sgn(d_t - theta_k) = 0,
    over the active set { k : |d_t - theta_k| < c }.

    Special cases:
      p = 1: weighted median (exact, closed form).
      p = 2: weighted mean over the active set (closed form).
      otherwise: bisection on the active-set FOC, which is monotone
                 non-decreasing in d_t.
    """
    if theta_t.size == 0:
        return 0.5

    # First identify the active set relative to the WEIGHTED CENTER of mass
    # (cheaper to do once than evaluate inside the bisection inner loop).
    # We use the median as a robust seed for the active-set membership test.
    seed = float(np.median(theta_t))
    active = np.abs(seed - theta_t) < c
    if not active.any():
        # Fall back to all reports if no one is in active range at the seed.
        active = np.ones_like(theta_t, dtype=bool)

    a_theta = theta_t[active]
    a_w = w_t[active]
    if a_theta.size == 0:
        return 0.5

    # Exact closed-form shortcuts.
    if abs(p - 1.0) < 1e-9:
        return _weighted_median(a_theta, a_w)
    if abs(p - 2.0) < 1e-9:
        s = a_w.sum()
        if s > 0:
            return float((a_w * a_theta).sum() / s)
        return float(a_theta.mean())

    # General p: bisection on the monotone FOC.
    lo, hi = float(a_theta.min()), float(a_theta.max())
    if hi - lo < tol:
        return float(0.5 * (lo + hi))

    def foc(d):
        diff = d - a_theta
        mag = np.maximum(np.abs(diff), 1e-12) ** (p - 1.0)
        return float((a_w * mag * np.sign(diff)).sum())

    f_lo, f_hi = foc(lo), foc(hi)
    if f_lo * f_hi > 0:
        return lo if abs(f_lo) <= abs(f_hi) else hi
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        f_mid = foc(mid)
        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def solve_allocation_power(
    theta: np.ndarray,
    *,
    c: float = 0.05,
    p: float = 1.0,
    L: int = 50,
    tau: float = 1e-10,
    eps: float = 1e-6,
    d_init: Optional[np.ndarray] = None,
) -> PowerAllocationResult:
    """Gauss-Seidel solver for the power-kernel multiplicative VCG allocation.

    Args:
        theta: (K, T) reports in [eps, 1-eps].
        c: kernel truncation; deviations >= c contribute 1 - c^p.
        p: kernel exponent (>= 1). p=1 recovers L1-Huber; p=2 recovers
           truncated squared.
        L: max sweeps.
        tau: convergence tolerance on Delta V.
        eps: clipping range for d.
        d_init: optional warm-start (T,); defaults to per-coordinate median.
    """
    theta = np.clip(np.asarray(theta, dtype=np.float64), eps, 1.0 - eps)
    K, T = theta.shape

    if d_init is None:
        d = np.median(theta, axis=0).copy()
    else:
        d = np.asarray(d_init, dtype=np.float64).copy()
    d = np.clip(d, eps, 1.0 - eps)

    V_hist = [social_welfare(d, theta, c, p)]
    converged = False

    for _ in range(L):
        V_old = V_hist[-1]
        for t in range(T):
            mask = np.ones(T, dtype=bool)
            mask[t] = False
            # w_k^t = prod_{t' != t} kappa(d_{t'} - theta_k_{t'}).
            factors_excl = power_factors(d[mask], theta[:, mask], c, p)
            w_t = factors_excl.prod(axis=1)
            # Solve M-estimator FOC.
            d_new = _solve_coordinate_foc(theta[:, t], w_t, c, p)
            d_new = float(np.clip(d_new, eps, 1.0 - eps))
            # Guard monotonicity: only accept if V does not decrease.
            d_old_t = float(d[t])
            d[t] = d_new
            V_after = social_welfare(d, theta, c, p)
            if V_after < V_old - 1e-9:
                d[t] = d_old_t

        V_new = social_welfare(d, theta, c, p)
        V_hist.append(V_new)
        if V_new - V_old < tau:
            converged = True
            break
        V_old = V_new

    # Final reputation.
    factors_full = power_factors(d, theta, c, p)
    v_full = factors_full.prod(axis=1, keepdims=True)
    safe = factors_full > 1e-30
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(safe, v_full / np.where(safe, factors_full, 1.0), 0.0)

    return PowerAllocationResult(
        d=d, w=w, V_history=np.array(V_hist), converged=converged,
    )