"""vcg/allocation_huber.py — bounded-influence (Huber) multiplicative VCG allocation.

Value function:
    v_k(d, theta^k) = prod_t k_c(d_t - theta^k_t)
where the per-step kernel is the truncated quadratic
    k_c(x) = 1 - min(x^2, c^2)

This caps the per-step penalty at c^2. A discriminator with a single
confidently-wrong report (|d_t - theta^k_t| >= c) contributes a constant
factor (1 - c^2) for that step — its overall reputation w^t_k cannot be
killed by one outlier, but its cross-step consistency still counts.

When c >= 1 (so the cap never binds for reports in [0, 1]), the mechanism
recovers the original multiplicative VCG. Smaller c is more outlier-robust.

Coordinate update at step t becomes a weighted mean over the ACTIVE SET:
    A_t(d) := {k : |d_t - theta^k_t| < c}
    d_t <- sum_{k in A_t} theta^k_t * w^t_k / sum_{k in A_t} w^t_k
If A_t = empty (no discriminator is within c of d_t), d_t is left unchanged.
The reputation weight w^t_k = prod_{t' != t} k_c(d_{t'} - theta^k_{t'}) is
computed over all t' != t (not restricted to active steps at t').

The full V(d) is concave in each d_t (piecewise quadratic with continuous
derivative at the active/inactive boundary), so Gauss-Seidel sweeps remain
monotone in V and converge unconditionally. The per-coordinate update
above maximises V conditional on the active set at d_t^old being preserved;
we add a per-coordinate revert if V actually decreases (a guard against
rare active-set transition cases).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class HuberAllocationResult:
    d: np.ndarray              # (T,) — final allocation
    w: np.ndarray              # (K, T) — final reputation weights
    V_history: np.ndarray      # V at each sweep boundary
    converged: bool
    c: float                   # truncation parameter used
    active_history: np.ndarray # (n_sweeps,) — # of (k, t) active per sweep


def _huber_factors(d: np.ndarray, theta: np.ndarray, c: float) -> np.ndarray:
    """k_c(d_t - theta^k_t) = 1 - min((d_t - theta^k_t)^2, c^2). Returns (K, T)."""
    sq = (d[None, :] - theta) ** 2
    return 1.0 - np.minimum(sq, c * c)


def social_welfare_huber(d: np.ndarray, theta: np.ndarray, c: float) -> float:
    """V(d) = sum_k prod_t k_c(d_t - theta^k_t)."""
    factors = _huber_factors(d, theta, c)
    return float(factors.prod(axis=1).sum())


def _per_disc_products_huber(d: np.ndarray, theta: np.ndarray, c: float) -> np.ndarray:
    """v_k(d, theta^k) = prod_t k_c(d_t - theta^k_t). Returns (K,)."""
    return _huber_factors(d, theta, c).prod(axis=1)


def compute_weights_huber(d: np.ndarray, theta: np.ndarray, c: float) -> np.ndarray:
    """w^t_k = prod_{t' != t} k_c(d_{t'} - theta^k_{t'}). Returns (K, T)."""
    factors = _huber_factors(d, theta, c)             # (K, T)
    v = factors.prod(axis=1, keepdims=True)           # (K, 1)
    safe = factors > 1e-30
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(safe, v / np.where(safe, factors, 1.0), 0.0)
    if not safe.all():
        for k, t in zip(*np.where(~safe)):
            mask = np.ones(theta.shape[1], dtype=bool)
            mask[t] = False
            w[k, t] = float(np.prod(factors[k, mask])) if mask.any() else 1.0
    return w


def solve_allocation_huber(
    theta: np.ndarray,
    *,
    c: float = 0.3,
    L: int = 100,
    tau: float = 1e-10,
    eps: float = 1e-6,
    d_init: Optional[np.ndarray] = None,
) -> HuberAllocationResult:
    """Gauss-Seidel solver for the Huber-kernel multiplicative VCG allocation.

    Args:
        theta: (K, T) discriminator reports, each in [eps, 1-eps].
        c: Huber truncation. Smaller c -> more robust to outlier reports.
            c >= 1 (with reports in [0, 1]) recovers original VCG.
        L: max sweeps.
        tau: convergence tolerance on Delta V.
        eps: clipping range for d.
        d_init: optional warm-start (T,); defaults to simple mean across K.

    Returns:
        HuberAllocationResult with d, w, V_history, converged, c, active_history.
    """
    if c <= 0:
        raise ValueError(f"c must be positive; got {c}")
    theta = np.clip(np.asarray(theta, dtype=np.float64), eps, 1.0 - eps)
    K, T = theta.shape
    if d_init is None:
        d = theta.mean(axis=0)
    else:
        d = np.asarray(d_init, dtype=np.float64).copy()
    d = np.clip(d, eps, 1.0 - eps)

    V_hist = [social_welfare_huber(d, theta, c)]
    active_history = []
    converged = False

    for _ in range(L):
        V_old = V_hist[-1]
        n_active_sweep = 0
        for t in range(T):
            # Reputation weights at step t use the latest d for t' != t.
            mask = np.ones(T, dtype=bool)
            mask[t] = False
            factors_excl = _huber_factors(d[mask], theta[:, mask], c)  # (K, T-1)
            w_t = factors_excl.prod(axis=1)                            # (K,)

            # Active set at current d[t].
            dev = d[t] - theta[:, t]
            active = (dev * dev) < (c * c)
            n_active_sweep += int(active.sum())

            if active.any():
                w_t_a = w_t[active]
                theta_a = theta[active, t]
                den = float(w_t_a.sum())
                if den > 0:
                    d_new = float((theta_a * w_t_a).sum() / den)
                    d_new = float(np.clip(d_new, eps, 1.0 - eps))
                    # Per-coordinate monotonicity guard.
                    d_old_t = float(d[t])
                    d[t] = d_new
                    V_after = social_welfare_huber(d, theta, c)
                    if V_after < V_old - 1e-9:
                        d[t] = d_old_t
            # If A_t is empty, leave d[t] unchanged.

        active_history.append(n_active_sweep)
        V_new = social_welfare_huber(d, theta, c)
        if V_new < V_old - 1e-9:
            # Sweep-level monotonicity violated (should not happen with guard).
            break
        V_hist.append(V_new)
        if V_new - V_old < tau:
            converged = True
            break

    w = compute_weights_huber(d, theta, c)
    return HuberAllocationResult(
        d=d,
        w=w,
        V_history=np.array(V_hist),
        converged=converged,
        c=c,
        active_history=np.array(active_history),
    )