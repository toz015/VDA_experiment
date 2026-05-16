"""Algorithm 1: Gauss-Seidel allocation solver (Section 5.1 of implementation note).

Solves d* = argmax V(d), where V(d) = sum_k prod_t [1 - (d_t - theta_hat^k_t)^2].

The fixed-point condition is d_t = (sum_k theta_hat^k_t * w^t_k) / (sum_k w^t_k),
with w^t_k = prod_{t'!=t} [1 - (d_{t'} - theta_hat^k_{t'})^2] (cross-step reputation).
Gauss-Seidel sweeps update one t at a time using the latest d for other coords;
this is guaranteed monotone in V and converges unconditionally.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class AllocationResult:
    d: np.ndarray            # (T,) — final allocation
    w: np.ndarray            # (K, T) — final reputation weights w^t_k
    V_history: np.ndarray    # (n_sweeps+1,) — V at each sweep boundary (V[0] = init)
    converged: bool          # True if dV < tau before hitting L


def social_welfare(d: np.ndarray, theta: np.ndarray) -> float:
    """V(d) = sum_k prod_t [1 - (d_t - theta^k_t)^2].

    theta: (K, T) report matrix; d: (T,) allocation.
    """
    factors = 1.0 - (d[None, :] - theta) ** 2     # (K, T)
    return float(factors.prod(axis=1).sum())


def _per_disc_products(d: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Per-discriminator full product v_k(d, theta^k) = prod_t [1 - (d_t - theta^k_t)^2].

    Returns shape (K,).
    """
    factors = 1.0 - (d[None, :] - theta) ** 2
    return factors.prod(axis=1)


def compute_weights(d: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """w^t_k = prod_{t' != t} [1 - (d_{t'} - theta^k_{t'})^2], for all (k, t).

    Returns (K, T). Uses w^t_k = v_k / [1 - (d_t - theta^k_t)^2] when the denom
    is safely non-zero; falls back to explicit leave-one-out product otherwise.
    """
    factors = 1.0 - (d[None, :] - theta) ** 2          # (K, T)
    v = factors.prod(axis=1, keepdims=True)            # (K, 1)
    safe = factors > 1e-30
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(safe, v / np.where(safe, factors, 1.0), 0.0)
    # For degenerate columns (factor ~ 0), recompute explicitly.
    if not safe.all():
        for k, t in zip(*np.where(~safe)):
            mask = np.ones(theta.shape[1], dtype=bool)
            mask[t] = False
            w[k, t] = float(np.prod(factors[k, mask])) if mask.any() else 1.0
    return w


def solve_allocation(
    theta: np.ndarray,
    *,
    L: int = 100,
    tau: float = 1e-10,
    eps: float = 1e-6,
    d_init: Optional[np.ndarray] = None,
) -> AllocationResult:
    """Gauss-Seidel solver for the multiplicative VCG allocation.

    Args:
        theta: (K, T) discriminator reports, each in [eps, 1-eps].
        L: max sweeps.
        tau: convergence tolerance on Delta V.
        eps: clipping range for d.
        d_init: optional warm-start (T,); defaults to simple mean across K.

    Returns AllocationResult with d, w, V_history, converged.
    """
    theta = np.clip(np.asarray(theta, dtype=np.float64), eps, 1.0 - eps)
    K, T = theta.shape
    if d_init is None:
        d = theta.mean(axis=0)
    else:
        d = np.asarray(d_init, dtype=np.float64).copy()
    d = np.clip(d, eps, 1.0 - eps)

    V_hist = [social_welfare(d, theta)]
    converged = False
    for _ in range(L):
        V_old = V_hist[-1]
        for t in range(T):
            # w^t_k uses the latest d values for t' != t (Gauss-Seidel, in-place).
            mask = np.ones(T, dtype=bool)
            mask[t] = False
            factors_excl = 1.0 - (d[mask][None, :] - theta[:, mask]) ** 2  # (K, T-1)
            w_t = factors_excl.prod(axis=1)                                # (K,)
            num = float((theta[:, t] * w_t).sum())
            den = float(w_t.sum())
            if den > 0:
                d[t] = num / den
            d[t] = float(np.clip(d[t], eps, 1.0 - eps))
        V_new = social_welfare(d, theta)
        # Monotonicity assertion (allow tiny float slack).
        assert V_new >= V_old - 1e-9, (
            f"V decreased: V_old={V_old}, V_new={V_new}. Implementation bug."
        )
        V_hist.append(V_new)
        if V_new - V_old < tau:
            converged = True
            break

    w = compute_weights(d, theta)
    return AllocationResult(d=d, w=w, V_history=np.array(V_hist), converged=converged)
