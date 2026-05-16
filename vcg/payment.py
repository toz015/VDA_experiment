"""Algorithm 2: VCG payment computation (Section 5.2).

Pi_k = S_{-k}(d*_{-k}) - S_{-k}(d*),  S_{-k}(d) = sum_{j!=k} v_j(d, theta_hat^j).

For each k we re-solve the allocation problem with D_k removed (leave-one-out),
warm-started from the full-set d*; the difference between the two values of
"others' welfare" is the informational contribution of D_k.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from .allocation import AllocationResult, solve_allocation, _per_disc_products


@dataclass
class PaymentResult:
    payments: np.ndarray              # (K,) — Pi_k, clipped to >= 0
    raw_payments: np.ndarray          # (K,) — Pi_k before clipping (for diagnostics)
    leave_one_out: List[AllocationResult]   # K solver outputs (no entry for each k)


def compute_vcg_payments(
    theta: np.ndarray,
    full_alloc: AllocationResult,
    *,
    L: int = 100,
    tau: float = 1e-10,
    eps: float = 1e-6,
) -> PaymentResult:
    """Compute Pi_k for each discriminator.

    Args:
        theta: (K, T) reports.
        full_alloc: result of solve_allocation on the full theta (provides warm-start d*).
        L, tau, eps: passed to leave-one-out solver.

    Returns PaymentResult with payments (length K), raw payments, and the LOO solves.
    """
    theta = np.asarray(theta, dtype=np.float64)
    K, T = theta.shape
    d_full = full_alloc.d
    # v_j(d*, theta^j) for all j — used in every "others' welfare at d*" sum.
    v_full = _per_disc_products(d_full, theta)         # (K,)

    raw = np.zeros(K, dtype=np.float64)
    loo_results: List[AllocationResult] = []
    for k in range(K):
        # Drop discriminator k.
        mask = np.ones(K, dtype=bool)
        mask[k] = False
        theta_minus_k = theta[mask]                    # (K-1, T)

        # Warm-start from d* (full).
        res_mk = solve_allocation(
            theta_minus_k, L=L, tau=tau, eps=eps, d_init=d_full
        )
        loo_results.append(res_mk)

        # "Others' welfare" at d*_{-k}: sum_{j != k} v_j(d*_{-k}, theta_hat^j).
        v_minus_at_dmk = _per_disc_products(res_mk.d, theta_minus_k).sum()
        # "Others' welfare" at d*: sum_{j != k} v_j(d*, theta_hat^j).
        v_minus_at_dfull = float(v_full[mask].sum())
        raw[k] = v_minus_at_dmk - v_minus_at_dfull

    # Theory: Pi_k >= 0. Numerical slack: clip negatives to zero.
    assert (raw >= -1e-6).all(), f"VCG payment negative beyond tolerance: {raw}"
    payments = np.maximum(raw, 0.0)
    return PaymentResult(payments=payments, raw_payments=raw, leave_one_out=loo_results)
